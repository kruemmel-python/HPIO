"""
Streamlit-Seite für die Drohnensimulation.
Bindet die erweiterte :class:`~dronesim.DroneSwarm`-Logik ein und bietet
vollständige Kontrolle über Rollen, adaptive Gewichte und Umgebungsparameter.
"""

from __future__ import annotations

import io
import json
import time
from typing import Any, Dict

import imageio
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from dronesim import DroneSwarm

try:
    from streamlit_app import AppState
except ImportError:  # pragma: no cover - Fallback für isolierte Tests
    AppState = Any


# Farbpalette für die Visualisierung der Rollen
ROLE_COLORS = {
    "generalist": "#41b6c4",
    "scout": "#7fb800",
    "harvester": "#f95d6a",
}


def _serialize_config(cfg: Dict[str, Any]) -> str:
    """Hilfsfunktion, um verschachtelte Dicts deterministisch zu vergleichen."""

    def _convert(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return [_convert(v) for v in value]
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in sorted(value.items())}
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return float(value)
        return value

    return json.dumps(_convert(cfg), sort_keys=True, ensure_ascii=False)


def _normalize_mix(scout: float, harvester: float) -> Dict[str, float]:
    """Normiert die Rollengewichte und stellt sicher, dass Generalisten übrig bleiben."""

    scout = max(0.0, float(scout))
    harvester = max(0.0, float(harvester))
    remainder = max(0.0, 1.0 - (scout + harvester))
    if scout + harvester > 1.0:
        total = scout + harvester
        scout /= total
        harvester /= total
        remainder = 0.0
    mix = {
        "generalist": remainder,
        "scout": scout,
        "harvester": harvester,
    }
    return mix


def _role_config_from_inputs(
    generalist: Dict[str, float],
    scout: Dict[str, float],
    harvester: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """Fasst die Rollenparameter in einem Dictionary zusammen."""

    return {
        "generalist": generalist,
        "scout": scout,
        "harvester": harvester,
    }


def _ensure_role_battery(sw: DroneSwarm, role_cfg: Dict[str, Dict[str, float]]) -> None:
    """Synchronisiert Batteriekapazitäten der existierenden Drohnen mit dem UI-Setup."""

    for drone in sw.drones:
        cfg = role_cfg.get(drone.role, role_cfg["generalist"])
        capacity = float(cfg.get("battery_capacity", sw.battery_capacity))
        drone.battery_capacity = capacity
        drone.battery = min(drone.battery, capacity)


def page_dronesim(state: AppState) -> None:
    """Zeigt die Drohnen-Simulation im Streamlit-Frontend."""

    st.markdown("## 🛸 Drohnen-Schwarm-Simulation")
    st.caption(
        "Bio-inspirierter Schwarm mit k-Nachbarschaftskohäsion, prädiktiver Vermeidung, "
        "Rollenlogik und Batterie-Management auf dem HPIO-Feld Φ."
    )

    # --- Φ-Quelle priorisieren -------------------------------------------------
    if "phi_snapshot" in st.session_state:
        phi_src = st.session_state["phi_snapshot"]
    elif state.controller and hasattr(state.controller.optimizer, "field"):
        phi_src = state.controller.optimizer.field.phi
    else:
        st.warning("Kein HPIO-Feld aktiv. Standardfeld wird verwendet.")
        phi_src = np.zeros((160, 160), dtype=np.float32)

    phi = np.array(phi_src, dtype=np.float32, copy=True)
    height, width = phi.shape

    # --- Sidebar: vollständige Parameterkontrolle -----------------------------
    with st.sidebar:
        st.markdown("### Schwarmparameter")
        num_drones = st.slider("Anzahl Drohnen", 6, 240, 52, 2)
        step = st.slider("Schrittweite", 0.1, 5.0, 1.2, 0.05)
        inertia = st.slider("Trägheit", 0.0, 0.95, 0.68, 0.01)
        avoidance_radius = st.slider("Vermeidungsradius", 0.0, 25.0, 6.0, 0.5)
        cohesion_neighbors = st.slider("Kohäsions-Nachbarn", 1, 24, 6, 1,
                                        help="Anzahl der Nachbarn für lokale Kohäsion")
        boundary_mode = st.selectbox(
            "Randbedingungen",
            options=["reflect", "soft", "periodic"],
            index=0,
            help="Modus für den Raumabschluss der Simulation",
        )

        st.markdown("### Basis & Energie")
        base_x = st.number_input("Basis X", 0.0, float(width - 1), float(width / 2), 1.0)
        base_y = st.number_input("Basis Y", 0.0, float(height - 1), float(height / 2), 1.0)
        generalist_battery = st.slider("Batterie-Kapazität (Generalist)", 100.0, 2000.0, 600.0, 10.0)

        st.markdown("### Rollenverteilung")
        scout_mix = st.slider("Scout-Anteil", 0.0, 1.0, 0.2, 0.05)
        harvester_mix = st.slider("Harvester-Anteil", 0.0, 1.0, 0.2, 0.05)
        role_mix = _normalize_mix(scout_mix, harvester_mix)
        st.caption(
            f"Generalistenanteil wird automatisch auf {role_mix['generalist']:.2f} normalisiert."
        )

        st.markdown("### Feld-Dynamik")
        evap = st.slider("Verdunstung Φ", 0.0, 0.2, 0.02, 0.005)
        relax_alpha = st.slider("Relaxation Φ", 0.0, 1.0, 0.25, 0.01)
        run_steps = st.number_input("Simulationsschritte", 1, 1500, 200, 1)

        st.markdown("### Rollen-Feintuning")
        with st.expander("Gains & Ablage nach Rolle", expanded=False):
            st.markdown("**Generalisten**")
            gen_curiosity = st.slider("Neugier (Generalist)", 0.0, 1.5, 0.20, 0.01)
            gen_coherence = st.slider("Kohärenz (Generalist)", 0.0, 1.0, 0.15, 0.01)
            gen_avoid = st.slider("Vermeidung (Generalist)", 0.0, 2.5, 0.40, 0.05)
            gen_deposit = st.slider("Ablage-Sigma (Generalist)", 0.3, 4.0, 1.6, 0.1)

            st.markdown("**Scouts**")
            scout_curiosity = st.slider("Neugier (Scout)", 0.0, 2.0, 0.55, 0.01)
            scout_coherence = st.slider("Kohärenz (Scout)", 0.0, 1.0, 0.08, 0.01)
            scout_avoid = st.slider("Vermeidung (Scout)", 0.0, 2.5, 0.35, 0.05)
            scout_deposit = st.slider("Ablage-Sigma (Scout)", 0.3, 4.0, 0.8, 0.1)
            scout_battery = st.slider("Batterie (Scout)", 50.0, 2000.0, 510.0, 10.0)

            st.markdown("**Harvester**")
            harvester_curiosity = st.slider("Neugier (Harvester)", 0.0, 1.5, 0.15, 0.01)
            harvester_coherence = st.slider("Kohärenz (Harvester)", 0.0, 2.5, 0.26, 0.01)
            harvester_avoid = st.slider("Vermeidung (Harvester)", 0.0, 2.5, 0.45, 0.05)
            harvester_deposit = st.slider("Ablage-Sigma (Harvester)", 0.3, 6.0, 2.9, 0.1)
            harvester_battery = st.slider("Batterie (Harvester)", 100.0, 2500.0, 720.0, 10.0)

        st.markdown("### Φ-Livekopplung")
        live_coupling = st.checkbox("HPIO-Φ live einkoppeln", value=True)
        blend = st.slider(
            "Blend-Faktor (neu ↔ alt)",
            0.0,
            1.0,
            0.25,
            0.05,
            help="0.0 = nur Swarm-Feld, 1.0 = nur aktueller Φ-Snapshot",
        )

        st.markdown("### Steuerung")
        start_btn = st.button("Simulation initialisieren")
        record_run = st.checkbox("Simulation aufzeichnen (GIF)", value=False)
        record_fps = st.slider("GIF-Bildrate", 2, 30, 12, 1)

    role_config = _role_config_from_inputs(
        generalist={
            "curiosity_gain": gen_curiosity,
            "coherence_gain": gen_coherence,
            "avoidance_gain": gen_avoid,
            "deposit_sigma": gen_deposit,
            "battery_capacity": generalist_battery,
        },
        scout={
            "curiosity_gain": scout_curiosity,
            "coherence_gain": scout_coherence,
            "avoidance_gain": scout_avoid,
            "deposit_sigma": scout_deposit,
            "battery_capacity": scout_battery,
        },
        harvester={
            "curiosity_gain": harvester_curiosity,
            "coherence_gain": harvester_coherence,
            "avoidance_gain": harvester_avoid,
            "deposit_sigma": harvester_deposit,
            "battery_capacity": harvester_battery,
        },
    )

    swarm_config = {
        "width": int(width),
        "height": int(height),
        "n_drones": int(num_drones),
        "step": float(step),
        "inertia": float(inertia),
        "deposit_sigma": float(gen_deposit),
        "coherence_gain": float(gen_coherence),
        "curiosity": float(gen_curiosity),
        "avoidance_radius": float(avoidance_radius),
        "cohesion_neighbors": int(cohesion_neighbors),
        "evap": float(evap),
        "relax_alpha": float(relax_alpha),
        "boundary_mode": boundary_mode,
        "base_position": (float(base_x), float(base_y)),
        "battery_capacity": float(generalist_battery),
        "role_mix": role_mix,
        "role_config": role_config,
    }

    cfg_signature = _serialize_config(swarm_config)
    stored_sig = st.session_state.get("swarm_config_signature")

    swarm: DroneSwarm
    need_init = start_btn or "swarm" not in st.session_state or stored_sig != cfg_signature

    if need_init:
        swarm = DroneSwarm(**swarm_config)
        swarm.field = phi.copy()
        _ensure_role_battery(swarm, role_config)
        st.session_state.swarm = swarm
        st.session_state.swarm_config_signature = cfg_signature
        st.success("Schwarm neu initialisiert und Parameter übernommen.")
    else:
        swarm = st.session_state.swarm  # type: ignore[assignment]
        if swarm.field.shape != phi.shape:
            swarm.field = np.zeros_like(phi)
        swarm.set_params(
            step=float(step),
            inertia=float(inertia),
            deposit_sigma=float(gen_deposit),
            coherence_gain=float(gen_coherence),
            curiosity=float(gen_curiosity),
            avoidance_radius=float(avoidance_radius),
            cohesion_neighbors=int(cohesion_neighbors),
            evap=float(evap),
            relax_alpha=float(relax_alpha),
            boundary_mode=boundary_mode,
        )
        swarm.base_position = np.array([float(base_x), float(base_y)], dtype=np.float32)
        swarm.role_config = role_config
        _ensure_role_battery(swarm, role_config)

    swarm = st.session_state.swarm  # type: ignore[assignment]

    progress = st.progress(0.0)
    placeholder = st.empty()

    frames: list[np.ndarray] = []
    for i in range(int(run_steps)):
        if live_coupling and "phi_snapshot" in st.session_state:
            snap = np.array(st.session_state["phi_snapshot"], dtype=np.float32, copy=False)
            if snap.shape == swarm.field.shape:
                swarm.field = (1.0 - float(blend)) * swarm.field + float(blend) * snap

        swarm.step_swarm()
        progress.progress((i + 1) / run_steps)

        img = np.log(np.abs(swarm.field) + 1e-6)
        fig, ax = plt.subplots(figsize=(5.6, 5.6))
        ax.imshow(img, cmap="plasma", origin="upper")

        for role, color in ROLE_COLORS.items():
            pts = np.array(
                [d.pos for d in swarm.drones if d.active and d.role == role],
                dtype=np.float32,
            )
            if len(pts):
                ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    color=color,
                    s=18,
                    alpha=0.9,
                    edgecolors="black",
                    linewidths=0.3,
                    label=role.capitalize(),
                )

        returning = np.array(
            [d.pos for d in swarm.drones if d.active and d.returning_to_base],
            dtype=np.float32,
        )
        if len(returning):
            ax.scatter(
                returning[:, 0],
                returning[:, 1],
                marker="s",
                s=28,
                color="white",
                edgecolors="black",
                linewidths=0.4,
                label="Rückkehr",
            )

        base = swarm.base_position
        ax.scatter(
            [base[0]],
            [base[1]],
            marker="X",
            s=80,
            color="#f7f7f7",
            edgecolors="black",
            linewidths=1.0,
            label="Basis",
        )

        ax.set_xlim(0, width - 1)
        ax.set_ylim(height - 1, 0)
        ax.set_title(f"Iteration {swarm.iteration} — aktive Drohnen: {swarm.active_count}/{len(swarm.drones)}")
        ax.axis("off")
        if len(ax.collections) > 1:
            ax.legend(loc="lower right", fontsize=8, frameon=True)

        fig.canvas.draw()
        if record_run:
            width_px, height_px = fig.canvas.get_width_height()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape((height_px, width_px, 3))
            frames.append(frame)

        placeholder.pyplot(fig)
        plt.close(fig)
        time.sleep(0.02)

    st.success("Simulation abgeschlossen ✅")

    avg_battery = float(
        np.mean([d.battery for d in swarm.drones if d.active])
    ) if swarm.active_count else 0.0
    st.info(
        f"Durchschnittlicher Batteriestand aktiver Drohnen: {avg_battery:0.1f} | "
        f"Rückkehrende Drohnen: {sum(1 for d in swarm.drones if d.returning_to_base)}"
    )

    if record_run and frames:
        buffer = io.BytesIO()
        frame_duration = 1.0 / max(1, int(record_fps))
        imageio.mimsave(buffer, frames, format="gif", duration=frame_duration)
        buffer.seek(0)
        st.download_button(
            "GIF herunterladen",
            buffer,
            file_name=f"dronesim_{int(time.time())}.gif",
            mime="image/gif",
        )