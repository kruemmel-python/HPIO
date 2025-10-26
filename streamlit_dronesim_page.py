"""
Streamlit-Seite: Drohnen-Simulation für HPIO
--------------------------------------------
Simuliert autonome Drohnen, die das HPIO-Feld Φ als Umwelt nutzen.
Die Drohnen folgen Gradienten, deponieren Spuren und koordinieren sich schwarmartig.
"""

from __future__ import annotations
import time
import numpy as np
import streamlit as st
from dataclasses import dataclass, field
from typing import Any
import matplotlib.pyplot as plt

try:
    from streamlit_app import AppState
except ImportError:
    AppState = Any  # fallback


# ---------- Hilfsfunktionen ----------

def bilinear_sample(arr: np.ndarray, x: float, y: float) -> float:
    """Bilineare Interpolation arr[y, x] für Float-Positionen."""
    h, w = arr.shape
    x = float(np.clip(x, 0, w - 1))
    y = float(np.clip(y, 0, h - 1))
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    Ia = arr[y0, x0]
    Ib = arr[y0, x1]
    Ic = arr[y1, x0]
    Id = arr[y1, x1]
    top = Ia * (1 - fx) + Ib * fx
    bot = Ic * (1 - fx) + Id * fx
    return float(top * (1 - fy) + bot * fy)


def box_blur(field: np.ndarray) -> np.ndarray:
    """Kleiner 3x3 Box-Blur als Relaxation (ohne externe Abhängigkeiten)."""
    h, w = field.shape
    p = np.pad(field, 1, mode="edge")
    acc = (
        p[0:h,   0:w] + p[0:h,   1:w+1] + p[0:h,   2:w+2] +
        p[1:h+1, 0:w] + p[1:h+1, 1:w+1] + p[1:h+1, 2:w+2] +
        p[2:h+2, 0:w] + p[2:h+2, 1:w+1] + p[2:h+2, 2:w+2]
    )
    return acc / 9.0


# ---------- Basisklassen ----------

@dataclass
class Drone:
    """Einzelne Drohne im 2D-Schwarm."""
    pos: np.ndarray  # (x, y)
    vel: np.ndarray  # (vx, vy)
    battery: float = 1.0
    active: bool = True


@dataclass
class SwarmController:
    """Verwaltet mehrere Drohnen und interagiert mit HPIO-Feld."""
    num_drones: int
    field: np.ndarray
    step: float = 1.2
    momentum: float = 0.6
    deposit_sigma: float = 1.5
    coherence_gain: float = 0.15
    curiosity: float = 0.35
    avoidance_radius: float = 6.0
    evap: float = 0.02
    relax_alpha: float = 0.22
    iteration: int = 0
    drones: list[Drone] = field(default_factory=list)

    def __post_init__(self) -> None:
        h, w = self.field.shape  # KORREKT: (height, width)
        rng = np.random.default_rng(1234)
        self.drones = [
            Drone(
                pos=np.array([rng.uniform(0, w - 1), rng.uniform(0, h - 1)], dtype=np.float32),
                vel=rng.normal(0, 0.1, size=2).astype(np.float32),
            )
            for _ in range(self.num_drones)
        ]

    def _deposit(self, x: int, y: int) -> None:
        """Drohne legt Gaußspur ab."""
        h, w = self.field.shape
        rad = max(1, int(3 * self.deposit_sigma))
        x0, x1 = max(0, x - rad), min(w, x + rad + 1)
        y0, y1 = max(0, y - rad), min(h, y + rad + 1)
        if x0 >= x1 or y0 >= y1:
            return
        xs = np.arange(x0, x1)
        ys = np.arange(y0, y1)
        X, Y = np.meshgrid(xs, ys)
        g = np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * self.deposit_sigma ** 2)).astype(np.float32)
        self.field[y0:y1, x0:x1] += g

    def step_swarm(self) -> None:
        """Ein Simulationsschritt für den gesamten Schwarm."""
        if not self.drones:
            return

        # Feldgradient
        gy, gx = np.gradient(self.field.astype(np.float32))  # gy=d/dy, gx=d/dx
        h, w = self.field.shape

        # Globaler Schwerpunkt (für leichte Kohärenz)
        mean_pos = np.mean([d.pos for d in self.drones], axis=0)

        for i, d in enumerate(self.drones):
            if not d.active:
                continue

            # Bilinear gesampelter Gradient
            gx_val = bilinear_sample(gx, d.pos[0], d.pos[1])
            gy_val = bilinear_sample(gy, d.pos[0], d.pos[1])
            grad = np.array([gx_val, gy_val], dtype=np.float32)
            gn = np.linalg.norm(grad) + 1e-8
            grad_dir = grad / gn

            # Neugier + Kohärenz
            noise = np.random.normal(0.0, 1.0, size=2).astype(np.float32)
            noise /= (np.linalg.norm(noise) + 1e-8)

            coh_vec = (mean_pos - d.pos).astype(np.float32)
            coh_vec /= (np.linalg.norm(coh_vec) + 1e-8)

            # Einfache Kollisionsvermeidung (weiche Abstoßung)
            avoid = np.zeros(2, dtype=np.float32)
            if self.avoidance_radius > 0:
                for j, o in enumerate(self.drones):
                    if i == j:
                        continue
                    diff = d.pos - o.pos
                    dist = float(np.linalg.norm(diff) + 1e-8)
                    if 0 < dist < self.avoidance_radius:
                        avoid += (diff / dist)

            # Antriebsvektor (Gewichte gern im UI tunen)
            drive = (
                0.65 * grad_dir +
                self.curiosity * 0.35 * noise +
                self.coherence_gain * coh_vec +
                0.4 * avoid
            )

            # Velocity-Update (Trägheit) + Schritt
            d.vel = self.momentum * d.vel + (1.0 - self.momentum) * drive
            # sanfte Kappung
            sp = np.linalg.norm(d.vel) + 1e-8
            max_sp = max(0.5, 3.0 * self.step)
            if sp > max_sp:
                d.vel *= (max_sp / sp)

            d.pos += self.step * d.vel

            # Ränder reflektierend
            if d.pos[0] < 0:
                d.pos[0] = -d.pos[0]; d.vel[0] = abs(d.vel[0])
            elif d.pos[0] > w - 1:
                d.pos[0] = 2 * (w - 1) - d.pos[0]; d.vel[0] = -abs(d.vel[0])

            if d.pos[1] < 0:
                d.pos[1] = -d.pos[1]; d.vel[1] = abs(d.vel[1])
            elif d.pos[1] > h - 1:
                d.pos[1] = 2 * (h - 1) - d.pos[1]; d.vel[1] = -abs(d.vel[1])

            # Batterie & Ablage
            d.battery -= 0.001
            if d.battery <= 0:
                d.active = False
            else:
                self._deposit(int(d.pos[0]), int(d.pos[1]))

        # Feld-Update: Verdunstung + Relaxation
        if self.evap > 0:
            self.field *= (1.0 - float(self.evap))
        if self.relax_alpha > 0:
            blurred = box_blur(self.field)
            a = float(self.relax_alpha)
            self.field = (1.0 - a) * self.field + a * blurred

        self.iteration += 1


# ---------- Streamlit Page ----------

def page_dronesim(state: AppState) -> None:
    """Zeigt die Drohnen-Simulation im Streamlit-UI (mit Φ-Snapshot, Re-Init & Live-Kopplung)."""
    st.markdown("## 🛸 Drohnen-Schwarm-Simulation")
    st.caption("Simulation eines bio-inspirierten Schwarms auf Basis des HPIO-Feldes Φ.")

    # --- Φ-Quelle priorisieren: Snapshot > Live > Fallback -------------------
    if "phi_snapshot" in st.session_state:
        phi_src = st.session_state["phi_snapshot"]
    elif state.controller and hasattr(state.controller.optimizer, "field"):
        phi_src = state.controller.optimizer.field.phi
    else:
        st.warning("Kein HPIO-Feld aktiv. Standardfeld wird verwendet.")
        phi_src = np.zeros((160, 160), dtype=np.float32)
    phi = np.array(phi_src, dtype=np.float32, copy=True)

    # --- Sidebar-Parameter ---------------------------------------------------
    with st.sidebar:
        st.markdown("### Schwarmparameter")
        num_drones = st.slider("Anzahl Drohnen", 4, 200, 32, 4)
        step = st.slider("Schrittweite", 0.1, 5.0, 1.2, 0.1)
        momentum = st.slider("Trägheit", 0.0, 0.99, 0.6, 0.01)
        deposit_sigma = st.slider("Ablage-Sigma", 0.5, 4.0, 1.5, 0.1)
        coherence_gain = st.slider("Kohärenz-Gain", 0.0, 1.0, 0.15, 0.01)
        curiosity = st.slider("Neugier", 0.0, 1.0, 0.35, 0.01)
        avoidance_radius = st.slider("Vermeidungsradius", 0.0, 20.0, 6.0, 0.5)
        evap = st.slider("Verdunstung Φ", 0.0, 0.2, 0.02, 0.01)
        relax_alpha = st.slider("Relaxation Φ", 0.0, 1.0, 0.22, 0.01)
        run_steps = st.number_input("Simulationsschritte", 1, 1000, 100, 1)

        st.markdown("### Φ-Livekopplung")
        live_coupling = st.checkbox("HPIO-Φ live einkoppeln (Blending)", value=True)
        blend = st.slider("Blend-Faktor (neu ↔ alt)", 0.0, 1.0, 0.2, 0.05,
                          help="0.0 = nur Swarm-Feld, 1.0 = nur aktueller Φ-Snapshot")

        st.markdown("### Steuerung")
        start_btn = st.button("Start Simulation")

    # --- (Re)Init des Schwarms ----------------------------------------------
    need_init = False
    if start_btn or "swarm_controller" not in st.session_state:
        need_init = True
    else:
        # Re-Init erzwingen, wenn sich Φ-Shape ändert (z.B. Preset/Grid geändert)
        if getattr(st.session_state.swarm_controller, "field", None) is None:
            need_init = True
        else:
            if st.session_state.swarm_controller.field.shape != phi.shape:
                need_init = True

    if need_init:
        # Alte Instanz ggf. sauber verwerfen, dann mit aktuellem Φ neu aufsetzen
        st.session_state.swarm_controller = SwarmController(
            num_drones=num_drones,
            field=phi,
            step=float(step),
            momentum=float(momentum),
            deposit_sigma=float(deposit_sigma),
            coherence_gain=float(coherence_gain),
            curiosity=float(curiosity),
            avoidance_radius=float(avoidance_radius),
            evap=float(evap),
            relax_alpha=float(relax_alpha),
        )
        st.success("Schwarm initialisiert.")

    swarm: SwarmController = st.session_state.swarm_controller
    # Laufende Parameter-Änderungen an die Instanz durchreichen (ohne kompletten Reset)
    swarm.step = float(step)
    swarm.momentum = float(momentum)
    swarm.deposit_sigma = float(deposit_sigma)
    swarm.coherence_gain = float(coherence_gain)
    if hasattr(swarm, "curiosity"):        swarm.curiosity = float(curiosity)
    if hasattr(swarm, "avoidance_radius"): swarm.avoidance_radius = float(avoidance_radius)
    if hasattr(swarm, "evap"):             swarm.evap = float(evap)
    if hasattr(swarm, "relax_alpha"):      swarm.relax_alpha = float(relax_alpha)
    if hasattr(swarm, "set_num_drones"):   swarm.set_num_drones(int(num_drones))  # falls vorhanden

    # --- Simulation ----------------------------------------------------------
    progress = st.progress(0.0)
    placeholder = st.empty()

    for i in range(int(run_steps)):
        # Optional: aktuelles HPIO-Φ einkoppeln (blenden), wenn vorhanden
        if live_coupling and "phi_snapshot" in st.session_state:
            snap = np.array(st.session_state["phi_snapshot"], dtype=np.float32, copy=False)
            if snap.shape == swarm.field.shape:
                # Blend: neues Feld (HPIO) vs. internes Swarm-Feld
                swarm.field = (1.0 - float(blend)) * swarm.field + float(blend) * snap

        # Einen Simulationsschritt ausführen
        swarm.step_swarm()
        progress.progress((i + 1) / run_steps)

        # Darstellung
        img = np.log(np.abs(swarm.field) + 1e-6)
        drones_xy = np.array([d.pos for d in swarm.drones if d.active])

        fig, ax = plt.subplots(figsize=(5.6, 5.6))
        ax.imshow(img, cmap="plasma", origin="lower")
        if len(drones_xy):
            ax.scatter(
                drones_xy[:, 0], drones_xy[:, 1],
                color="cyan", s=14, alpha=0.85,
                edgecolors="black", linewidths=0.3,
            )
        ax.set_title(f"Iteration {swarm.iteration}", fontsize=12)
        ax.axis("off")
        placeholder.pyplot(fig)
        plt.close(fig)

        time.sleep(0.02)

    st.success("Simulation abgeschlossen ✅")

