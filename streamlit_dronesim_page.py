"""
Streamlit-Seite: Drohnen-Simulation für HPIO
--------------------------------------------
Simuliert autonome Drohnen, die das HPIO-Feld Φ als Umwelt nutzen.
Die Drohnen folgen Gradienten, deponieren Spuren und koordinieren sich schwarmartig.
"""

from __future__ import annotations
import io
import time
import numpy as np
import streamlit as st
from dataclasses import dataclass, field
from typing import Any
from numpy.random import Generator
import matplotlib.pyplot as plt
import imageio

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


def bilinear_sample_vec(arr: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilineare Interpolation für Vektor-Eingaben (xs, ys)."""
    h, w = arr.shape
    xs = np.clip(xs, 0.0, w - 1.0)
    ys = np.clip(ys, 0.0, h - 1.0)

    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    fx = xs - x0
    fy = ys - y0

    Ia = arr[y0, x0]
    Ib = arr[y0, x1]
    Ic = arr[y1, x0]
    Id = arr[y1, x1]

    top = Ia * (1.0 - fx) + Ib * fx
    bot = Ic * (1.0 - fx) + Id * fx
    return (top * (1.0 - fy) + bot * fy).astype(np.float32)


def box_blur(field: np.ndarray) -> np.ndarray:
    """3x3 Box-Blur ohne Randplateaus (nutzt reflektierendes Padding)."""
    h, w = field.shape
    p = np.pad(field, 1, mode="reflect")
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
    curiosity: float = 0.2
    avoidance_radius: float = 6.0
    evap: float = 0.02
    relax_alpha: float = 0.22
    iteration: int = 0
    drones: list[Drone] = field(default_factory=list)
    rng: Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        h, w = self.field.shape  # (height, width)
        self.rng = np.random.default_rng(1234)
        self.drones = []
        for _ in range(self.num_drones):
            x = self.rng.uniform(0.0, max(w - 1.0, 1.0))
            y = self.rng.uniform(0.0, max(h - 1.0, 1.0))
            theta = self.rng.uniform(0.0, 2.0 * np.pi)
            v0 = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32) * 0.1
            self.drones.append(
                Drone(
                    pos=np.array([x, y], dtype=np.float32),
                    vel=v0,
                )
            )

    def _deposit(self, cx: float, cy: float) -> None:
        """Drohne legt Gaußspur mit gleitendem Zentrum ab."""
        if self.deposit_sigma <= 0:
            return
        h, w = self.field.shape
        sigma = float(self.deposit_sigma)
        rad = max(1, int(3.0 * sigma))
        x_floor = int(np.floor(cx))
        y_floor = int(np.floor(cy))
        x0 = max(0, x_floor - rad)
        x1 = min(w, x_floor + rad + 1)
        y0 = max(0, y_floor - rad)
        y1 = min(h, y_floor + rad + 1)
        if x0 >= x1 or y0 >= y1:
            return
        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)
        patch = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
        self.field[y0:y1, x0:x1] += patch

    def step_swarm(self) -> None:
        """Ein Simulationsschritt für den gesamten Schwarm."""
        if not self.drones:
            return

        h, w = self.field.shape
        field32 = self.field.astype(np.float32, copy=False)
        gy, gx = np.gradient(field32, edge_order=1)

        active_idx = [i for i, d in enumerate(self.drones) if d.active]
        if not active_idx:
            return

        pos = np.vstack([self.drones[i].pos for i in active_idx]).astype(np.float32)
        vel = np.vstack([self.drones[i].vel for i in active_idx]).astype(np.float32)

        xs = pos[:, 0]
        ys = pos[:, 1]
        gx_s = bilinear_sample_vec(gx, xs, ys)
        gy_s = bilinear_sample_vec(gy, xs, ys)
        grad = np.stack([gx_s, gy_s], axis=1)
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-8
        grad_dir = grad / grad_norm

        noise = self.rng.normal(0.0, 1.0, size=grad.shape).astype(np.float32)
        noise_norm = np.linalg.norm(noise, axis=1, keepdims=True) + 1e-8
        curiosity_dir = noise / noise_norm

        mean_pos = pos.mean(axis=0, keepdims=True)
        coh_vec = mean_pos - pos
        coh_norm = np.linalg.norm(coh_vec, axis=1, keepdims=True) + 1e-8
        cohesion_dir = coh_vec / coh_norm

        avoidance = np.zeros_like(pos, dtype=np.float32)
        if self.avoidance_radius > 0 and pos.shape[0] > 1:
            diff = pos[:, None, :] - pos[None, :, :]
            dist = np.linalg.norm(diff, axis=2) + 1e-8
            mask = (dist < self.avoidance_radius) & (~np.eye(pos.shape[0], dtype=bool))
            if np.any(mask):
                avoidance = ((diff / dist[..., None]) * mask[..., None]).sum(axis=1).astype(np.float32)

        w_grad = 0.65
        w_cur = float(self.curiosity)
        w_coh = float(self.coherence_gain)
        w_avoid = 0.40

        drive = (
            w_grad * grad_dir
            + w_cur * curiosity_dir
            + w_coh * cohesion_dir
            + w_avoid * avoidance
        ).astype(np.float32)

        vel = (self.momentum * vel + self.step * drive).astype(np.float32)
        speed = np.linalg.norm(vel, axis=1, keepdims=True) + 1e-8
        max_sp = max(0.5, 3.0 * self.step)
        vel = vel * np.minimum(1.0, max_sp / speed)

        pos = pos + vel

        for i in range(pos.shape[0]):
            if pos[i, 0] < 0:
                pos[i, 0] = -pos[i, 0]
                vel[i, 0] = abs(vel[i, 0])
            elif pos[i, 0] > w - 1:
                pos[i, 0] = 2 * (w - 1) - pos[i, 0]
                vel[i, 0] = -abs(vel[i, 0])

            if pos[i, 1] < 0:
                pos[i, 1] = -pos[i, 1]
                vel[i, 1] = abs(vel[i, 1])
            elif pos[i, 1] > h - 1:
                pos[i, 1] = 2 * (h - 1) - pos[i, 1]
                vel[i, 1] = -abs(vel[i, 1])

        for arr_idx, drone_idx in enumerate(active_idx):
            d = self.drones[drone_idx]
            d.pos[:] = pos[arr_idx]
            d.vel[:] = vel[arr_idx]
            d.battery -= 0.001
            if d.battery <= 0:
                d.active = False
                continue
            self._deposit(float(d.pos[0]), float(d.pos[1]))

        # Feld-Update: Verdunstung + Relaxation
        if self.evap > 0:
            self.field *= max(0.0, 1.0 - float(self.evap))
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
        curiosity = st.slider("Neugier", 0.0, 1.0, 0.20, 0.01)
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
        record_run = st.checkbox("Simulation aufzeichnen (GIF)", value=False)
        record_fps = st.slider("GIF-Bildrate", 2, 30, 15, 1)

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

    frames: list[np.ndarray] = []
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
        ax.imshow(img, cmap="plasma", origin="upper")
        if len(drones_xy):
            ax.scatter(
                drones_xy[:, 0], drones_xy[:, 1],
                color="cyan", s=14, alpha=0.85,
                edgecolors="black", linewidths=0.3,
            )
        ax.set_title(f"Iteration {swarm.iteration}", fontsize=12)
        ax.axis("off")
        fig.canvas.draw()
        if record_run:
            width, height = fig.canvas.get_width_height()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape((height, width, 3))
            frames.append(frame)
        placeholder.pyplot(fig)
        plt.close(fig)

        time.sleep(0.02)

    st.success("Simulation abgeschlossen ✅")

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
