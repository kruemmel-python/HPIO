# dronesim.py
# Bio-inspirierte 2D-Drohnenschwarm-Simulation auf Basis eines skalareren Feldes Φ.
# Passt zur Streamlit-Seite "Drohnen-Simulation" in deinem Projekt.

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np


# ------------------------------- Utilities -------------------------------- #

def _bilinear_sample(arr: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Bilineare Interpolation auf arr[y, x] für beliebige Float-Koordinaten.
    x, y sind gleich lange Vektoren. Werte außerhalb werden an den Rand geclamped.
    """
    h, w = arr.shape
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)

    Ia = arr[y0, x0]
    Ib = arr[y0, x1]
    Ic = arr[y1, x0]
    Id = arr[y1, x1]

    fx = x - x0
    fy = y - y0

    top = Ia * (1 - fx) + Ib * fx
    bot = Ic * (1 - fx) + Id * fx
    return top * (1 - fy) + bot * fy


def _gaussian_stamp(shape: Tuple[int, int], cx: float, cy: float, sigma: float, amp: float = 1.0) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Erzeuge einen kleinen 2D-Gaußabdruck um (cx,cy) und gib (patch, (y0,y1,x0,x1)) zurück,
    damit er effizient in ein Feld addiert werden kann.
    """
    h, w = shape
    rad = max(1, int(3.0 * sigma))
    x0 = max(0, int(math.floor(cx)) - rad)
    x1 = min(w, int(math.floor(cx)) + rad + 1)
    y0 = max(0, int(math.floor(cy)) - rad)
    y1 = min(h, int(math.floor(cy)) + rad + 1)

    if x0 >= x1 or y0 >= y1:
        return np.zeros((0, 0), dtype=np.float32), (0, 0, 0, 0)

    xs = np.arange(x0, x1)
    ys = np.arange(y0, y1)
    X, Y = np.meshgrid(xs, ys)
    g = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
    return (amp * g, (y0, y1, x0, x1))


def _box_blur(field: np.ndarray) -> np.ndarray:
    """Kleiner 3×3-Boxblur als Relaxation (ohne externe Abhängigkeiten)."""
    h, w = field.shape
    padded = np.pad(field, 1, mode="edge")
    acc = (
        padded[0:h,   0:w] + padded[0:h,   1:w+1] + padded[0:h,   2:w+2] +
        padded[1:h+1, 0:w] + padded[1:h+1, 1:w+1] + padded[1:h+1, 2:w+2] +
        padded[2:h+2, 0:w] + padded[2:h+2, 1:w+1] + padded[2:h+2, 2:w+2]
    )
    return acc / 9.0


# ------------------------------- Data Model -------------------------------- #

@dataclass
class Drone:
    pos: np.ndarray          # shape (2,) -> (x, y) in Weltkoordinaten (Pixel)
    vel: np.ndarray          # shape (2,)
    active: bool = True


class DroneSwarm:
    """
    Einfacher Drohnen-Schwarm, der sich entlang Gradienten eines Feldes Φ bewegt,
    mit Neugier, Kohärenz (Schwarmdrift) und Vermeidung (Kollisionsschutz).
    """

    def __init__(
        self,
        width: int,
        height: int,
        n_drones: int = 32,
        step: float = 1.2,
        inertia: float = 0.68,
        deposit_sigma: float = 1.6,
        coherence_gain: float = 0.15,
        curiosity: float = 0.5,
        avoidance_radius: float = 6.0,
        evap: float = 0.02,
        relax_alpha: float = 0.25,
        seed: Optional[int] = None,
    ) -> None:
        self.w = int(width)
        self.h = int(height)
        self.iteration = 0

        self.step = float(step)
        self.inertia = float(inertia)
        self.deposit_sigma = float(deposit_sigma)
        self.coherence_gain = float(coherence_gain)
        self.curiosity = float(curiosity)
        self.avoidance_radius = float(avoidance_radius)
        self.evap = float(evap)
        self.relax_alpha = float(relax_alpha)

        self.rng = np.random.default_rng(seed)

        # Feld Φ (Pheromon/Heatmap)
        self.field = np.zeros((self.h, self.w), dtype=np.float32)

        # Drohnen initial verteilen
        self.drones: List[Drone] = []
        for _ in range(int(n_drones)):
            x = self.rng.uniform(0, self.w - 1)
            y = self.rng.uniform(0, self.h - 1)
            theta = self.rng.uniform(0, 2 * np.pi)
            v0 = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32) * 0.1
            self.drones.append(Drone(pos=np.array([x, y], dtype=np.float32), vel=v0))

    # ----------------------------- API / Controls ---------------------------- #

    def set_params(
        self,
        *,
        step: Optional[float] = None,
        inertia: Optional[float] = None,
        deposit_sigma: Optional[float] = None,
        coherence_gain: Optional[float] = None,
        curiosity: Optional[float] = None,
        avoidance_radius: Optional[float] = None,
        evap: Optional[float] = None,
        relax_alpha: Optional[float] = None,
    ) -> None:
        if step is not None:
            self.step = float(step)
        if inertia is not None:
            self.inertia = float(inertia)
        if deposit_sigma is not None:
            self.deposit_sigma = float(deposit_sigma)
        if coherence_gain is not None:
            self.coherence_gain = float(coherence_gain)
        if curiosity is not None:
            self.curiosity = float(curiosity)
        if avoidance_radius is not None:
            self.avoidance_radius = float(avoidance_radius)
        if evap is not None:
            self.evap = float(evap)
        if relax_alpha is not None:
            self.relax_alpha = float(relax_alpha)

    def inject_hotspots(self, n: int = 12, sigma: float = 10.0, amp: float = 1.0) -> None:
        """Hilfsfunktion: Initiale Hotspots ins Feld zeichnen, damit es etwas zu 'finden' gibt."""
        for _ in range(int(n)):
            cx = self.rng.uniform(0, self.w - 1)
            cy = self.rng.uniform(0, self.h - 1)
            patch, (y0, y1, x0, x1) = _gaussian_stamp((self.h, self.w), cx, cy, sigma, amp)
            if patch.size:
                self.field[y0:y1, x0:x1] += patch

    def overlay_external_field(self, phi: np.ndarray, scale: float = 1.0) -> None:
        """Optional: Externes Feld (z.B. aus HPIO) überlagern."""
        if phi.shape != self.field.shape:
            raise ValueError("Form des externen Feldes passt nicht.")
        self.field = (self.field + scale * phi.astype(np.float32)) * 0.5

    # --------------------------- Simulation Core ----------------------------- #

    def step_swarm(self) -> None:
        """Ein Simulationsschritt für den gesamten Schwarm."""
        if not self.drones:
            return

        # --- Feld-Gradient (zieht Drohnen in Richtung hoher Intensität) --- #
        gy, gx = np.gradient(self.field.astype(np.float32))  # gy -> d/dy, gx -> d/dx

        # Positionsarrays
        pos = np.vstack([d.pos for d in self.drones]).astype(np.float32)   # (N,2)
        vel = np.vstack([d.vel for d in self.drones]).astype(np.float32)   # (N,2)

        xs = pos[:, 0]
        ys = pos[:, 1]

        # Gradient an Drohnenposition bilinear abtasten
        gx_s = _bilinear_sample(gx, xs, ys)
        gy_s = _bilinear_sample(gy, xs, ys)
        grad = np.stack([gx_s, gy_s], axis=1)  # (N,2)

        # Normierte Gradientenrichtung (NaNs vermeiden)
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-8
        grad_dir = grad / grad_norm

        # Neugier (leichte Zufallskomponente, normiert)
        noise = self.rng.normal(0.0, 1.0, size=vel.shape).astype(np.float32)
        noise_norm = np.linalg.norm(noise, axis=1, keepdims=True) + 1e-8
        curiosity_dir = noise / noise_norm

        # Kohärenz (zum lokalen Schwerpunkt hin)
        mean_pos = pos.mean(axis=0, keepdims=True)  # globaler Schwerpunkt reicht hier
        coh_vec = (mean_pos - pos)
        coh_norm = np.linalg.norm(coh_vec, axis=1, keepdims=True) + 1e-8
        cohesion_dir = coh_vec / coh_norm

        # Einfache Kollisionsvermeidung (weiche Abstoßung)
        avoidance = np.zeros_like(pos)
        if len(self.drones) > 1 and self.avoidance_radius > 0:
            for i in range(pos.shape[0]):
                diff = pos[i] - pos  # Vektoren von anderen zu i
                dist = np.linalg.norm(diff, axis=1) + 1e-8
                mask = (dist < self.avoidance_radius) & (dist > 0)
                if np.any(mask):
                    push = (diff[mask] / dist[mask, None]).sum(axis=0)
                    avoidance[i] = push

        # Neue Geschwindigkeit (Trägheit + Kombination der Treiber)
        # Gewichte können nach Geschmack getuned werden.
        w_grad = 0.65
        w_cur = 0.20
        w_coh = float(self.coherence_gain)  # aus UI
        w_avoid = 0.40

        drive = (
            w_grad * grad_dir +
            w_cur * curiosity_dir +
            w_coh * cohesion_dir +
            w_avoid * avoidance
        )

        vel = self.inertia * vel + self.step * drive

        # Maximalgeschwindigkeit (sanfte Kappung)
        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        max_speed = max(0.5, 3.0 * self.step)
        factor = np.minimum(1.0, (max_speed / (speed + 1e-8))).astype(np.float32)
        vel = vel * factor

        # Positionen aktualisieren
        pos = pos + vel

        # Ränder: reflektieren
        for i in range(pos.shape[0]):
            if pos[i, 0] < 0:
                pos[i, 0] = -pos[i, 0]
                vel[i, 0] = abs(vel[i, 0])
            elif pos[i, 0] > self.w - 1:
                pos[i, 0] = 2 * (self.w - 1) - pos[i, 0]
                vel[i, 0] = -abs(vel[i, 0])

            if pos[i, 1] < 0:
                pos[i, 1] = -pos[i, 1]
                vel[i, 1] = abs(vel[i, 1])
            elif pos[i, 1] > self.h - 1:
                pos[i, 1] = 2 * (self.h - 1) - pos[i, 1]
                vel[i, 1] = -abs(vel[i, 1])

        # Zustände zurückschreiben
        for i, d in enumerate(self.drones):
            d.pos[:] = pos[i]
            d.vel[:] = vel[i]

        # Pheromon-Ablage (kleine Gauss-Stempel an jeder Drohnenposition)
        if self.deposit_sigma > 0:
            sig = float(self.deposit_sigma)
            for i in range(pos.shape[0]):
                cx, cy = pos[i, 0], pos[i, 1]
                patch, (y0, y1, x0, x1) = _gaussian_stamp((self.h, self.w), cx, cy, sigma=sig, amp=1.0)
                if patch.size:
                    self.field[y0:y1, x0:x1] += patch

        # Verdunsten & Relaxation
        if self.evap > 0:
            self.field *= max(0.0, 1.0 - float(self.evap))
        if self.relax_alpha > 0:
            blurred = _box_blur(self.field)
            a = float(self.relax_alpha)
            self.field = (1.0 - a) * self.field + a * blurred

        self.iteration += 1

    # ----------------------------- Convenience ------------------------------ #

    @property
    def positions(self) -> np.ndarray:
        return np.vstack([d.pos for d in self.drones]).astype(np.float32)

    @property
    def active_count(self) -> int:
        return sum(1 for d in self.drones if d.active)
