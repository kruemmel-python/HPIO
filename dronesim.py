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
    role: str = "generalist"
    battery: float = 1.0
    battery_capacity: float = 1.0
    returning_to_base: bool = False
    curiosity_gain: float = 0.20
    coherence_gain: float = 0.15
    avoidance_gain: float = 0.40
    deposit_sigma: float = 1.6
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
        cohesion_neighbors: int = 6,
        evap: float = 0.02,
        relax_alpha: float = 0.25,
        boundary_mode: str = "reflect",
        base_position: Optional[Tuple[float, float]] = None,
        battery_capacity: float = 600.0,
        role_mix: Optional[dict] = None,
        role_config: Optional[dict] = None,
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
        self.cohesion_neighbors = max(1, int(cohesion_neighbors))
        self.evap = float(evap)
        self.relax_alpha = float(relax_alpha)
        self.boundary_mode = boundary_mode
        self.base_position = (
            np.array(base_position, dtype=np.float32)
            if base_position is not None
            else np.array([self.w * 0.5, self.h * 0.5], dtype=np.float32)
        )
        self.battery_capacity = float(battery_capacity)

        self.rng = np.random.default_rng(seed)

        # Feld Φ (Pheromon/Heatmap)
        self.field = np.zeros((self.h, self.w), dtype=np.float32)

        # Drohnen initial verteilen
        self.role_config = self._build_role_config(role_config)
        self.drones: List[Drone] = []

        mix = self._expand_role_mix(role_mix, int(n_drones))
        for role in mix:
            x = self.rng.uniform(0, self.w - 1)
            y = self.rng.uniform(0, self.h - 1)
            theta = self.rng.uniform(0, 2 * np.pi)
            v0 = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32) * 0.1
            drone = self._instantiate_drone(role, np.array([x, y], dtype=np.float32), v0)
            self.drones.append(drone)

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
        cohesion_neighbors: Optional[int] = None,
        evap: Optional[float] = None,
        relax_alpha: Optional[float] = None,
        boundary_mode: Optional[str] = None,
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
        if cohesion_neighbors is not None:
            self.cohesion_neighbors = max(1, int(cohesion_neighbors))
        if evap is not None:
            self.evap = float(evap)
        if relax_alpha is not None:
            self.relax_alpha = float(relax_alpha)
        if boundary_mode is not None:
            self.boundary_mode = boundary_mode

    # --------------------------- Role Management --------------------------- #

    def _build_role_config(self, overrides: Optional[dict]) -> dict:
        base_config = {
            "generalist": {
                "curiosity_gain": 0.20,
                "coherence_gain": self.coherence_gain,
                "avoidance_gain": 0.40,
                "deposit_sigma": self.deposit_sigma,
                "battery_capacity": self.battery_capacity,
            },
            "scout": {
                "curiosity_gain": 0.55,
                "coherence_gain": max(0.05, self.coherence_gain * 0.5),
                "avoidance_gain": 0.35,
                "deposit_sigma": max(0.5, self.deposit_sigma * 0.5),
                "battery_capacity": self.battery_capacity * 0.85,
            },
            "harvester": {
                "curiosity_gain": max(0.05, self.curiosity * 0.3),
                "coherence_gain": self.coherence_gain * 1.75,
                "avoidance_gain": 0.45,
                "deposit_sigma": self.deposit_sigma * 1.8,
                "battery_capacity": self.battery_capacity * 1.2,
            },
        }
        if overrides:
            for name, conf in overrides.items():
                if name in base_config:
                    base_config[name].update(conf)
                else:
                    base_config[name] = conf
        return base_config

    def _expand_role_mix(self, role_mix: Optional[dict], n_drones: int) -> List[str]:
        if not role_mix:
            role_mix = {"generalist": 0.6, "scout": 0.2, "harvester": 0.2}

        total = float(sum(role_mix.values()))
        if total <= 0:
            role_mix = {"generalist": 1.0}
            total = 1.0
        roles: List[str] = []
        residual = n_drones
        for i, (role, weight) in enumerate(role_mix.items()):
            if i == len(role_mix) - 1:
                count = residual
            else:
                count = int(round((weight / total) * n_drones))
                count = min(count, residual)
            roles.extend([role] * max(0, count))
            residual -= count
        while residual > 0:
            roles.append("generalist")
            residual -= 1
        self.rng.shuffle(roles)
        return roles[:n_drones]

    def _instantiate_drone(self, role: str, pos: np.ndarray, vel: np.ndarray) -> Drone:
        conf = self.role_config.get(role, self.role_config["generalist"])
        capacity = float(conf.get("battery_capacity", self.battery_capacity))
        return Drone(
            pos=pos.astype(np.float32),
            vel=vel.astype(np.float32),
            role=role,
            battery=capacity,
            battery_capacity=capacity,
            curiosity_gain=float(conf.get("curiosity_gain", self.curiosity)),
            coherence_gain=float(conf.get("coherence_gain", self.coherence_gain)),
            avoidance_gain=float(conf.get("avoidance_gain", 0.4)),
            deposit_sigma=float(conf.get("deposit_sigma", self.deposit_sigma)),
        )

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

    # --------------------------- Helper Mechanics --------------------------- #

    def _compute_local_cohesion(self, pos: np.ndarray) -> np.ndarray:
        """Berechne Kohäsionsrichtungen basierend auf k nächsten Nachbarn."""
        n = pos.shape[0]
        if n <= 1:
            return np.zeros_like(pos)

        diffs = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diffs, axis=2)
        dist += np.eye(n) * np.finfo(np.float32).max
        neighbor_count = min(self.cohesion_neighbors, n - 1)
        indices = np.argpartition(dist, neighbor_count, axis=1)[:, :neighbor_count]

        cohesion_vec = np.zeros_like(pos)
        for i in range(n):
            neighbors = indices[i]
            neighbor_pos = pos[neighbors]
            local_mean = neighbor_pos.mean(axis=0)
            vec = local_mean - pos[i]
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                cohesion_vec[i] = vec / norm
        return cohesion_vec

    def _compute_predictive_avoidance(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Prädiktive Kollisionsvermeidung via Zeit-bis-Kollision Heuristik."""
        n = pos.shape[0]
        avoidance = np.zeros_like(pos)
        if n <= 1 or self.avoidance_radius <= 0:
            return avoidance

        diffs = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diffs, axis=2)
        rel_vel = vel[:, None, :] - vel[None, :, :]

        with np.errstate(divide="ignore", invalid="ignore"):
            closing = -np.sum(diffs * rel_vel, axis=2) / (dist + 1e-8)
        time_to_collision = dist / (closing + 1e-6)

        for i in range(n):
            mask = (dist[i] > 0) & (dist[i] < self.avoidance_radius)
            if not np.any(mask):
                continue
            influence = np.zeros(2, dtype=np.float32)
            for j in np.where(mask)[0]:
                dir_vec = diffs[i, j]
                d = dist[i, j]
                closing_speed = max(0.0, closing[i, j])
                ttc = max(1e-3, time_to_collision[i, j])
                urgency = (self.avoidance_radius - d) / self.avoidance_radius
                urgency *= 1.0 + closing_speed * 0.5 + 1.0 / (ttc + 1e-3)
                steer = dir_vec / (d + 1e-6)
                orthogonal = np.array([-steer[1], steer[0]], dtype=np.float32)
                align = rel_vel[i, j]
                lateral = orthogonal * np.sign(np.dot(orthogonal, align))
                influence += (steer + 0.5 * lateral) * urgency
            norm = np.linalg.norm(influence)
            if norm > 1e-8:
                avoidance[i] = influence / norm
        return avoidance

    def _update_energy_and_homeward_drive(self, pos: np.ndarray, vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Verwalte Batterien und liefere Heimwärtsvektoren samt Gewichten."""
        home_dir = np.zeros_like(pos)
        weights = np.zeros((pos.shape[0], 1), dtype=np.float32)
        base = self.base_position.astype(np.float32)
        for i, drone in enumerate(self.drones):
            if not drone.active:
                continue
            speed = float(np.linalg.norm(vel[i]))
            consumption = 0.8 + speed * 0.15
            drone.battery = max(0.0, drone.battery - consumption)
            if drone.battery <= 0.0:
                drone.active = False
                vel[i] = 0.0
                continue

            distance_to_base = np.linalg.norm(base - pos[i])
            threshold = max(0.25 * drone.battery_capacity, 80.0)
            if (drone.battery < threshold and distance_to_base > 3.0) or drone.returning_to_base:
                drone.returning_to_base = True
                dir_vec = base - pos[i]
                norm = np.linalg.norm(dir_vec)
                if norm > 1e-8:
                    home_dir[i] = dir_vec / norm
                    weights[i] = 1.8 * (1.0 + (threshold - drone.battery) / threshold)
            if drone.returning_to_base and distance_to_base <= 3.0:
                drone.battery = drone.battery_capacity
                drone.returning_to_base = False
        return home_dir, weights

    def _apply_boundary_conditions(self, pos: np.ndarray, vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.boundary_mode == "periodic":
            pos[:, 0] = np.mod(pos[:, 0], self.w)
            pos[:, 1] = np.mod(pos[:, 1], self.h)
            return pos, vel

        if self.boundary_mode == "soft":
            margin = 5.0
            damping = 0.4
            for i in range(pos.shape[0]):
                if pos[i, 0] < margin:
                    vel[i, 0] += (margin - pos[i, 0]) * damping
                elif pos[i, 0] > self.w - margin:
                    vel[i, 0] -= (pos[i, 0] - (self.w - margin)) * damping
                if pos[i, 1] < margin:
                    vel[i, 1] += (margin - pos[i, 1]) * damping
                elif pos[i, 1] > self.h - margin:
                    vel[i, 1] -= (pos[i, 1] - (self.h - margin)) * damping
            pos = np.clip(pos, [0.0, 0.0], [self.w - 1.0, self.h - 1.0])
            return pos, vel

        # Default: reflektierend
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
        return pos, vel

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

        # Lokale Kohäsion (k-nächste Nachbarn)
        cohesion_dir = self._compute_local_cohesion(pos)

        # Kollisionsvermeidung mit prädiktivem Steuern
        avoidance = self._compute_predictive_avoidance(pos, vel)

        # Adaptive Gewichtung in Abhängigkeit des lokalen Zustands
        gradient_strength = np.tanh(np.linalg.norm(grad, axis=1, keepdims=True) * 0.8)
        curiosity_gain = np.array([d.curiosity_gain for d in self.drones], dtype=np.float32)[:, None]
        coherence_gain = np.array([d.coherence_gain for d in self.drones], dtype=np.float32)[:, None]
        avoidance_gain = np.array([d.avoidance_gain for d in self.drones], dtype=np.float32)[:, None]

        w_grad = 0.65 * gradient_strength
        w_cur = np.clip(curiosity_gain * (1.0 - gradient_strength + 0.25), 0.05, 1.5)
        w_coh = np.clip(coherence_gain * (0.5 + gradient_strength * 0.8), 0.0, 2.0)
        w_avoid = np.clip(avoidance_gain * (1.0 + (1.0 - gradient_strength)), 0.1, 2.5)

        # Batterie-Management und Heimkehrlogik
        home_dir, w_home = self._update_energy_and_homeward_drive(pos, vel)
        active_mask = np.array([1.0 if d.active else 0.0 for d in self.drones], dtype=np.float32)[:, None]

        drive = (
            w_grad * grad_dir +
            w_cur * curiosity_dir +
            w_coh * cohesion_dir +
            w_avoid * avoidance +
            w_home * home_dir
        ) * active_mask

        vel = self.inertia * vel + self.step * drive
        vel *= active_mask

        # Maximalgeschwindigkeit (sanfte Kappung)
        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        max_speed = max(0.5, 3.0 * self.step)
        factor = np.minimum(1.0, (max_speed / (speed + 1e-8))).astype(np.float32)
        vel = vel * factor

        # Positionen aktualisieren
        pos = pos + vel

        # Ränder je nach Modus behandeln
        pos, vel = self._apply_boundary_conditions(pos, vel)

        # Zustände zurückschreiben
        for i, d in enumerate(self.drones):
            d.pos[:] = pos[i]
            d.vel[:] = vel[i]

        # Pheromon-Ablage (kleine Gauss-Stempel an jeder Drohnenposition)
        if self.deposit_sigma > 0:
            for i in range(pos.shape[0]):
                drone = self.drones[i]
                if drone.returning_to_base or not drone.active:
                    continue
                sigma = float(max(0.3, drone.deposit_sigma))
                cx, cy = pos[i, 0], pos[i, 1]
                patch, (y0, y1, x0, x1) = _gaussian_stamp((self.h, self.w), cx, cy, sigma=sigma, amp=1.0)
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
