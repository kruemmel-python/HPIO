from __future__ import annotations

from dataclasses import dataclass
import dataclasses as dc
from typing import Callable, Literal, Tuple

import numpy as np

ObjectiveName = Literal["rastrigin", "ackley", "himmelblau"]


# ---------------------------------------------------------------------------
# Konfigurations-Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class FieldParams:
    grid_size: tuple[int, int] = (160, 160)
    relax_alpha: float = 0.25
    evap: float = 0.04
    kernel_sigma: float = 1.6


@dataclass
class AgentParams:
    count: int = 64
    step: float = 0.35
    curiosity: float = 0.45
    momentum: float = 0.65
    # NEU: Breite der Feldablage (Standard so gewählt, dass altes Verhalten nahekommt)
    deposit_sigma: float = 1.6    # in Grid-Pixeln (σ). 0 => Punktablage
    # NEU: Kohärenz in Richtung global best (0..~1). 0 = aus.
    coherence_gain: float = 0.0


@dataclass
class HPIOConfig:
    objective: ObjectiveName
    iters: int = 420
    seed: int = 123
    use_gpu: bool = False
    visualize: bool = False
    bounds: tuple[tuple[float, float], tuple[float, float]] = ((-5.5, 5.5), (-5.5, 5.5))
    report_every: int = 20
    anneal_step_from: float = 1.0
    anneal_step_to: float = 0.2
    anneal_curiosity_from: float = 1.0
    anneal_curiosity_to: float = 0.25
    early_patience: int = 90
    early_tol: float = 1e-4
    polish_h: float = 1e-3
    # NEU: Gewichte für Intensitäts-/Phasenanteil der Ablage
    w_intensity: float = 1.0
    w_phase: float = 0.0
    # NEU: Phasen-Spannweite in Einheiten von π (z. B. 2.4 → 2.4π)
    phase_span_pi: float = 2.0
    field: FieldParams = dc.field(default_factory=FieldParams)
    agent: AgentParams = dc.field(default_factory=AgentParams)


# ---------------------------------------------------------------------------
# Zielfunktionen
# ---------------------------------------------------------------------------
def rastrigin_fn(x: np.ndarray) -> float:
    a = 10.0
    return a * 2 + np.sum(x**2 - a * np.cos(2 * np.pi * x))


def ackley_fn(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    a = 20.0
    b = 0.2
    c = 2 * np.pi
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(s1 / 2.0))
    term2 = -np.exp(s2 / 2.0)
    return term1 + term2 + a + np.e


def himmelblau_fn(x: np.ndarray) -> float:
    x1, x2 = x
    return (x1**2 + x2 - 11) ** 2 + (x1 + x2**2 - 7) ** 2


OBJECTIVES: dict[ObjectiveName, Callable[[np.ndarray], float]] = {
    "rastrigin": rastrigin_fn,
    "ackley": ackley_fn,
    "himmelblau": himmelblau_fn,
}


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    size = max(3, int(round(6 * sigma)))
    if size % 2 == 0:
        size += 1
    radius = size // 2
    xs = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(xs**2) / (2.0 * max(1e-9, sigma)**2))
    kernel /= np.sum(kernel)
    return kernel


def _apply_conv1d(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = len(kernel) // 2
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(arr, pad_width, mode="edge")

    def _conv_line(line: np.ndarray) -> np.ndarray:
        return np.convolve(line, kernel, mode="valid")

    return np.apply_along_axis(_conv_line, axis, padded)


def _stamp_gaussian(grid: np.ndarray, cx: int, cy: int, sigma: float, amount: float) -> None:
    """
    2D-Gaussian um (cx,cy) addieren. sigma in Pixeln. Summe der Maske wird auf 'amount' normiert.
    """
    if sigma <= 0.0:
        # Punktablage (legacy)
        if 0 <= cy < grid.shape[0] and 0 <= cx < grid.shape[1]:
            grid[cy, cx] += amount
        return

    rad = max(1, int(round(3 * sigma)))
    y0, y1 = max(0, cy - rad), min(grid.shape[0], cy + rad + 1)
    x0, x1 = max(0, cx - rad), min(grid.shape[1], cx + rad + 1)
    if y0 >= y1 or x0 >= x1:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1]
    g = np.exp(-(((xx - cx) ** 2) + ((yy - cy) ** 2)) / (2.0 * sigma ** 2))
    s = g.sum()
    if s > 0:
        g *= (amount / s)
        grid[y0:y1, x0:x1] += g


# ---------------------------------------------------------------------------
# Feld- und Agentenimplementierung
# ---------------------------------------------------------------------------
@dataclass
class AgentState:
    pos: np.ndarray
    vel: np.ndarray
    best_pos: np.ndarray
    best_val: float


class Field:
    def __init__(self, params: FieldParams, bounds: tuple[Tuple[float, float], Tuple[float, float]]):
        self.p = params
        self.bounds = bounds
        gy, gx = params.grid_size[1], params.grid_size[0]
        self.phi = np.zeros((gy, gx), dtype=np.float64)
        self._kernel = _gaussian_kernel_1d(max(1e-3, params.kernel_sigma))

    def world_to_grid(self, pos: np.ndarray) -> tuple[int, int]:
        gx = self._grid_x(pos[0])
        gy = self._grid_y(pos[1])
        return gx, gy

    def world_to_grid_float(self, pos: np.ndarray) -> tuple[float, float]:
        return self._grid_xf(pos[0]), self._grid_yf(pos[1])

    def _grid_xf(self, x: float) -> float:
        xmin, xmax = self.bounds[0]
        nx = (x - xmin) / (xmax - xmin)
        nx = np.clip(nx, 0.0, 1.0)
        return nx * (self.p.grid_size[0] - 1)

    def _grid_yf(self, y: float) -> float:
        ymin, ymax = self.bounds[1]
        ny = (y - ymin) / (ymax - ymin)
        ny = np.clip(ny, 0.0, 1.0)
        return ny * (self.p.grid_size[1] - 1)

    def _grid_x(self, x: float) -> int:
        return int(round(self._grid_xf(x)))

    def _grid_y(self, y: float) -> int:
        return int(round(self._grid_yf(y)))

    def deposit(self, pos: np.ndarray, amount: float, sigma_px: float = 0.0) -> None:
        """
        Ablage im Feld. Wenn sigma_px>0 → 2D-Gauss, sonst Punktablage.
        """
        gx, gy = self.world_to_grid(pos)
        gy = int(np.clip(gy, 0, self.phi.shape[0] - 1))
        gx = int(np.clip(gx, 0, self.phi.shape[1] - 1))
        if sigma_px > 0.0:
            _stamp_gaussian(self.phi, gx, gy, sigma=float(sigma_px), amount=float(amount))
        else:
            self.phi[gy, gx] += amount

    def sample_gradient(self, pos: np.ndarray) -> np.ndarray:
        gx, gy = self.world_to_grid_float(pos)
        ix = int(np.clip(round(gx), 1, self.phi.shape[1] - 2))
        iy = int(np.clip(round(gy), 1, self.phi.shape[0] - 2))

        dphidx = (self.phi[iy, ix + 1] - self.phi[iy, ix - 1]) * 0.5
        dphidy = (self.phi[iy + 1, ix] - self.phi[iy - 1, ix]) * 0.5

        xmin, xmax = self.bounds[0]
        ymin, ymax = self.bounds[1]
        scale_x = (self.p.grid_size[0] - 1) / (xmax - xmin)
        scale_y = (self.p.grid_size[1] - 1) / (ymax - ymin)
        return np.array([dphidx * scale_x, dphidy * scale_y], dtype=np.float64)

    def relax(self) -> None:
        blurred = _apply_conv1d(self.phi, self._kernel, axis=1)
        blurred = _apply_conv1d(blurred, self._kernel, axis=0)
        self.phi *= (1.0 - self.p.evap)
        self.phi += self.p.relax_alpha * (blurred - self.phi)


# ---------------------------------------------------------------------------
# Optimierer
# ---------------------------------------------------------------------------
class HPIO:
    def __init__(self, cfg: HPIOConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.f = OBJECTIVES[cfg.objective]
        self.bounds = cfg.bounds
        self.field = Field(cfg.field, bounds=cfg.bounds)
        self.agents: list[AgentState] = []
        self._init_agents()
        self.gbest_pos = self.agents[0].best_pos.copy()
        self.gbest_val = self.agents[0].best_val
        for a in self.agents:
            if a.best_val < self.gbest_val:
                self.gbest_val = a.best_val
                self.gbest_pos = a.best_pos.copy()

    def _init_agents(self) -> None:
        (xmin, xmax), (ymin, ymax) = self.bounds
        for _ in range(self.cfg.agent.count):
            pos = np.array([
                self.rng.uniform(xmin, xmax),
                self.rng.uniform(ymin, ymax),
            ], dtype=np.float64)
            val = float(self.f(pos))
            agent = AgentState(
                pos=pos,
                vel=np.zeros(2, dtype=np.float64),
                best_pos=pos.copy(),
                best_val=val,
            )
            self.agents.append(agent)

    def _clip_position(self, pos: np.ndarray) -> np.ndarray:
        (xmin, xmax), (ymin, ymax) = self.bounds
        pos[0] = np.clip(pos[0], xmin, xmax)
        pos[1] = np.clip(pos[1], ymin, ymax)
        return pos

    def _deposit_from_agent(self, agent: AgentState, amplitude: float) -> None:
        """
        Ablage kombiniert Intensitäts- und (optional) Phasenanteil.
        - Intensität:  1/(1+best_val) → bessere Agenten legen stärker ab.
        - Phase: leichte Modulation anhand Bewegungsrichtung (sinusförmig),
                 skaliert über phase_span_pi und w_phase.
        """
        # Intensitätsanteil
        base = amplitude * (1.0 / (1.0 + agent.best_val))

        # Phasenanteil (richtungsabhängig – optional)
        phase_term = 0.0
        if self.cfg.w_phase != 0.0:
            v = agent.vel
            vn = np.linalg.norm(v)
            if vn > 1e-12:
                # Winkel ∈ (-π, π]
                ang = float(np.arctan2(v[1], v[0]))
                # Map auf [-phase_span_pi*π, +phase_span_pi*π] durch Skalierung
                span = float(self.cfg.phase_span_pi) * np.pi
                phase_term = np.sin(ang * (span / np.pi))  # = sin(ang * phase_span_pi)
            # Leicht dämpfen, damit Phase nicht dominiert
            phase_term *= 0.5 * base

        amount = self.cfg.w_intensity * base + self.cfg.w_phase * phase_term

        # Ablage-Fußabdruck
        sigma_px = max(0.0, float(self.cfg.agent.deposit_sigma))
        self.field.deposit(agent.best_pos, float(amount), sigma_px=sigma_px)

    def _move_agent(self, agent: AgentState, *, step_scale: float, curiosity_scale: float) -> None:
        grad = self.field.sample_gradient(agent.pos)
        gnorm = np.linalg.norm(grad)
        if gnorm > 1e-9:
            grad = grad / gnorm
        random_vec = self.rng.normal(size=2)
        random_vec /= np.linalg.norm(random_vec) + 1e-9

        target_step = self.cfg.agent.step * step_scale
        curiosity = self.cfg.agent.curiosity * curiosity_scale
        momentum = self.cfg.agent.momentum

        # NEU: Kohärenz in Richtung global best (hilft „Einrasten“)
        coh = 0.0
        if self.cfg.agent.coherence_gain > 0.0:
            to_gbest = self.gbest_pos - agent.pos
            ng = np.linalg.norm(to_gbest)
            if ng > 1e-12:
                coh_vec = to_gbest / ng
                coh = self.cfg.agent.coherence_gain
            else:
                coh_vec = np.zeros(2, dtype=np.float64)
                coh = 0.0
        else:
            coh_vec = np.zeros(2, dtype=np.float64)

        # Bewegungsrichtung: runter gegen Gradienten, plus Neugier, plus Kohärenz
        direction = -grad * target_step + curiosity * random_vec + coh * target_step * coh_vec
        agent.vel = momentum * agent.vel + (1.0 - momentum) * direction
        agent.pos = self._clip_position(agent.pos + agent.vel)

        val = float(self.f(agent.pos))
        if val < agent.best_val:
            agent.best_val = val
            agent.best_pos = agent.pos.copy()
            if val < self.gbest_val:
                self.gbest_val = val
                self.gbest_pos = agent.pos.copy()

    def run(self) -> tuple[np.ndarray, float]:
        no_improve = 0
        last_best = self.gbest_val

        for it in range(self.cfg.iters):
            t = it / max(1, self.cfg.iters - 1)
            step_scale = (1.0 - t) * self.cfg.anneal_step_from + t * self.cfg.anneal_step_to
            curiosity_scale = (1.0 - t) * self.cfg.anneal_curiosity_from + t * self.cfg.anneal_curiosity_to

            vals = np.array([a.best_val for a in self.agents], dtype=np.float64)
            ranks = np.argsort(np.argsort(vals))
            if len(self.agents) > 1:
                norm = (len(self.agents) - 1)
                amp = 0.8 + 0.5 * (1.0 - ranks / norm)
            else:
                amp = np.array([1.0])

            for agent, amp_scale in zip(self.agents, amp):
                self._deposit_from_agent(agent, float(amp_scale))

            self.field.relax()

            for agent in self.agents:
                self._move_agent(agent, step_scale=step_scale, curiosity_scale=curiosity_scale)

            if self.gbest_val < last_best - self.cfg.early_tol:
                last_best = self.gbest_val
                no_improve = 0
            else:
                no_improve += 1

            if no_improve > self.cfg.early_patience:
                break

        pos, val = self.local_quadratic_polish(self.f, self.gbest_pos, h=self.cfg.polish_h)
        if val < self.gbest_val:
            self.gbest_pos = pos
            self.gbest_val = val

        return self.gbest_pos.copy(), float(self.gbest_val)

    @staticmethod
    def local_quadratic_polish(f: Callable[[np.ndarray], float], pos: np.ndarray, h: float = 1e-3) -> tuple[np.ndarray, float]:
        pos = np.asarray(pos, dtype=np.float64)
        h = float(max(1e-6, h))

        def eval_at(offset: tuple[float, float]) -> float:
            return float(f(pos + np.array(offset, dtype=np.float64)))

        f00 = eval_at((0.0, 0.0))
        fx1 = eval_at((h, 0.0))
        fx_1 = eval_at((-h, 0.0))
        fy1 = eval_at((0.0, h))
        fy_1 = eval_at((0.0, -h))
        fxy = eval_at((h, h))
        fx_y = eval_at((h, -h))
        f_x_y = eval_at((-h, -h))
        f_xy = eval_at((-h, h))

        grad_x = (fx1 - fx_1) / (2 * h)
        grad_y = (fy1 - fy_1) / (2 * h)

        hxx = (fx1 - 2 * f00 + fx_1) / (h**2)
        hyy = (fy1 - 2 * f00 + fy_1) / (h**2)
        hxy = (fxy - fx_y - f_xy + f_x_y) / (4 * h**2)

        hessian = np.array([[hxx, hxy], [hxy, hyy]], dtype=np.float64)
        grad = np.array([grad_x, grad_y], dtype=np.float64)

        try:
            delta = np.linalg.solve(hessian, grad)
            cand = pos - delta
            val = float(f(cand))
            return cand, val
        except np.linalg.LinAlgError:
            return pos.copy(), float(f00)


# ---------------------------------------------------------------------------
# Konfigurations-Factory
# ---------------------------------------------------------------------------
def build_config(objective: ObjectiveName, *, use_gpu: bool = False, visualize: bool = False) -> HPIOConfig:
    if objective not in OBJECTIVES:
        raise ValueError(f"Unbekannte Zielfunktion: {objective}")

    cfg = HPIOConfig(objective=objective)
    cfg.use_gpu = use_gpu
    cfg.visualize = visualize

    if objective == "rastrigin":
        cfg.bounds = ((-5.12, 5.12), (-5.12, 5.12))
        cfg.iters = 420
        cfg.agent.count = 80
        cfg.field.grid_size = (180, 180)
        cfg.anneal_step_from = 1.1
        cfg.anneal_step_to = 0.18
        cfg.anneal_curiosity_from = 0.9
        cfg.anneal_curiosity_to = 0.2
        # etwas Phasenführung hilft gegen 1.0-Ring
        cfg.w_intensity = 1.0
        cfg.w_phase = 0.0
        cfg.phase_span_pi = 2.2

    elif objective == "ackley":
        cfg.bounds = ((-5.0, 5.0), (-5.0, 5.0))
        cfg.iters = 380
        cfg.agent.count = 72
        cfg.field.grid_size = (170, 170)
        cfg.anneal_step_from = 0.9
        cfg.anneal_step_to = 0.22
        cfg.anneal_curiosity_from = 1.0
        cfg.anneal_curiosity_to = 0.3
        # Ackley profitiert von etwas Phase, kann aber bei 0.0 starten
        cfg.w_intensity = 1.0
        cfg.w_phase = 0.0
        cfg.phase_span_pi = 2.4

    elif objective == "himmelblau":
        cfg.bounds = ((-6.0, 6.0), (-6.0, 6.0))
        cfg.iters = 320
        cfg.agent.count = 60
        cfg.field.grid_size = (150, 150)
        cfg.anneal_step_from = 0.8
        cfg.anneal_step_to = 0.2
        cfg.anneal_curiosity_from = 0.8
        cfg.anneal_curiosity_to = 0.25
        cfg.early_patience = 60
        cfg.early_tol = 1e-8
        cfg.polish_h = 1e-3
        cfg.w_intensity = 1.0
        cfg.w_phase = 0.0
        cfg.phase_span_pi = 2.0

    return cfg


__all__ = [
    "AgentState",
    "Field",
    "FieldParams",
    "AgentParams",
    "HPIOConfig",
    "HPIO",
    "build_config",
]
