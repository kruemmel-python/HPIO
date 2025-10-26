"""Streamlit app for interactive HPIO experimentation and visualization."""
from __future__ import annotations

import csv
import io
import json
import math
import time
from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Any, Optional

import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from pathlib import Path

from hpio import AgentParams, FieldParams, HPIO, HPIOConfig, build_config


MAX_VIDEO_FRAMES = 5000


# ---------------------------------------------------------------------------
# Helpers: configuration conversion and defaults
# ---------------------------------------------------------------------------
@dataclass
class StepResult:
    iteration: int
    best_val: float
    best_pos: np.ndarray
    delta_best: float
    field_phi: np.ndarray
    agents_grid: np.ndarray
    agents_world: np.ndarray
    elapsed: float
    total_time: float
    early_stop_triggered: bool


@dataclass
class HPIOController:
    cfg: HPIOConfig
    optimizer: HPIO = field(init=False)
    iteration: int = 0
    no_improve: int = 0
    last_best: float = field(init=False)
    start_ts: float = field(default_factory=time.perf_counter)
    total_time: float = 0.0
    best_history: list[tuple[int, float, np.ndarray]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.optimizer = HPIO(self.cfg)
        self.last_best = float(self.optimizer.gbest_val)
        self.best_history.append((0, float(self.optimizer.gbest_val), self.optimizer.gbest_pos.copy()))

    def step(self) -> Optional[StepResult]:
        if self.iteration >= self.cfg.iters:
            return None

        t0 = time.perf_counter()
        opt = self.optimizer
        cfg = opt.cfg

        t = self.iteration / max(1, cfg.iters - 1)
        step_scale = (1.0 - t) * cfg.anneal_step_from + t * cfg.anneal_step_to
        curiosity_scale = (1.0 - t) * cfg.anneal_curiosity_from + t * cfg.anneal_curiosity_to

        vals = np.array([a.best_val for a in opt.agents], dtype=np.float64)
        if len(vals) > 0:
            ranks = np.argsort(np.argsort(vals))
            norm = max(1, len(vals) - 1)
            amp = 0.8 + 0.5 * (1.0 - ranks / norm)
        else:
            amp = np.array([], dtype=np.float64)

        for agent, amp_scale in zip(opt.agents, amp):
            opt._deposit_from_agent(agent, float(amp_scale))

        opt.field.relax()

        for agent in opt.agents:
            opt._move_agent(agent, step_scale=step_scale, curiosity_scale=curiosity_scale)

        improved = False
        if opt.gbest_val < self.last_best - cfg.early_tol:
            self.last_best = float(opt.gbest_val)
            self.no_improve = 0
            improved = True
        else:
            self.no_improve += 1

        self.iteration += 1
        dt = time.perf_counter() - t0
        self.total_time = time.perf_counter() - self.start_ts

        agents_world = np.array([a.pos for a in opt.agents], dtype=np.float64)
        agents_grid = np.array([opt.field.world_to_grid_float(a.pos) for a in opt.agents], dtype=np.float64)

        delta_best = float(opt.gbest_val - self.best_history[-1][1])
        self.best_history.append((self.iteration, float(opt.gbest_val), opt.gbest_pos.copy()))

        early_stop = self.no_improve > cfg.early_patience
        if early_stop or self.iteration >= cfg.iters:
            pos, val = opt.local_quadratic_polish(opt.f, opt.gbest_pos, h=cfg.polish_h)
            if val < opt.gbest_val:
                opt.gbest_pos = pos
                opt.gbest_val = val

        return StepResult(
            iteration=self.iteration,
            best_val=float(opt.gbest_val),
            best_pos=opt.gbest_pos.copy(),
            delta_best=float(delta_best),
            field_phi=opt.field.phi.copy(),
            agents_grid=agents_grid,
            agents_world=agents_world,
            elapsed=dt,
            total_time=self.total_time,
            early_stop_triggered=early_stop and not improved,
        )

    def reset(self, keep_seed: bool = True) -> None:
        rng = np.random.default_rng()
        if not keep_seed:
            self.cfg.seed = int(rng.integers(0, 2**32 - 1))
        self.optimizer = HPIO(self.cfg)
        self.iteration = 0
        self.no_improve = 0
        self.last_best = float(self.optimizer.gbest_val)
        self.start_ts = time.perf_counter()
        self.total_time = 0.0
        self.best_history = [(0, float(self.optimizer.gbest_val), self.optimizer.gbest_pos.copy())]


@dataclass
class AppState:
    cfg: HPIOConfig
    controller: Optional[HPIOController] = None
    running: bool = False
    paused: bool = False
    viz_every: int = 1
    trail_length: int = 60
    overlay: bool = True
    logs: list[str] = field(default_factory=list)
    last_result: Optional[StepResult] = None
    last_plot_png: Optional[bytes] = None
    fps: float = 0.0
    video_active: bool = False
    video_params: dict[str, Any] = field(default_factory=dict)
    video_frames: list[tuple[int, bytes]] = field(default_factory=list)
    video_limit_notified: bool = False
    video_last_path: Optional[str] = None
    video_progress_total: int = 0
    video_summary: Optional[dict[str, Any]] = None
    parameter_dirty: bool = False
    experiment_results: dict[str, Any] = field(default_factory=dict)



def gpu_available() -> bool:
    try:
        import pyopencl as cl  # type: ignore

        platforms = cl.get_platforms()
        return bool(platforms)
    except Exception:
        return False


def trigger_rerun() -> None:
    """Request a rerun using the available Streamlit API."""

    rerun_fn = getattr(st, "rerun", None)
    if rerun_fn is None:
        rerun_fn = getattr(st, "experimental_rerun", None)

    if rerun_fn is None:
        raise RuntimeError("Streamlit rerun API unavailable in this version")

    rerun_fn()


def get_state() -> AppState:
    if "app_state" not in st.session_state:
        base_cfg = build_config("rastrigin")
        st.session_state.app_state = AppState(cfg=base_cfg)
    state = st.session_state.app_state
    if not hasattr(state, "video_limit_notified"):
        state.video_limit_notified = False
    return state


def config_to_nested(cfg: HPIOConfig) -> dict[str, Any]:
    data = asdict(cfg)
    data["field"] = asdict(cfg.field)
    data["agent"] = asdict(cfg.agent)
    return data


def nested_to_config(data: dict[str, Any]) -> HPIOConfig:
    field_data = data.get("field", {})
    agent_data = data.get("agent", {})
    cfg = HPIOConfig(
        objective=data.get("objective", "rastrigin"),
        iters=int(data.get("iters", 420)),
        seed=int(data.get("seed", 123)),
        use_gpu=bool(data.get("use_gpu", False)),
        visualize=bool(data.get("visualize", False)),
        bounds=tuple(tuple(x) for x in data.get("bounds", ((-5.12, 5.12), (-5.12, 5.12)))),
        report_every=int(data.get("report_every", 20)),
        anneal_step_from=float(data.get("anneal_step_from", 1.0)),
        anneal_step_to=float(data.get("anneal_step_to", 0.2)),
        anneal_curiosity_from=float(data.get("anneal_curiosity_from", 1.0)),
        anneal_curiosity_to=float(data.get("anneal_curiosity_to", 0.25)),
        early_patience=int(data.get("early_patience", 90)),
        early_tol=float(data.get("early_tol", 1e-4)),
        polish_h=float(data.get("polish_h", 1e-3)),
        w_intensity=float(data.get("w_intensity", 1.0)),
        w_phase=float(data.get("w_phase", 0.0)),
        phase_span_pi=float(data.get("phase_span_pi", 2.0)),
    )
    cfg.field = FieldParams(
        grid_size=tuple(field_data.get("grid_size", (160, 160))),
        relax_alpha=float(field_data.get("relax_alpha", 0.25)),
        evap=float(field_data.get("evap", 0.04)),
        kernel_sigma=float(field_data.get("kernel_sigma", 1.6)),
    )
    cfg.agent = AgentParams(
        count=int(agent_data.get("count", 64)),
        step=float(agent_data.get("step", 0.35)),
        curiosity=float(agent_data.get("curiosity", 0.45)),
        momentum=float(agent_data.get("momentum", 0.65)),
        deposit_sigma=float(agent_data.get("deposit_sigma", 1.6)),
        coherence_gain=float(agent_data.get("coherence_gain", 0.0)),
    )
    return cfg


def render_heatmap(
    phi: np.ndarray,
    agents_grid: np.ndarray,
    trails: list[list[tuple[float, float]]],
    iteration: int,
    best_val: float,
    overlay: bool,
    trail_length: int,
) -> bytes:
    fig, ax = plt.subplots(figsize=(6, 6))
    data = np.log(np.abs(phi) + 1e-6)
    im = ax.imshow(data, origin="lower", cmap="inferno")
    if agents_grid.size:
        ax.scatter(
            agents_grid[:, 0],
            agents_grid[:, 1],
            s=22,
            facecolors="#22d3ee",
            edgecolors="#0f172a",
            linewidths=0.4,
        )
    for path in trails:
        if len(path) < 2:
            continue
        xs, ys = zip(*path[-trail_length:])
        ax.plot(xs, ys, color="#38bdf8", linewidth=0.9, alpha=0.7)
    if overlay:
        ax.set_title(f"Iteration {iteration} • Best = {best_val:.4f}")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log|Φ|")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def update_trails(
    trails: list[list[tuple[float, float]]],
    agents_grid: np.ndarray,
    trail_length: int,
) -> list[list[tuple[float, float]]]:
    if not trails or len(trails) != len(agents_grid):
        trails = [[tuple(p)] for p in agents_grid.tolist()]
    else:
        for idx, point in enumerate(agents_grid.tolist()):
            trails[idx].append(tuple(point))
            if len(trails[idx]) > trail_length:
                trails[idx] = trails[idx][-trail_length:]
    return trails


def append_log(state: AppState, message: str) -> None:
    state.logs.append(message)
    if len(state.logs) > 5000:
        state.logs = state.logs[-5000:]


def ensure_controller(state: AppState) -> None:
    if state.controller is None or state.parameter_dirty:
        state.controller = HPIOController(state.cfg)
        state.parameter_dirty = False
        state.logs = ["Controller neu initialisiert"]
        state.last_plot_png = None
        state.last_result = None
        state.video_limit_notified = False
        state.video_last_path = None
        state.video_progress_total = 0
        state.video_summary = None


def ensure_video_defaults(state: AppState) -> None:
    defaults = {
        "filename": f"{state.cfg.objective}_seed{state.cfg.seed}.mp4",
        "format": "mp4",
        "fps": 30,
        "viz_freq": max(1, state.viz_every),
        "overlay": state.overlay,
        "encoder": "medium",
        "crf": 23,
    }
    if not state.video_params:
        state.video_params = defaults.copy()
        return
    for key, value in defaults.items():
        state.video_params.setdefault(key, value)


def compute_expected_video_frames(state: AppState) -> int:
    ensure_video_defaults(state)
    video_freq = int(state.video_params.get("viz_freq", state.viz_every) or state.viz_every)
    video_freq = max(1, video_freq)
    total_iters = state.cfg.iters
    if state.controller is not None:
        total_iters = state.controller.cfg.iters
    expected = math.ceil(total_iters / video_freq)
    return max(1, int(expected))


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} {units[-1]}"


def execute_step(state: AppState) -> None:
    if state.controller is None:
        return
    result = state.controller.step()
    if result is None:
        state.running = False
        append_log(state, "Lauf beendet")
        return

    state.last_result = result
    fps = 1.0 / result.elapsed if result.elapsed > 1e-9 else float("inf")
    state.fps = fps
    message = (
        f"iter={result.iteration:04d} | best={result.best_val:.6f} | Δbest={result.delta_best:+.3e} | "
        f"pos=({result.best_pos[0]:.3f}, {result.best_pos[1]:.3f})"
    )
    append_log(state, message)
    if result.early_stop_triggered:
        append_log(state, "⏹️ Early-Stopping ausgelöst")
        state.running = False

    trails = getattr(state, "_trails", [])
    trails = update_trails(trails, result.agents_grid, state.trail_length)
    state._trails = trails

    should_render = (result.iteration % state.viz_every) == 0 or not state.last_plot_png
    if should_render:
        state.last_plot_png = render_heatmap(
            result.field_phi,
            result.agents_grid,
            trails,
            result.iteration,
            result.best_val,
            state.overlay,
            state.trail_length,
        )
        if state.video_active and state.last_plot_png:
            video_freq = int(state.video_params.get("viz_freq", state.viz_every) or state.viz_every)
            video_freq = max(1, video_freq)
            if result.iteration % video_freq == 0:
                if len(state.video_frames) >= MAX_VIDEO_FRAMES:
                    if not state.video_limit_notified:
                        append_log(state, "⚠️ Frame-Limit erreicht – älteste Frames werden überschrieben.")
                        state.video_limit_notified = True
                    state.video_frames.pop(0)
                state.video_frames.append((result.iteration, state.last_plot_png))


def render_status_box(result: Optional[StepResult], state: AppState) -> None:
    st.markdown("### Status")
    cols = st.columns(2)
    if result is None:
        cols[0].metric("Iteration", state.controller.iteration if state.controller else 0)
        cols[0].metric("Best Value", "–")
    else:
        cols[0].metric("Iteration", result.iteration)
        cols[0].metric("Best Value", f"{result.best_val:.6f}", delta=f"{result.delta_best:+.3e}")

    if result is not None:
        cols[1].metric(
            "Best Position",
            f"({result.best_pos[0]:.3f}, {result.best_pos[1]:.3f})",
        )
        cols[1].metric("Δ Best", f"{result.delta_best:+.3e}")
    else:
        cols[1].metric("Best Position", "–")
        cols[1].metric("Δ Best", "–")

    if result is not None:
        st.caption(
            f"Zeit pro Iteration: {result.elapsed * 1000:.2f} ms • Gesamtzeit: {result.total_time:.2f} s • FPS: {state.fps:.1f}"
        )
    elif state.controller:
        st.caption(f"Gesamtzeit: {state.controller.total_time:.2f} s")


def render_console(logs: list[str]) -> None:
    st.markdown("### Konsole")
    st.text_area(
        "Konsole",
        value="\n".join(logs[-400:]),
        height=360,
        label_visibility="collapsed",
        disabled=True,
    )


def render_parameter_snapshot(cfg: HPIOConfig) -> None:
    st.markdown("### Parameter Snapshot")
    snapshot = {
        "objective": cfg.objective,
        "iters": cfg.iters,
        "seed": cfg.seed,
        "field": {
            "grid": cfg.field.grid_size,
            "relax_alpha": cfg.field.relax_alpha,
            "evap": cfg.field.evap,
            "kernel_sigma": cfg.field.kernel_sigma,
        },
        "agent": {
            "count": cfg.agent.count,
            "step": cfg.agent.step,
            "curiosity": cfg.agent.curiosity,
            "momentum": cfg.agent.momentum,
            "deposit_sigma": cfg.agent.deposit_sigma,
            "coherence_gain": cfg.agent.coherence_gain,
        },
        "weights": {
            "w_intensity": cfg.w_intensity,
            "w_phase": cfg.w_phase,
            "phase_span_pi": cfg.phase_span_pi,
        },
        "anneal": {
            "step": (cfg.anneal_step_from, cfg.anneal_step_to),
            "curiosity": (cfg.anneal_curiosity_from, cfg.anneal_curiosity_to),
        },
        "early_stop": {
            "patience": cfg.early_patience,
            "tol": cfg.early_tol,
        },
        "polish_h": cfg.polish_h,
    }
    st.json(snapshot)


# ---------------------------------------------------------------------------
# Page: Start / Run
# ---------------------------------------------------------------------------

def page_run(state: AppState) -> None:
    st.markdown("## 🎬 Start / Run")
    cfg = state.cfg

    available_gpu = gpu_available()

    if state.parameter_dirty:
        st.markdown(
            "<span style='background-color:#fbbf24; color:#0f172a; padding:4px 10px; "
            "border-radius:999px; font-weight:600;'>Parameter geändert</span> "
            "Reset oder neuer Start aktualisiert den Lauf.",
            unsafe_allow_html=True,
        )

    with st.sidebar:
        st.markdown("### Basis-Setup")
        options = ["rastrigin", "ackley", "himmelblau"]
        objective = st.selectbox(
            "Zielfunktion",
            options=options,
            index=options.index(cfg.objective),
        )
        if objective != cfg.objective:
            state.cfg = build_config(objective)
            cfg = state.cfg
            state.parameter_dirty = True
            state.controller = None

        gpu_toggle = st.checkbox(
            "GPU (PyOpenCL)",
            value=cfg.use_gpu and available_gpu,
            disabled=not available_gpu,
        )
        cfg.use_gpu = bool(gpu_toggle and available_gpu)
        if not available_gpu:
            st.info("PyOpenCL/GPU nicht verfügbar oder kein Gerät gefunden – Lauf erfolgt auf CPU.")

        st.markdown("### Seed & Iterationen")
        col_seed, col_button = st.columns([3, 1])
        new_seed = col_seed.number_input("Seed", value=int(cfg.seed), step=1, format="%d")
        cfg.seed = int(new_seed)
        if col_button.button("🎲 Zufalls-Seed"):
            cfg.seed = int(np.random.default_rng().integers(0, 2**32 - 1))
            append_log(state, f"Seed aktualisiert → {cfg.seed}")
            trigger_rerun()

        cfg.iters = int(
            st.number_input(
                "Iterationen",
                value=int(cfg.iters),
                min_value=1,
                max_value=20000,
                step=10,
            )
        )

        st.markdown("### Visualisierung")
        st.caption("FPS ist nur Anzeige – Rendering jeder n-ten Iteration")
        state.viz_every = int(
            st.number_input(
                "Viz-Frequenz",
                value=int(state.viz_every),
                min_value=1,
                max_value=50,
                step=1,
            )
        )
        state.overlay = st.checkbox("Overlay (Iteration / Bestwert)", value=state.overlay)
        state.trail_length = int(
            st.slider(
                "Traillänge",
                min_value=5,
                max_value=200,
                value=int(state.trail_length),
                step=5,
            )
        )

        st.markdown("### Run-Kontrollen")
        colA, colB, colC = st.columns(3)
        if colA.button("Start", disabled=state.running):
            state.controller = HPIOController(cfg)
            state.running = True
            state.paused = False
            state.logs = ["Run initialisiert"]
            state.video_frames = []
            state.video_limit_notified = False
            state.last_result = None
            state._trails = []
            state.video_summary = None
            trigger_rerun()
        if colB.button("Pause" if not state.paused else "Weiter", disabled=not state.running):
            if state.controller:
                state.paused = not state.paused
                trigger_rerun()
        if colC.button("Stop", disabled=not state.running):
            state.running = False
            state.paused = False
            trigger_rerun()

        col_step, col_reset, col_seed_reset = st.columns(3)
        if col_step.button("Schritt vor", disabled=state.running and not state.paused):
            ensure_controller(state)
            execute_step(state)
        if col_reset.button("Reset"):
            if state.controller:
                state.controller.reset(keep_seed=True)
                state.logs.append("Reset durchgeführt")
                state.last_result = None
                state.last_plot_png = None
                state._trails = []
                state.video_limit_notified = False
                state.video_summary = None
                trigger_rerun()
        if col_seed_reset.button("Reset + neuer Seed"):
            ensure_controller(state)
            if state.controller:
                state.controller.reset(keep_seed=False)
                append_log(state, "Reset mit neuem Seed")
                state.last_result = None
                state.last_plot_png = None
                state._trails = []
                state.video_limit_notified = False
                state.video_summary = None
                trigger_rerun()

    if state.running and not state.paused:
        ensure_controller(state)
        execute_step(state)

    left, right = st.columns([2.5, 1.5])
    with left:
        st.markdown("### Heatmap & Agents")
        if state.last_plot_png:
            st.image(state.last_plot_png, width="stretch")
        else:
            st.info("Noch keine Visualisierung – Run starten oder Schritt ausführen.")
        render_parameter_snapshot(cfg)

    with right:
        render_status_box(state.last_result, state)
        render_console(state.logs)

    if state.running and not state.paused:
        delay = 0.01
        if state.fps > 60.0:
            delay = 0.03
        time.sleep(delay)
        trigger_rerun()


# ---------------------------------------------------------------------------
# Page: Parameters
# ---------------------------------------------------------------------------

def page_parameters(state: AppState) -> None:
    st.markdown("## ⚙️ Parameter: Feld & Agenten")
    cfg = state.cfg

    st.markdown("### Feld")
    with st.form("field_form"):
        c1, c2 = st.columns(2)
        grid_w = c1.number_input(
            "Grid Breite",
            value=int(cfg.field.grid_size[0]),
            min_value=16,
            max_value=512,
            step=2,
        )
        grid_h = c2.number_input(
            "Grid Höhe",
            value=int(cfg.field.grid_size[1]),
            min_value=16,
            max_value=512,
            step=2,
        )
        if grid_w > 256 or grid_h > 256:
            st.warning("Große Grids (>256²) können sehr langsam sein.")
        relax_alpha = st.slider(
            "relax_alpha",
            min_value=0.0,
            max_value=1.0,
            value=float(cfg.field.relax_alpha),
            step=0.01,
        )
        evap = st.slider(
            "evap",
            min_value=0.0,
            max_value=1.0,
            value=float(cfg.field.evap),
            step=0.01,
        )
        kernel_sigma = st.number_input(
            "kernel_sigma",
            value=float(cfg.field.kernel_sigma),
            min_value=0.1,
            max_value=10.0,
            step=0.1,
        )
        submitted = st.form_submit_button("Übernehmen")
        if submitted:
            cfg.field = FieldParams(
                grid_size=(int(grid_w), int(grid_h)),
                relax_alpha=float(relax_alpha),
                evap=float(evap),
                kernel_sigma=float(kernel_sigma),
            )
            state.parameter_dirty = True
            st.success("Feldparameter aktualisiert – Reset oder Start für neuen Lauf.")

    st.markdown("### Agenten & Ablage")
    with st.form("agents_form"):
        count = st.number_input("count", value=int(cfg.agent.count), min_value=1, max_value=2000, step=1)
        step_val = st.number_input("step", value=float(cfg.agent.step), min_value=0.01, max_value=5.0, step=0.05)
        curiosity = st.number_input("curiosity", value=float(cfg.agent.curiosity), min_value=0.0, max_value=5.0, step=0.05)
        momentum = st.slider("momentum", min_value=0.0, max_value=0.99, value=float(cfg.agent.momentum), step=0.01)
        deposit_sigma = st.number_input("deposit_sigma", value=float(cfg.agent.deposit_sigma), min_value=0.0, max_value=10.0, step=0.1)
        coherence_gain = st.slider("coherence_gain", min_value=0.0, max_value=1.0, value=float(cfg.agent.coherence_gain), step=0.01)
        w_intensity = st.number_input("w_intensity", value=float(cfg.w_intensity), min_value=0.0, max_value=10.0, step=0.1)
        w_phase = st.number_input("w_phase", value=float(cfg.w_phase), min_value=0.0, max_value=10.0, step=0.1)
        phase_span_pi = st.number_input("phase_span_pi", value=float(cfg.phase_span_pi), min_value=0.0, max_value=10.0, step=0.1)
        anneal_step_from = st.number_input("anneal_step_from", value=float(cfg.anneal_step_from), min_value=0.0, max_value=5.0, step=0.05)
        anneal_step_to = st.number_input("anneal_step_to", value=float(cfg.anneal_step_to), min_value=0.0, max_value=5.0, step=0.05)
        anneal_curiosity_from = st.number_input("anneal_curiosity_from", value=float(cfg.anneal_curiosity_from), min_value=0.0, max_value=5.0, step=0.05)
        anneal_curiosity_to = st.number_input("anneal_curiosity_to", value=float(cfg.anneal_curiosity_to), min_value=0.0, max_value=5.0, step=0.05)
        early_patience = st.number_input("early_patience", value=int(cfg.early_patience), min_value=1, max_value=2000, step=1)
        early_tol = st.number_input("early_tol", value=float(cfg.early_tol), min_value=1e-8, max_value=1.0, step=1e-4, format="%.6f")
        polish_h = st.number_input("polish_h", value=float(cfg.polish_h), min_value=1e-6, max_value=0.1, step=1e-4, format="%.6f")
        submitted_agents = st.form_submit_button("Übernehmen")
        if submitted_agents:
            cfg.agent = AgentParams(
                count=int(count),
                step=float(step_val),
                curiosity=float(curiosity),
                momentum=float(momentum),
                deposit_sigma=float(deposit_sigma),
                coherence_gain=float(coherence_gain),
            )
            cfg.w_intensity = float(w_intensity)
            cfg.w_phase = float(w_phase)
            cfg.phase_span_pi = float(phase_span_pi)
            cfg.anneal_step_from = float(anneal_step_from)
            cfg.anneal_step_to = float(anneal_step_to)
            cfg.anneal_curiosity_from = float(anneal_curiosity_from)
            cfg.anneal_curiosity_to = float(anneal_curiosity_to)
            cfg.early_patience = int(early_patience)
            cfg.early_tol = float(early_tol)
            cfg.polish_h = float(polish_h)
            state.parameter_dirty = True
            st.success("Agentenparameter aktualisiert.")

    col_default, col_apply = st.columns(2)
    if col_default.button("Auf Defaults zurücksetzen"):
        state.cfg = build_config(state.cfg.objective)
        state.parameter_dirty = True
        state.controller = None
        trigger_rerun()
    if col_apply.button("Auf Preset übertragen"):
        state.parameter_dirty = True
        st.info("Preset-Übernahme erfolgt über die Preset-Seite.")


# ---------------------------------------------------------------------------
# Page: Presets
# ---------------------------------------------------------------------------

def load_presets() -> dict[str, HPIOConfig]:
    presets = {
        "rastrigin-gpu-pro": build_config("rastrigin", use_gpu=gpu_available()),
        "ackley-gpu-pro": build_config("ackley", use_gpu=gpu_available()),
        "himmelblau-cpu-pro": build_config("himmelblau", use_gpu=False),
    }
    for cfg in presets.values():
        cfg.visualize = True
    return presets


def diff_configs(cfg_a: HPIOConfig, cfg_b: HPIOConfig) -> list[tuple[str, Any, Any]]:
    diff: list[tuple[str, Any, Any]] = []
    dict_a = config_to_nested(cfg_a)
    dict_b = config_to_nested(cfg_b)

    def walk(prefix: str, da: Any, db: Any) -> None:
        if isinstance(da, dict) and isinstance(db, dict):
            for key in sorted(set(da) | set(db)):
                walk(f"{prefix}.{key}" if prefix else key, da.get(key), db.get(key))
        else:
            if da != db:
                diff.append((prefix, da, db))

    walk("", dict_a, dict_b)
    return diff


def config_to_cli(cfg: HPIOConfig) -> str:
    parts = ["python hpio_record.py", cfg.objective]
    if cfg.use_gpu:
        parts.append("--gpu")
    parts += ["--video", f"runs/{cfg.objective}.mp4"]
    parts += ["--fps", "30", "--size", "1280x720"]
    parts += ["--viz-freq", "1", "--seed", str(cfg.seed)]
    return " ".join(parts)


def page_presets(state: AppState) -> None:
    st.markdown("## 🎛️ Presets")
    presets = load_presets()
    if "custom_presets" not in st.session_state:
        st.session_state.custom_presets = {}
    custom_presets: dict[str, HPIOConfig] = st.session_state.custom_presets

    preset_names = list(presets.keys()) + list(custom_presets.keys())
    preset_choice = st.selectbox(
        "Preset wählen",
        options=preset_names,
        index=preset_names.index("rastrigin-gpu-pro") if "rastrigin-gpu-pro" in preset_names else 0,
    )
    chosen_cfg = presets.get(preset_choice) or custom_presets[preset_choice]

    col_apply, col_download = st.columns(2)
    if col_apply.button("Preset anwenden"):
        state.cfg = nested_to_config(config_to_nested(chosen_cfg))
        state.parameter_dirty = True
        state.controller = None
        st.success(f"Preset '{preset_choice}' angewendet.")
    if col_download.button("Preset speichern (JSON)"):
        payload = json.dumps(config_to_nested(state.cfg), indent=2)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        st.download_button(
            "Download JSON",
            data=payload,
            file_name=f"{state.cfg.objective}_preset_{stamp}.json",
            mime="application/json",
        )

    uploaded = st.file_uploader("Preset laden (JSON)", type="json")
    if uploaded is not None:
        try:
            data = json.load(uploaded)
            cfg = nested_to_config(data)
            name = data.get("name", f"custom-{len(custom_presets)+1}")
            custom_presets[name] = cfg
            st.success(f"Preset '{name}' geladen und gespeichert.")
        except Exception as exc:
            st.error(f"Konnte Preset nicht laden: {exc}")

    st.markdown("### Diff zur aktuellen Konfiguration")
    diff = diff_configs(state.cfg, chosen_cfg)
    if diff:
        st.table(
            {
                "Parameter": [d[0] for d in diff],
                "Aktuell": [d[1] for d in diff],
                "Preset": [d[2] for d in diff],
            }
        )
    else:
        st.info("Keine Abweichungen – Config entspricht dem Preset.")

    st.markdown("### Copy as CLI")
    st.code(config_to_cli(state.cfg))


# ---------------------------------------------------------------------------
# Page: Aufnahme / Export
# ---------------------------------------------------------------------------

def build_video_from_frames(
    frames: list[tuple[int, bytes]],
    fps: int,
    fmt: str,
    *,
    encoder: str = "medium",
    crf: int = 23,
) -> tuple[str, bytes]:
    import io as _io

    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("imageio wird benötigt, um Videos zu schreiben.") from exc

    if not frames:
        raise RuntimeError("Keine Frames vorhanden – Aufnahme starten und Lauf durchführen.")

    imgs = []
    for _, png in frames:
        img = imageio.imread(_io.BytesIO(png))
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        imgs.append(img)

    fmt = fmt.lower()
    ext = fmt if fmt in {"mp4", "mkv", "avi"} else "mp4"
    output = _io.BytesIO()
    output_params: list[str] = []
    if encoder:
        output_params.extend(["-preset", str(encoder)])
    if crf is not None:
        output_params.extend(["-crf", str(int(crf))])
    if ext == "mp4":
        output_params.extend(["-movflags", "faststart"])

    writer = imageio.get_writer(
        output,
        format="ffmpeg",
        fps=int(fps),
        codec="libx264",
        quality=None,
        macro_block_size=None,
        pixelformat="yuv420p",
        output_params=output_params,
    )
    with writer:
        for img in imgs:
            writer.append_data(img)

    return ext, output.getvalue()


def resolve_video_path(raw_filename: str, fmt: str) -> Path:
    filename = raw_filename or "run.mp4"
    path = Path(filename)
    if path.suffix.lower() != f".{fmt.lower()}":
        path = path.with_suffix(f".{fmt.lower()}")
    if not path.is_absolute() and path.parent == Path("."):
        path = Path("runs") / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_video_to_disk(state: AppState) -> Path:
    ensure_video_defaults(state)
    fmt = state.video_params.get("format", "mp4")
    fps = int(state.video_params.get("fps", 30) or 30)
    encoder = state.video_params.get("encoder", "medium")
    crf = int(state.video_params.get("crf", 23) or 23)
    fmt, payload = build_video_from_frames(
        state.video_frames,
        fps=fps,
        fmt=fmt,
        encoder=encoder,
        crf=crf,
    )
    path = resolve_video_path(state.video_params.get("filename", ""), fmt)
    path.write_bytes(payload)
    return path


def page_recording(state: AppState) -> None:
    st.markdown("## 🎥 Aufnahme & Export")
    st.markdown("### Video-Einstellungen")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    ensure_video_defaults(state)
    with st.form("video_form"):
        filename = st.text_input("Dateiname", value=state.video_params.get("filename", "run.mp4"))
        format_choice = st.selectbox(
            "Format",
            options=["mp4", "mkv", "avi"],
            index=["mp4", "mkv", "avi"].index(state.video_params.get("format", "mp4")),
        )
        fps = st.number_input("FPS", value=int(state.video_params.get("fps", 30)), min_value=1, max_value=120, step=1)
        viz_freq = st.number_input(
            "Viz-Frequenz (Frames)",
            value=int(state.video_params.get("viz_freq", state.viz_every)),
            min_value=1,
            max_value=100,
        )
        overlay = st.checkbox("Overlay übernehmen", value=state.video_params.get("overlay", True))
        encoder_preset = st.selectbox(
            "Encoder-Preset",
            options=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
            index=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"].index(
                state.video_params.get("encoder", "medium")
            ),
        )
        crf = st.slider("CRF (0-51)", min_value=0, max_value=51, value=int(state.video_params.get("crf", 23)))
        submitted = st.form_submit_button("Speichern")
        if submitted:
            ensure_video_defaults(state)
            state.video_params = {
                "filename": filename,
                "format": format_choice,
                "fps": int(fps),
                "viz_freq": int(viz_freq),
                "overlay": bool(overlay),
                "encoder": encoder_preset,
                "crf": int(crf),
            }
            state.video_progress_total = compute_expected_video_frames(state)
            st.success("Video-Parameter aktualisiert.")

    col_start, col_stop = st.columns(2)
    if col_start.button("Aufnahme starten"):
        ensure_video_defaults(state)
        state.video_active = True
        state.video_frames = []
        state.video_limit_notified = False
        state.video_last_path = None
        state.video_progress_total = compute_expected_video_frames(state)
        state.video_summary = None
        append_log(state, "🎬 Aufnahme gestartet")
    if col_stop.button("Aufnahme stoppen"):
        ensure_video_defaults(state)
        frames_captured = len(state.video_frames)
        fps = int(state.video_params.get("fps", 30) or 30)
        video_freq = int(state.video_params.get("viz_freq", state.viz_every) or state.viz_every)
        video_freq = max(1, video_freq)
        state.video_active = False
        append_log(state, "🛑 Aufnahme gestoppt")
        if frames_captured:
            try:
                saved_path = save_video_to_disk(state)
                state.video_last_path = str(saved_path)
                append_log(state, f"💾 Video gespeichert: {saved_path}")
                st.success(f"Video gespeichert: {saved_path}")
                duration = frames_captured / max(1, fps)
                size_bytes = saved_path.stat().st_size
                state.video_summary = {
                    "status": "success",
                    "path": str(saved_path),
                    "frames": frames_captured,
                    "fps": fps,
                    "duration": duration,
                    "size_bytes": size_bytes,
                    "viz_freq": video_freq,
                }
                state.video_frames = []
            except Exception as exc:
                st.error(f"Video konnte nicht gespeichert werden: {exc}")
                append_log(state, f"❌ Video speichern fehlgeschlagen: {exc}")
                state.video_summary = {
                    "status": "error",
                    "message": str(exc),
                    "frames": frames_captured,
                }
        else:
            warning_msg = "Keine Frames aufgezeichnet – kein Video gespeichert."
            st.warning(warning_msg)
            state.video_summary = {
                "status": "warning",
                "message": warning_msg,
            }
        state.video_progress_total = 0

    if state.video_active:
        ensure_video_defaults(state)
        total_frames = state.video_progress_total or compute_expected_video_frames(state)
        captured_frames = len(state.video_frames)
        fps = int(state.video_params.get("fps", 30) or 30)
        progress_value = min(1.0, captured_frames / max(1, total_frames))
        st.progress(progress_value)
        st.caption(
            f"Aufgezeichnete Frames: {captured_frames}/{total_frames} • ca. {captured_frames / max(1, fps):.1f} s bei {fps} FPS"
        )

    if state.video_summary:
        summary = state.video_summary
        status = summary.get("status", "info")
        if status == "success":
            details = [
                f"Datei: {summary['path']}",
                f"Frames: {summary['frames']} @ {summary['fps']} FPS (Viz-Frequenz {summary['viz_freq']})",
                f"Dauer: {summary['duration']:.2f} s",
                f"Größe: {format_bytes(summary['size_bytes'])}",
            ]
            st.success("**Kurzbericht Video**\n" + "\n".join(f"- {line}" for line in details))
        elif status == "warning":
            st.warning(summary.get("message", "Keine Informationen verfügbar."))
        elif status == "error":
            st.error(summary.get("message", "Unbekannter Fehler beim Speichern."))

    if state.video_frames:
        st.caption(f"Gespeicherte Frames: {len(state.video_frames)} / {MAX_VIDEO_FRAMES}")
        if len(state.video_frames) >= MAX_VIDEO_FRAMES:
            st.warning("Frame-Limit erreicht – ältere Frames wurden überschrieben.")
        if st.button("Video exportieren"):
            try:
                fmt, payload = build_video_from_frames(
                    state.video_frames,
                    fps=int(state.video_params.get("fps", 30)),
                    fmt=state.video_params.get("format", "mp4"),
                    encoder=state.video_params.get("encoder", "medium"),
                    crf=int(state.video_params.get("crf", 23)),
                )
                st.download_button(
                    "Download Video",
                    data=payload,
                    file_name=state.video_params.get(
                        "filename",
                        f"{state.cfg.objective}_seed{state.cfg.seed}_{stamp}.{fmt}",
                    ),
                    mime=f"video/{fmt}",
                )
            except Exception as exc:
                st.error(f"Video konnte nicht erzeugt werden: {exc}")
    elif state.video_last_path:
        st.success(f"Letztes gespeichertes Video: {state.video_last_path}")

    st.markdown("### Artefakte")
    cfg_json = json.dumps(config_to_nested(state.cfg), indent=2)
    st.download_button(
        "Config exportieren (JSON)",
        data=cfg_json,
        file_name=f"{state.cfg.objective}_seed{state.cfg.seed}_{stamp}_config.json",
        mime="application/json",
    )

    if state.controller and state.controller.best_history:
        history = state.controller.best_history
        lines = ["iter,gbest_val,gbest_pos_x,gbest_pos_y"]
        for it, val, pos in history:
            lines.append(f"{it},{val},{pos[0]},{pos[1]}")
        csv_payload = "\n".join(lines)
        st.download_button(
            "Best-Trajectory (CSV)",
            data=csv_payload,
            file_name=f"{state.cfg.objective}_seed{state.cfg.seed}_{stamp}_trajectory.csv",
            mime="text/csv",
        )

    if state.video_frames:
        if st.button("Heatmap-Snapshots exportieren (ZIP)"):
            import zipfile

            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for idx, (it, frame) in enumerate(state.video_frames):
                    zf.writestr(f"frame_{idx:04d}_iter{it:04d}.png", frame)
            st.download_button(
                "Download Snapshots",
                data=zip_buf.getvalue(),
                file_name=f"{state.cfg.objective}_seed{state.cfg.seed}_{stamp}_snapshots.zip",
                mime="application/zip",
            )

    if state.logs:
        log_payload = "\n".join(state.logs)
        st.download_button(
            "Log exportieren (TXT)",
            data=log_payload,
            file_name=f"{state.cfg.objective}_seed{state.cfg.seed}_{stamp}_log.txt",
            mime="text/plain",
        )

    st.markdown("### Hinweise")
    st.info(
        "FFmpeg (Matplotlib) oder OpenCV erhöhen die Kompatibilität. Für verlustarme Videos CRF senken, für kleinere Dateien erhöhen."
    )


# ---------------------------------------------------------------------------
# Page: Experiments
# ---------------------------------------------------------------------------

def parse_numeric_list(text: str) -> list[float]:
    if not text.strip():
        return []
    items: list[float] = []
    for chunk in text.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            items.append(float(chunk))
        except ValueError:
            raise ValueError(f"Ungültiger Wert: {chunk}")
    return items


def apply_config_value(cfg: HPIOConfig, key: str, value: float) -> None:
    if key.startswith("field."):
        _, attr = key.split(".", 1)
        target = cfg.field
    elif key.startswith("agent."):
        _, attr = key.split(".", 1)
        target = cfg.agent
    else:
        attr = key
        target = cfg

    if not hasattr(target, attr):
        raise AttributeError(f"Unbekannter Parameter: {key}")

    current = getattr(target, attr)
    if isinstance(current, bool):
        coerced: Any = bool(value)
    elif isinstance(current, int):
        coerced = int(value)
    else:
        coerced = float(value)
    setattr(target, attr, coerced)


def run_experiment(cfg: HPIOConfig, seeds: list[int]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for seed in seeds:
        exp_cfg = nested_to_config(config_to_nested(cfg))
        exp_cfg.seed = int(seed)
        try:
            opt = HPIO(exp_cfg)
            pos, val = opt.run()
            results.append(
                {
                    "seed": int(seed),
                    "best_val": float(val),
                    "best_pos_x": float(pos[0]),
                    "best_pos_y": float(pos[1]),
                    "iters": opt.cfg.iters,
                    "config": config_to_nested(exp_cfg),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "seed": int(seed),
                    "error": str(exc),
                    "config": config_to_nested(exp_cfg),
                }
            )
    return results


def run_parameter_grid(cfg: HPIOConfig, grid: dict[str, list[float]]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    combos = list(product(*grid.values()))
    results: list[dict[str, Any]] = []
    for values in combos:
        exp_cfg = nested_to_config(config_to_nested(cfg))
        combo = {k: v for k, v in zip(keys, values)}
        try:
            for key, value in combo.items():
                apply_config_value(exp_cfg, key, value)
            opt = HPIO(exp_cfg)
            pos, val = opt.run()
            results.append(
                {
                    "param_combo": dict(combo),
                    "best_val": float(val),
                    "best_pos_x": float(pos[0]),
                    "best_pos_y": float(pos[1]),
                    "config": config_to_nested(exp_cfg),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "param_combo": dict(combo),
                    "error": str(exc),
                    "config": config_to_nested(exp_cfg),
                }
            )
    return results


def run_parameter_table(cfg: HPIOConfig, rows: list[dict[str, float]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in rows:
        exp_cfg = nested_to_config(config_to_nested(cfg))
        combo = dict(row)
        try:
            for key, value in combo.items():
                apply_config_value(exp_cfg, key, value)
            opt = HPIO(exp_cfg)
            pos, val = opt.run()
            results.append(
                {
                    "param_combo": combo,
                    "best_val": float(val),
                    "best_pos_x": float(pos[0]),
                    "best_pos_y": float(pos[1]),
                    "config": config_to_nested(exp_cfg),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "param_combo": combo,
                    "error": str(exc),
                    "config": config_to_nested(exp_cfg),
                }
            )
    return results


def page_experiments(state: AppState) -> None:
    st.markdown("## 🧪 Experimente (Batch & Benchmark)")
    cfg = state.cfg

    tab_seeds, tab_presets, tab_grid = st.tabs(["Seeds-Sweep", "Preset-Vergleich", "Parameter-Raster"])

    with tab_seeds:
        seed_text = st.text_input("Seeds (Komma, Bereich a-b oder Anzahl n)", value="0, 1, 2")
        if st.button("Seeds-Lauf starten", key="run_seeds"):
            try:
                seeds = []
                if "-" in seed_text and ":" not in seed_text:
                    lo, hi = seed_text.split("-", 1)
                    seeds = list(range(int(lo), int(hi) + 1))
                elif seed_text.strip().isdigit():
                    count = int(seed_text.strip())
                    seeds = list(range(count))
                else:
                    seeds = [int(float(s.strip())) for s in seed_text.split(",") if s.strip()]
                with st.spinner("Runs werden ausgeführt..."):
                    results = run_experiment(cfg, seeds)
                state.experiment_results["seeds"] = results
                st.success("Seeds-Sweep abgeschlossen")
            except Exception as exc:
                st.error(f"Fehler beim Seeds-Sweep: {exc}")
        if "seeds" in state.experiment_results:
            st.dataframe(state.experiment_results["seeds"])

    with tab_presets:
        presets = load_presets()
        selected = st.multiselect("Presets wählen", options=list(presets.keys()), default=list(presets.keys())[:2])
        runs_per_preset = st.number_input("Runs pro Preset", value=1, min_value=1, max_value=10, step=1)
        if st.button("Preset-Benchmark starten"):
            results: list[dict[str, Any]] = []
            rng = np.random.default_rng(cfg.seed)
            with st.spinner("Presets werden verglichen..."):
                for name in selected:
                    preset_cfg = nested_to_config(config_to_nested(presets[name]))
                    for _ in range(int(runs_per_preset)):
                        seed = int(rng.integers(0, 2**32 - 1))
                        preset_cfg.seed = seed
                        try:
                            opt = HPIO(preset_cfg)
                            pos, val = opt.run()
                            results.append(
                                {
                                    "preset": name,
                                    "seed": seed,
                                    "best_val": float(val),
                                    "best_pos_x": float(pos[0]),
                                    "best_pos_y": float(pos[1]),
                                    "config": config_to_nested(preset_cfg),
                                }
                            )
                        except Exception as exc:
                            results.append(
                                {
                                    "preset": name,
                                    "seed": seed,
                                    "error": str(exc),
                                    "config": config_to_nested(preset_cfg),
                                }
                            )
            state.experiment_results["presets"] = results
            st.success("Preset-Vergleich abgeschlossen")
        if "presets" in state.experiment_results:
            st.dataframe(state.experiment_results["presets"])

    with tab_grid:
        param_key = st.text_input("Parameter-Key (z.B. field.relax_alpha)", value="field.relax_alpha")
        values_text = st.text_input("Werte (Komma-Liste)", value="0.24,0.26,0.28")
        w_phase_text = st.text_input("Weitere Parameter (key=value;...)", value="w_phase=0.4;w_intensity=1.0")
        uploaded_csv = st.file_uploader("Parameter-CSV (optional)", type="csv", key="param_csv")
        if st.button("Parameter-Raster ausführen"):
            try:
                if uploaded_csv is not None:
                    csv_text = uploaded_csv.getvalue().decode("utf-8")
                    reader = csv.DictReader(io.StringIO(csv_text))
                    rows: list[dict[str, float]] = []
                    for row in reader:
                        parsed_row: dict[str, float] = {}
                        for key, value in row.items():
                            if value is None or not value.strip():
                                continue
                            try:
                                parsed_row[key.strip()] = float(value)
                            except ValueError as exc:
                                raise ValueError(
                                    f"Ungültiger Wert '{value}' in Spalte '{key}'"
                                ) from exc
                        if parsed_row:
                            rows.append(parsed_row)
                    if not rows:
                        raise ValueError("CSV enthält keine gültigen Parameterzeilen.")
                    with st.spinner("Parameter-Tabelle läuft..."):
                        results = run_parameter_table(cfg, rows)
                else:
                    grid_values = parse_numeric_list(values_text)
                    extra_grid: dict[str, list[float]] = {param_key: grid_values}
                    if w_phase_text.strip():
                        for part in w_phase_text.split(";"):
                            part = part.strip()
                            if not part:
                                continue
                            key, value_list = part.split("=")
                            extra_grid[key.strip()] = parse_numeric_list(value_list)
                    with st.spinner("Parameter-Raster läuft..."):
                        results = run_parameter_grid(cfg, extra_grid)
                state.experiment_results["grid"] = results
                st.success("Parameter-Raster abgeschlossen")
            except Exception as exc:
                st.error(f"Raster konnte nicht durchgeführt werden: {exc}")
        if "grid" in state.experiment_results:
            st.dataframe(state.experiment_results["grid"])

    if state.experiment_results:
        st.markdown("### Export & Visualisierung")
        combined = json.dumps(state.experiment_results, indent=2)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        st.download_button(
            "Resultate als JSON",
            data=combined,
            file_name=f"{state.cfg.objective}_experiments_{stamp}.json",
            mime="application/json",
        )


# ---------------------------------------------------------------------------
# Page: Help / Documentation
# ---------------------------------------------------------------------------

def page_help(state: AppState) -> None:
    st.markdown("## 📚 Hilfe & Dokumentation")
    st.markdown(
        r"""
        ### Was ist HPIO?
        HPIO (Hybrid Pheromone Inspired Optimizer) kombiniert ein Feld \(Φ\), mehrere Agenten sowie
        Ablage- und Relaxationsmechanismen, um Zielfunktionen wie Rastrigin, Ackley oder Himmelblau zu minimieren.
        Agenten erkunden den Raum, deponieren Intensität/Phasen-Spuren im Feld und folgen Gradienten sowie globalen Bestwerten.

        ### Parameter-Glossar
        - **relax_alpha**: Stärke der Feldglättung – kleinere Werte = schärfere Peaks.
        - **evap**: Verdunstung, reduziert alte Spuren.
        - **kernel_sigma**: Breite des Gaußfilters in der Relaxation.
        - **step / curiosity / momentum**: Bewegungsparameter der Agenten.
        - **deposit_sigma**: Fußabdruck der Ablage im Grid.
        - **coherence_gain**: Drift in Richtung global best.
        - **w_intensity / w_phase / phase_span_pi**: Gewichtung und Phasen-Spannweite der Ablage.
        - **anneal_*:** lineare Interpolation über die Laufzeit hinweg.
        - **early_patience / early_tol**: Frühabbruch, wenn der Bestwert kaum besser wird.

        ### Troubleshooting
        - **Hängt bei ~1.0 (Rastrigin)**: `w_phase ↑`, `relax_alpha ↓`, `deposit_sigma ↓`, `momentum ↓`
        - **Ackley bleibt ~0.7**: `relax_alpha ↓`, `w_phase ↑`, `curiosity_to ↓`, `momentum ↓`
        - **Zappeln**: `momentum ↑` leicht oder `coherence_gain ↑` leicht

        ### Performance-Tipps
        - Gridgröße ≤ 192², Viz-Frequenz hochsetzen, Video getrennt aufnehmen.
        - GPU optional – wenn PyOpenCL verfügbar ist, lässt sich das Toggle im Start-Tab aktivieren.

        ### Kompatibilität & Tools
        - Videoexport nutzt `imageio`. Für FFmpeg/OpenCV siehe `hpio_record.py`.
        - CLI-Einzeiler über "Copy as CLI" auf der Preset-Seite generieren.
        """
    )


# ---------------------------------------------------------------------------
# Main application entry
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="HPIO Control Center", layout="wide")
    state = get_state()

    pages = {
        "Start / Run": page_run,
        "Parameter": page_parameters,
        "Presets": page_presets,
        "Aufnahme & Export": page_recording,
        "Experimente": page_experiments,
        "Hilfe": page_help,
    }
    page_name = st.sidebar.radio("Seite wählen", list(pages.keys()))

    pages[page_name](state)


if __name__ == "__main__":
    main()


