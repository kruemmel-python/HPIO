# -*- coding: utf-8 -*-
"""
hpio_record.py – Videoaufzeichnung (MP4/MKV) der HPIO-Optimierung
robuste Version:
- lädt die lokale hpio.py (neben dieser Datei) explizit via importlib.util
- nutzt Komposition statt Vererbung (kein subclassing von hpio.HPIO)
- gleicht die HPIO.run-Schleife nach und rendert pro Iteration Frames

Aufruf:
  python hpio_record.py rastrigin --video runs/rastrigin.mp4 --fps 30 --size 1280x720
  python hpio_record.py ackley --gpu --video runs/ackley_gpu.mkv --fps 24 --size 1600x900
  python hpio_record.py himmelblau --video runs/himmelblau.mp4 --viz-freq 2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Literal

import numpy as np
import importlib.util
import sys

ObjectiveName = Literal["rastrigin", "ackley", "himmelblau"]

# ------------------------------------------------------------
# Lokale hpio.py sicher laden (aus demselben Ordner wie diese Datei)
# ------------------------------------------------------------
def load_local_hpio():
    here = Path(__file__).resolve().parent
    hpio_path = here / "hpio.py"
    if not hpio_path.exists():
        print(f"[record] Konnte {hpio_path} nicht finden.", file=sys.stderr)
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("hpio_local", str(hpio_path))
    if spec is None or spec.loader is None:
        print("[record] Konnte Modul-Spec für hpio.py nicht erstellen.", file=sys.stderr)
        sys.exit(1)
    mod = importlib.util.module_from_spec(spec)
    # 🔧 WICHTIG: Vor exec_module registrieren, damit dataclasses Zugriff hat
    sys.modules[spec.name] = mod  # <-- NEU
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


hpio = load_local_hpio()

# ------------------------------------------------------------
# kleine Hilfen
# ------------------------------------------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def lerp(a: float, b: float, t: float) -> float:
    return (1.0 - t) * a + t * b

def smoothstep(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)

# ------------------------------------------------------------
# Video-Writer (FFmpeg bevorzugt, OpenCV als Fallback)
# ------------------------------------------------------------
class VideoWriter:
    def __init__(self, fname: str, fps: int, width: int, height: int):
        self.fname = fname
        self.fps = fps
        self.width = width
        self.height = height
        self.backend = None
        self._writer = None
        self._fig = None
        try:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            self._plt = plt
            self._fig = plt.figure(figsize=(width/100, height/100), dpi=100)
            self._ffmpeg_writer = animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=2000)
            self._ffmpeg_writer.setup(self._fig, fname, dpi=100)
            self.backend = "ffmpeg"
            self._writer = self._ffmpeg_writer
            return
        except Exception:
            self.backend = None
            self._writer = None
            self._fig = None
            self._plt = None
        try:
            import cv2 as cv
            self._cv = cv
            ext = Path(fname).suffix.lower()
            if ext in [".mp4", ".m4v", ".mov"]:
                fourcc = cv.VideoWriter_fourcc(*"mp4v")
            elif ext in [".mkv", ".avi"]:
                fourcc = cv.VideoWriter_fourcc(*"XVID")
            else:
                fourcc = cv.VideoWriter_fourcc(*"mp4v")
            vw = cv.VideoWriter(fname, fourcc, fps, (width, height))
            if not vw.isOpened():
                raise RuntimeError("OpenCV VideoWriter konnte nicht geöffnet werden.")
            self.backend = "opencv"
            self._writer = vw
        except Exception as e:
            raise RuntimeError(
                "Kein Video-Backend verfügbar. Installiere FFmpeg (für Matplotlib) "
                "oder OpenCV (`pip install opencv-python`)."
            ) from e

    @property
    def fig(self):
        return self._fig if self.backend == "ffmpeg" else None

    def add_frame_from_figure(self):
        if self.backend != "ffmpeg":
            raise RuntimeError("Nur für FFmpeg-Backend verfügbar.")
        self._writer.grab_frame()

    def add_frame_from_rgb(self, rgb: np.ndarray):
        if self.backend != "opencv":
            raise RuntimeError("Nur für OpenCV-Backend verfügbar.")
        bgr = rgb[:, :, ::-1]
        self._writer.write(bgr)

    def close(self):
        if self.backend == "ffmpeg" and self._writer is not None:
            self._writer.finish()
        elif self.backend == "opencv" and self._writer is not None:
            self._writer.release()
        self._writer = None

# ------------------------------------------------------------
# Recorder-UI (Heatmap links, Konsole rechts)
# ------------------------------------------------------------
class HPIORecorder:
    def __init__(self, writer: VideoWriter, grid_size: tuple[int,int], text_lines: int = 28):
        self.writer = writer
        self.text_lines = text_lines
        self.logs: list[str] = []

        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = writer.fig or plt.figure(figsize=(writer.width/100, writer.height/100), dpi=100)
        self.fig.clf()
        self.gs = self.fig.add_gridspec(1, 2, width_ratios=[2.0, 1.0])

        self.axL = self.fig.add_subplot(self.gs[0, 0])
        self.axR = self.fig.add_subplot(self.gs[0, 1])
        self.axR.set_title("Konsole", fontsize=11)
        self.axR.axis("off")
        self.txt = self.axR.text(0.02, 0.98, "", va="top", ha="left",
                                 family="monospace", fontsize=9,
                                 transform=self.axR.transAxes,
                                 bbox=dict(facecolor="#0b1220", alpha=0.85, edgecolor="#94a3b8", boxstyle="round,pad=0.5"),
                                 color="#e2e8f0")

        gy, gx = grid_size[1], grid_size[0]
        dummy = np.zeros((gy, gx), dtype=np.float64)
        self.im = self.axL.imshow(dummy, origin="lower", interpolation="nearest")
        self.axL.set_title("|Φ| (log) + Agents", fontsize=11)
        self.scatter = None
        self.trail_lines = []
        self.trails = []
        self.max_trail = 80
        self.fig.tight_layout()

    def log(self, line: str):
        self.logs.append(line)
        if len(self.logs) > 5000:
            self.logs = self.logs[-5000:]

    def _update_right(self):
        block = "\n".join(self.logs[-self.text_lines:])
        self.txt.set_text(block)

    def _update_left(self, field_phi: np.ndarray, agents_px: np.ndarray, trail_subset: int = 30):
        I = np.abs(field_phi).astype(np.float64)
        I_disp = np.log(I + 1e-6)
        self.im.set_data(I_disp)

        if not self.trails:
            self.trails = [[] for _ in range(min(trail_subset, agents_px.shape[0]))]

        for i in range(len(self.trails)):
            xg, yg = int(agents_px[i, 0]), int(agents_px[i, 1])
            self.trails[i].append((xg, yg))
            if len(self.trails[i]) > self.max_trail:
                self.trails[i].pop(0)

        for ln in self.trail_lines:
            ln.remove()
        self.trail_lines = []
        for path in self.trails:
            if len(path) > 1:
                xs, ys = zip(*path)
                ln, = self.axL.plot(xs, ys, linewidth=1.0, alpha=0.7)
                self.trail_lines.append(ln)

        if self.scatter is None:
            self.scatter = self.axL.scatter(agents_px[:, 0], agents_px[:, 1], s=14, marker="o")
        else:
            self.scatter.set_offsets(agents_px)

    def frame(self, field_phi: np.ndarray, agents_px: np.ndarray, *, title: str | None = None):
        self._update_left(field_phi, agents_px)
        self._update_right()
        if title is not None:
            self.axL.set_title(title, fontsize=11)
        if self.writer.backend == "ffmpeg":
            self.fig.canvas.draw()
            self.writer.add_frame_from_figure()
        else:
            self.fig.canvas.draw()
            w, h = self.fig.canvas.get_width_height()
            rgb = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
            if (w, h) != (self.writer.width, self.writer.height):
                try:
                    import cv2 as cv
                    rgb = cv.resize(rgb, (self.writer.width, self.writer.height), interpolation=cv.INTER_AREA)
                except Exception:
                    pass
            self.writer.add_frame_from_rgb(rgb)

# ------------------------------------------------------------
# Kompositions-Runner: wir fahren die Schleife selbst
# ------------------------------------------------------------
class RecordingRunner:
    def __init__(self, cfg: "hpio.HPIOConfig", recorder: HPIORecorder, viz_every: int = 1):
        self.cfg = cfg
        self.recorder = recorder
        self.viz_every = max(1, int(viz_every))
        # Optimierer instanziieren
        self.opt = hpio.HPIO(cfg)  # nutzt deine Klasse direkt

    def _world_to_grid_np(self, pts: np.ndarray) -> np.ndarray:
        (xmin, xmax), (ymin, ymax) = self.opt.bounds
        gx, gy = self.opt.field.p.grid_size
        nx = (pts[:, 0] - xmin) / (xmax - xmin)
        ny = (pts[:, 1] - ymin) / (ymax - ymin)
        xg = np.clip((nx * (gx - 1)).astype(int), 0, gx - 1)
        yg = np.clip((ny * (gy - 1)).astype(int), 0, gy - 1)
        return np.stack([xg, yg], axis=1)

    def run(self) -> tuple[np.ndarray, float]:
        no_improve = 0
        last_best = float("inf")

        self.recorder.log(f"HPIO Recording – objective={self.cfg.objective}, gpu={self.cfg.use_gpu}, iters={self.cfg.iters}")
        self.recorder.log(f"grid={self.opt.field.p.grid_size}, relax_alpha={self.opt.field.p.relax_alpha}, evap={self.opt.field.p.evap}, sigma={self.opt.field.p.kernel_sigma}")
        self.recorder.log(f"agents={len(self.opt.agents)}, step={self.cfg.agent.step}, curiosity={self.cfg.agent.curiosity}, momentum={self.cfg.agent.momentum}")
        self.recorder.log("------------------------------------------------------------------")

        for it in range(1, self.cfg.iters + 1):
            t = (it - 1) / max(1, self.cfg.iters - 1)
            s = smoothstep(t)
            step_scale = lerp(self.cfg.anneal_step_from, self.cfg.anneal_step_to, s)
            curiosity_scale = lerp(self.cfg.anneal_curiosity_from, self.cfg.anneal_curiosity_to, s)

            vals = np.array([a.best_val for a in self.opt.agents])
            ranks = np.argsort(np.argsort(vals))
            amp_by_rank = 0.80 + 0.50 * (1.0 - ranks / max(1, len(self.opt.agents) - 1))
            if t > 0.85:
                amp_by_rank[:] = 1.0

            for a, amp_scale in zip(self.opt.agents, amp_by_rank):
                self.opt._deposit_from_agent(a, float(amp_scale))

            self.opt.field.relax()

            for a in self.opt.agents:
                self.opt._move_agent(a, step_scale=step_scale, curiosity_scale=curiosity_scale)

            if (it % self.cfg.report_every) == 0 or it == 1:
                msg = f"[HPIO] iter={it:4d}  best={self.opt.gbest_val: .6f}  at {self.opt.gbest_pos}"
                print(msg)
                self.recorder.log(msg)

            if (it % self.viz_every) == 0 or it == self.cfg.iters:
                pts = np.stack([a.pos for a in self.opt.agents], axis=0)
                agents_px = self._world_to_grid_np(pts)
                title = None
                if getattr(self.cfg, "overlay", False) or hasattr(self.cfg, "overlay"):
                    title = f"|Φ| (log) + Agents  •  iter={it}  •  best={self.opt.gbest_val:.6g}"
                self.recorder.frame(self.opt.field.phi, agents_px, title=title)

            if self.opt.gbest_val < last_best - self.cfg.early_tol:
                last_best = self.opt.gbest_val
                no_improve = 0
            else:
                no_improve += 1

            if no_improve > self.cfg.early_patience:
                stop_msg = f"[HPIO] Early-Stopping bei iter={it} (keine Verbesserung > {self.cfg.early_patience} Schritte)."
                print(stop_msg)
                self.recorder.log(stop_msg)
                break

        # Local Polish (über die Original-Staticmethoden aus hpio)
        p_pol, v_pol = self.opt.local_quadratic_polish(self.opt.f, self.opt.gbest_pos, h=self.cfg.polish_h)
        if v_pol < self.opt.gbest_val:
            self.opt.gbest_pos, self.opt.gbest_val = p_pol, v_pol
            self.recorder.log(f"[HPIO] local polish → best={self.opt.gbest_val:.9g} at {self.opt.gbest_pos}")

        return self.opt.gbest_pos, self.opt.gbest_val

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_size(s: str) -> tuple[int, int]:
    try:
        w, h = s.lower().replace("x", " ").split()
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError("Bitte als WIDTHxHEIGHT angeben, z.B. 1280x720")

def main():
    ap = argparse.ArgumentParser(description="HPIO Video-Recorder (robust)")
    ap.add_argument("objective", choices=["rastrigin","ackley","himmelblau"], help="Zielfunktion")
    ap.add_argument("--gpu", action="store_true", help="GPU-Relaxation (PyOpenCL)")
    ap.add_argument("--video", type=str, required=True, help="Ausgabe-Datei (mp4/mkv/avi)")
    ap.add_argument("--fps", type=int, default=30, help="Frames pro Sekunde")
    ap.add_argument("--size", type=parse_size, default=(1280, 720), help="Videogröße WIDTHxHEIGHT")
    ap.add_argument("--viz-freq", type=int, default=1, help="Jede n-te Iteration als Frame")
    ap.add_argument("--seed", type=int, default=123, help="Zufalls-Seed")
    ap.add_argument("--report-every", type=int, default=None, help="Alle n Iterationen loggen (überschreibt cfg.report_every)")
    ap.add_argument("--overlay", action="store_true", help="Zeige Iteration & Bestwert live im Plot")
    ap.add_argument("--ackley-tight", action="store_true", help="Aggressives Tuning für Ackley (GPU empfohlen)")
    args = ap.parse_args()

    out_path = Path(args.video)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = VideoWriter(fname=str(out_path), fps=args.fps, width=args.size[0], height=args.size[1])

    cfg = hpio.build_config(args.objective, use_gpu=args.gpu, visualize=False)
    cfg.seed = args.seed
    cfg.overlay = args.overlay  # kleiner Bequemlichkeits-Hack

    # Report-Intervall überschreiben?
    if args.report_every is not None:
        cfg.report_every = max(1, args.report_every)
    else:
        cfg.report_every = max(10, cfg.report_every)  # wie bisher

    # Aggressives Ackley-Tuning (deine früheren „guten“ Werte)
    if args.ackley_tight and args.objective == "ackley":
        cfg.field.relax_alpha = 0.30
        cfg.agent.deposit_sigma = 1.5
        cfg.anneal_curiosity_to = 0.14
        cfg.iters = max(cfg.iters, 400)

    recorder = HPIORecorder(writer=writer, grid_size=cfg.field.grid_size, text_lines=28)
    runner = RecordingRunner(cfg, recorder=recorder, viz_every=max(1, args.viz_freq))

    t0 = time.perf_counter()
    best_pos, best_val = runner.run()
    dt = time.perf_counter() - t0

    recorder.log("------------------------------------------------------------------")
    recorder.log(f"Ergebnis: best_val={best_val:.9g} at {best_pos} | Dauer={dt:.3f}s")

    # Abschlussframe
    pts = np.stack([a.pos for a in runner.opt.agents], axis=0)
    agents_px = runner._world_to_grid_np(pts)
    recorder.frame(runner.opt.field.phi, agents_px)

    writer.close()

    print("\n=== Aufnahme fertig ===========================")
    print(f"Video   : {out_path}")
    print(f"Best Val: {best_val:.12g}")
    print(f"Best Pos: {best_pos}")
    print("===============================================")

if __name__ == "__main__":
    main()
