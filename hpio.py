# hpio_bench.py
# Benchmark-Suite für HPIO (CPU/GPU, Mehrfach-Seeds, CSV + Plots)
# Python 3.12
from __future__ import annotations

import argparse, csv, time, importlib, sys, statistics as stats
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ObjectiveName = Literal["rastrigin", "ackley", "himmelblau"]

# Wir erwarten hpio.py im gleichen Ordner
try:
    hpio = importlib.import_module("hpio")
except Exception as e:
    print("Fehler: hpio.py konnte nicht importiert werden. Lege hpio_bench.py neben hpio.py.", file=sys.stderr)
    raise

@dataclass
class BenchResult:
    objective: str
    seed: int
    use_gpu: bool
    visualize: bool
    best_val: float
    best_pos_x: float
    best_pos_y: float
    iters_used: int
    elapsed_s: float

def run_one(objective: ObjectiveName, seed: int, use_gpu: bool, visualize: bool=False) -> BenchResult:
    # Konfiguration aufbauen (wie in hpio.build_config), aber Seed überschreiben
    cfg = hpio.build_config(objective, use_gpu=use_gpu, visualize=visualize)
    cfg.seed = seed
    # Für Benchmark: Visualisierung i. d. R. aus
    cfg.visualize = visualize

    # Wir zählen effektiv genutzte Iterationen über Early-Stopping-Heuristik:
    # Dazu instrumentieren wir minimal HPIO.run() via Subklasse (kein Edit von hpio.py nötig)
    class HPIOCount(hpio.HPIO):
        def run(self):
            start = time.perf_counter()
            it_before = getattr(self, "_iters_done", 0)
            # Mini-Hook: wir zählen Iterationsfortschritt, indem wir die Originalschleife laufen lassen
            # und die globale Variable anhand des Early-Stopping-Logs approximieren.
            # Sauberer Weg: wir messen Zeit & fügen eine Zählung in der Schleife hinzu.
            # Dafür nutzen wir Monkeypatch: wir ersetzen self._move_agent temporär, um iter++ an einem safe point zu setzen.
            self._iters_done = 0
            orig_move = self._move_agent
            def patched_move_agent(a, step_scale, curiosity_scale):
                orig_move(a, step_scale, curiosity_scale)
                # nur einmal pro Agent zu zählen wäre falsch -> wir zählen später aus Log; stattdessen hooken wir relax():
            orig_relax = self.field.relax
            def patched_relax():
                self._iters_done += 1
                orig_relax()
            self.field.relax = patched_relax  # type: ignore[attr-defined]

            best_pos, best_val = super().run()

            # Restore
            self.field.relax = orig_relax  # type: ignore[attr-defined]

            elapsed = time.perf_counter() - start
            return best_pos, best_val, self._iters_done, elapsed

    opt = HPIOCount(cfg)
    best_pos, best_val, iters_used, elapsed_s = opt.run()

    return BenchResult(
        objective=objective,
        seed=seed,
        use_gpu=use_gpu,
        visualize=visualize,
        best_val=float(best_val),
        best_pos_x=float(best_pos[0]),
        best_pos_y=float(best_pos[1]),
        iters_used=int(iters_used),
        elapsed_s=float(elapsed_s),
    )

def write_csv(path: Path, rows: list[BenchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["objective","seed","use_gpu","visualize","best_val","best_pos_x","best_pos_y","iters_used","elapsed_s"])
        for r in rows:
            w.writerow([r.objective, r.seed, int(r.use_gpu), int(r.visualize),
                        f"{r.best_val:.12g}", f"{r.best_pos_x:.9g}", f"{r.best_pos_y:.9g}",
                        r.iters_used, f"{r.elapsed_s:.6f}"])

def try_plot(objective: str, rows_cpu: list[BenchResult], rows_gpu: list[BenchResult], outdir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[bench] matplotlib nicht verfügbar – überspringe Plot.")
        return
    # einfache Boxplots best_val & elapsed
    plt.figure(figsize=(7.2, 4.4))
    data_vals = []
    labels = []
    if rows_cpu:
        data_vals.append([r.best_val for r in rows_cpu]); labels.append("CPU best_val")
    if rows_gpu:
        data_vals.append([r.best_val for r in rows_gpu]); labels.append("GPU best_val")
    if not data_vals:
        return
    plt.boxplot(data_vals, labels=labels, showmeans=True)
    plt.title(f"HPIO {objective} – Bestwerte (n={len(rows_cpu) or len(rows_gpu)})")
    plt.ylabel("best_val (kleiner ist besser)")
    plt.tight_layout()
    plt.savefig(outdir / f"hpio_{objective}_bestvals.png", dpi=140)

    plt.figure(figsize=(7.2, 4.4))
    data_t = []
    labels_t = []
    if rows_cpu:
        data_t.append([r.elapsed_s for r in rows_cpu]); labels_t.append("CPU Zeit [s]")
    if rows_gpu:
        data_t.append([r.elapsed_s for r in rows_gpu]); labels_t.append("GPU Zeit [s]")
    if data_t:
        plt.boxplot(data_t, labels=labels_t, showmeans=True)
        plt.title(f"HPIO {objective} – Laufzeit (n={len(rows_cpu) or len(rows_gpu)})")
        plt.ylabel("Sekunden")
        plt.tight_layout()
        plt.savefig(outdir / f"hpio_{objective}_times.png", dpi=140)

def summary_print(rows_cpu: list[BenchResult], rows_gpu: list[BenchResult], objective: str) -> None:
    def stats_line(rows, tag):
        if not rows:
            return f"{tag}: —"
        vals = [r.best_val for r in rows]
        times = [r.elapsed_s for r in rows]
        iters = [r.iters_used for r in rows]
        return (f"{tag}: best_val median={stats.median(vals):.3g}, mean={stats.fmean(vals):.3g} | "
                f"time median={stats.median(times):.3g}s | iters median={stats.median(iters)}")
    print(f"\n[{objective}] Benchmark-Zusammenfassung")
    print(stats_line(rows_cpu, "CPU"))
    print(stats_line(rows_gpu, "GPU"))

def main():
    ap = argparse.ArgumentParser(description="HPIO Benchmark-Suite")
    ap.add_argument("objective", choices=["rastrigin","ackley","himmelblau"], help="Zielfunktion")
    ap.add_argument("--seeds", type=int, default=10, help="Anzahl Seeds (Runs)")
    ap.add_argument("--start-seed", type=int, default=100, help="Startwert Seed")
    ap.add_argument("--cpu", action="store_true", help="Nur CPU laufen lassen")
    ap.add_argument("--gpu", action="store_true", help="Auch GPU laufen lassen")
    ap.add_argument("--viz", action="store_true", help="Visualisierung während eines Laufs (nur Seed=start-seed)")
    ap.add_argument("--out", type=str, default="bench_out", help="Ausgabeordner")
    args = ap.parse_args()

    objective: ObjectiveName = args.objective  # type: ignore[assignment]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    run_cpu = args.cpu or (not args.gpu)  # standard: CPU, außer man wählt nur GPU
    run_gpu = args.gpu

    rows_cpu: list[BenchResult] = []
    rows_gpu: list[BenchResult] = []

    for i in range(args.seeds):
        seed = args.start_seed + i
        do_viz = args.viz and (i == 0)

        if run_cpu:
            r = run_one(objective, seed=seed, use_gpu=False, visualize=do_viz)
            rows_cpu.append(r)
            print(f"[bench] CPU seed={seed} done: best={r.best_val:.6g} in {r.elapsed_s:.3f}s (iters={r.iters_used})")

        if run_gpu:
            r = run_one(objective, seed=seed, use_gpu=True, visualize=do_viz)
            rows_gpu.append(r)
            print(f"[bench] GPU seed={seed} done: best={r.best_val:.6g} in {r.elapsed_s:.3f}s (iters={r.iters_used})")

    # CSV pro Objective/Plattform
    if rows_cpu:
        write_csv(outdir / f"hpio_{objective}_cpu.csv", rows_cpu)
    if rows_gpu:
        write_csv(outdir / f"hpio_{objective}_gpu.csv", rows_gpu)

    # kompakte Plot-Zusammenfassung
    try_plot(objective, rows_cpu, rows_gpu, outdir=outdir)

    # Konsolen-Kurzbericht
    summary_print(rows_cpu, rows_gpu, objective)

if __name__ == "__main__":
    main()
