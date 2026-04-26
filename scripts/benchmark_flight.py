"""Benchmark GateNet + QuAdGate on a single flight directory.

    python scripts/benchmark_flight.py checkpoints/best_gatenet.pth \\
        data/piloted/flight-01p-ellipse
    python scripts/benchmark_flight.py checkpoints/best_gatenet.pth <flight> \\
        --stride 10 --out results/bench_flight01.png
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from gate_detection.config import load_config
from gate_detection.evaluation.benchmark import benchmark_flight
from gate_detection.inference.predictor import GatePredictor
from gate_detection.visualization.plotting import plot_flight_grid


_DIST_BINS = [0, 1, 2, 3, 5, 8, float("inf")]
_DIST_LBLS = ["<1 m", "1–2 m", "2–3 m", "3–5 m", "5–8 m", ">8 m"]


def _print_summary(b) -> None:
    det = b.detected_mask
    cerr = np.array([c for c in b.corner_errors if c is not None], dtype=float)

    print(f"\n{'─' * 52}")
    print(f"  Flight             : {b.flight_name}")
    print(f"  Frames evaluated   : {b.n_total}")
    print(f"  Throughput         : {b.fps:.1f} fps  (wall-clock, incl. I/O)")
    print()
    print(f"  ── QuAdGate detection ───────────────────────────")
    print(f"  Detection rate     : {det.sum():>4d}/{b.n_total}  ({100 * det.mean():.1f}%)")
    if len(cerr) > 0:
        print(f"  Corner error (px)  : mean={cerr.mean():.2f}  "
              f"median={np.median(cerr):.2f}  std={cerr.std():.2f}")
        for thr in (5, 10):
            n = int((cerr <= thr).sum())
            print(f"  Corner err ≤ {thr:>2d} px : {n:>4d}/{len(cerr)}  "
                  f"({100 * n / len(cerr):.1f}%)")
    else:
        print(f"  Corner error       : no successful detections")
    print(f"{'─' * 52}\n")


def _print_distance_table(b) -> None:
    dist = b.distances
    valid = ~np.isnan(dist)
    if valid.sum() == 0:
        return
    print(f"  Distance-sliced metrics  (drone → nearest gate)")
    print(f"  {'Dist':>8}  {'N':>5}  {'Det rate':>10}  {'Mean cerr (px)':>14}")
    print(f"  {'─' * 8}  {'─' * 5}  {'─' * 10}  {'─' * 14}")
    det = b.detected_mask
    for lo, hi, lbl in zip(_DIST_BINS[:-1], _DIST_BINS[1:], _DIST_LBLS):
        mask = valid & (dist >= lo) & (dist < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        m_det = det[mask].mean()
        ce = [b.corner_errors[i] for i in range(len(b.corner_errors))
              if mask[i] and b.corner_errors[i] is not None]
        m_ce = f"{np.mean(ce):.2f}" if ce else "—"
        print(f"  {lbl:>8}  {n:>5}  {m_det:>10.1%}  {m_ce:>14}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="GateNet + QuAdGate flight benchmark")
    p.add_argument("checkpoint", help="path to .pth checkpoint")
    p.add_argument("flight",     help="flight folder (e.g. data/piloted/flight-01p-ellipse)")
    p.add_argument("--config",   type=str, default=None)
    p.add_argument("--stride",   type=int, default=30)
    p.add_argument("--n",        type=int, default=16, help="rows in visual grid")
    p.add_argument("--out",      type=str, default="flight_benchmark.png")
    args = p.parse_args()

    cfg = load_config(args.config)
    predictor = GatePredictor(
        args.checkpoint,
        img_size=cfg["data"]["img_size"],
        model_name=cfg["model"].get("name", "gatenet"),
        width_factor=cfg["model"]["width_factor"],
        encoder_name=cfg["model"].get("encoder_name", "resnet34"),
        encoder_weights=cfg["model"].get("encoder_weights", None),
        dropout_p=cfg["model"].get("dropout_p", 0.0),
    )
    print(f"Checkpoint : {os.path.basename(args.checkpoint)}  "
          f"(epoch {predictor.checkpoint_meta['epoch']}, "
          f"val_loss={predictor.checkpoint_meta['val_loss']:.4f})")
    print(f"Flight     : {os.path.basename(os.path.normpath(args.flight))}")

    bench = benchmark_flight(predictor, args.flight,
                             stride=args.stride, n_visuals=args.n)

    print(f"Frames     : {bench.n_total} sampled (stride={args.stride})")
    _print_summary(bench)
    _print_distance_table(bench)

    plot_flight_grid(
        bench.visuals,
        flight_name=bench.flight_name, fps=bench.fps,
        n_total=bench.n_total,
        n_detected=int(bench.detected_mask.sum()),
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
