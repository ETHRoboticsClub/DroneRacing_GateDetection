"""Evaluate GateNet segmentation IoU on the synthetic test split.

    python scripts/evaluate.py checkpoints/best_gatenet.pth
    python scripts/evaluate.py checkpoints/best_gatenet.pth --threshold 0.3 --n 20
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from gate_detection.config import load_config
from gate_detection.evaluation.iou import evaluate_iou
from gate_detection.inference.predictor import GatePredictor
from gate_detection.visualization.plotting import plot_iou_grid


def main() -> None:
    p = argparse.ArgumentParser(description="Segmentation IoU eval on the test set")
    p.add_argument("checkpoint", help="path to .pth checkpoint")
    p.add_argument("--config",    type=str, default=None)
    p.add_argument("--n",         type=int, default=16, help="rows in visual grid")
    p.add_argument("--out",       type=str, default="iou_results.png")
    p.add_argument("--threshold", type=float, default=None,
                   help="binarisation threshold (default: from config)")
    args = p.parse_args()

    cfg = load_config(args.config)
    threshold = args.threshold if args.threshold is not None else cfg["eval"]["threshold"]

    predictor = GatePredictor(
        args.checkpoint,
        img_size=cfg["data"]["img_size"],
        model_name=cfg["model"].get("name", "gatenet"),
        width_factor=cfg["model"]["width_factor"],
        encoder_name=cfg["model"].get("encoder_name", "resnet34"),
        encoder_weights=cfg["model"].get("encoder_weights", None),
        dropout_p=cfg["model"].get("dropout_p", 0.0),
    )
    print(f"Loaded checkpoint: {os.path.basename(args.checkpoint)}  "
          f"(epoch {predictor.checkpoint_meta['epoch']}, "
          f"val_loss={predictor.checkpoint_meta['val_loss']:.4f})")
    print(f"Evaluating  [threshold={threshold:.2f}, device={predictor.device}] ...")

    results = evaluate_iou(predictor, threshold=threshold, n_visuals=args.n)

    arr = results.ious
    samples = results.samples
    print(f"\n{'─' * 46}")
    print(f"  Checkpoint : {os.path.basename(args.checkpoint)}")
    print(f"  N images   : {len(arr)}")
    print(f"  Mean IoU   : {results.mean:.4f}")
    print(f"  Median IoU : {results.median:.4f}")
    print(f"  Std IoU    : {results.std:.4f}")
    print(f"  Min IoU    : {arr.min():.4f}  ← {os.path.basename(samples[int(arr.argmin())][0])}")
    print(f"  Max IoU    : {arr.max():.4f}  ← {os.path.basename(samples[int(arr.argmax())][0])}")
    for t in (0.50, 0.75, 0.90):
        n = int((arr >= t).sum())
        print(f"  IoU ≥ {t:.2f} : {n:>3d}/{len(arr)}  ({100 * n / len(arr):.1f}%)")
    print(f"{'─' * 46}\n")

    ckpt_meta = {**predictor.checkpoint_meta, "name": os.path.basename(args.checkpoint)}
    plot_iou_grid(results.visuals, ckpt_meta,
                  mean_iou=results.mean, median_iou=results.median,
                  threshold=threshold, out_path=args.out)


if __name__ == "__main__":
    main()
