"""Run inference on a few images and write a visual grid.

    python scripts/inference.py checkpoints/best_gatenet.pth
    python scripts/inference.py checkpoints/best_gatenet.pth --n 12 --out inference.png
    python scripts/inference.py checkpoints/best_gatenet.pth --qag-debug --n 4
    python scripts/inference.py checkpoints/best_gatenet.pth --images path/a.jpg path/b.jpg
"""

from __future__ import annotations

import argparse
import glob
import os
import random

import numpy as np

from gate_detection.config import load_config
from gate_detection.data.labels import corners_to_mask, parse_pixel_corners
from gate_detection.inference.predictor import GatePredictor
from gate_detection.paths import DATA_ROOT, SYN_ROOT
from gate_detection.visualization.plotting import (
    plot_inference_grid,
    plot_inference_grid_with_qag_steps,
)


def _find_label(img_path: str) -> str | None:
    """Look up the label for ``img_path`` in either the raw flight layout or synthetic splits."""
    stem = os.path.splitext(os.path.basename(img_path))[0]
    flight = os.path.basename(os.path.dirname(os.path.dirname(img_path)))

    cand = os.path.join(os.path.dirname(os.path.dirname(img_path)),
                        f"label_{flight}", stem + ".txt")
    if os.path.exists(cand):
        return cand

    for d in ("train", "val", "test", "train_aug", "val_aug"):
        cand = os.path.join(SYN_ROOT, d, stem + ".txt")
        if os.path.exists(cand):
            return cand
    return None


def _build_row(predictor: GatePredictor, img_path: str) -> dict:
    img_rgb, pred = predictor.predict_image(img_path)
    lbl = _find_label(img_path)
    return {
        "img_rgb":   img_rgb,
        "gt_mask":   corners_to_mask(lbl, predictor.img_size) if lbl else
                     np.zeros((predictor.img_size, predictor.img_size), np.uint8),
        "pred_prob": pred,
        "priors":    parse_pixel_corners(lbl, predictor.img_size) if lbl else [],
        "fname":     os.path.basename(img_path),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="GateNet inference visualisation")
    p.add_argument("checkpoint")
    p.add_argument("--config",    type=str, default=None)
    p.add_argument("--images",    type=str, nargs="+", default=None,
                   help="specific image paths; if omitted, sample randomly from data/")
    p.add_argument("--n",         type=int, default=8,
                   help="number of random samples (ignored if --images is set)")
    p.add_argument("--out",       type=str, default=None)
    p.add_argument("--qag-debug", action="store_true",
                   help="show all QuAdGate intermediate steps")
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

    if args.images:
        paths = args.images
    else:
        all_imgs = glob.glob(
            os.path.join(str(DATA_ROOT), "**", "camera_*", "*.jpg"),
            recursive=True,
        )
        paths = random.sample(all_imgs, min(args.n, len(all_imgs)))

    rows = [_build_row(predictor, p_) for p_ in paths]
    out = args.out or ("inference_debug.png" if args.qag_debug else "inference.png")
    ckpt_meta = {**predictor.checkpoint_meta, "name": os.path.basename(args.checkpoint)}

    if args.qag_debug:
        plot_inference_grid_with_qag_steps(rows, ckpt_meta, out)
    else:
        plot_inference_grid(rows, ckpt_meta, out)


if __name__ == "__main__":
    main()
