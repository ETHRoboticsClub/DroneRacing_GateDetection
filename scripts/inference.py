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
import re
import sys

_SRC = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import cv2
import numpy as np
import torch

from gate_detection.config import load_config
from gate_detection.data.dataset import MASK_SUFFIX, _load_sam_mask
from gate_detection.data.labels import corners_to_mask, parse_pixel_corners
from gate_detection.inference.predictor import GatePredictor
from gate_detection.paths import DATA_ROOT, SAM_ROOT, SYN_ROOT
from gate_detection.visualization.plotting import (
    plot_inference_grid,
    plot_inference_grid_with_qag_steps,
)


def _read_camera_info(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse data/real-style camera_info.txt → (K, D)."""
    text = open(path).read()
    text = re.sub(r"np\.float64\(([^)]+)\)", r"\1", text)
    def grab(key):
        m = re.search(rf"^{key}\s*:\s*\[([^\]]+)\]", text, re.MULTILINE)
        if not m:
            return None
        return [float(v) for v in m.group(1).split(",")]
    K = np.array(grab("K"), dtype=np.float64).reshape(3, 3)
    D = np.array(grab("D"), dtype=np.float64).reshape(-1, 1)
    if D.size < 5:
        D = np.concatenate([D.flatten(),
                             np.zeros(5 - D.size, dtype=np.float64)]).reshape(-1, 1)
    return K, D


def _predict_with_undistort(predictor: GatePredictor, img_path: str,
                             K: np.ndarray | None, D: np.ndarray | None
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Like ``predictor.predict_image`` but optionally undistorts the image
    using a Brown-Conrady model before running the network. Pixels outside
    the original frame after undistortion are left black.
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(img_path)
    if K is not None and D is not None and float(np.linalg.norm(D)) > 0:
        h, w = img_bgr.shape[:2]
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0.0,
                                                  newImgSize=(w, h))
        img_bgr = cv2.undistort(img_bgr, K, D, None, new_K)

    img_t, img_rgb = predictor._preprocess_bgr(img_bgr)
    with torch.no_grad():
        pred = predictor._predict_tensor(img_t).squeeze().cpu().numpy()
    return img_rgb, pred


def _find_label(img_path: str) -> str | None:
    """Look up the .txt label for ``img_path``."""
    stem = os.path.splitext(os.path.basename(img_path))[0]

    # Same directory as the image (sam_masked layout)
    cand = os.path.join(os.path.dirname(img_path), stem + ".txt")
    if os.path.exists(cand):
        return cand

    # Raw flight layout: <flight>/label_<flight>/<stem>.txt
    flight = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
    cand = os.path.join(os.path.dirname(os.path.dirname(img_path)),
                        f"label_{flight}", stem + ".txt")
    if os.path.exists(cand):
        return cand

    # Fallback: sam_masked or synthetic splits
    for root in (SAM_ROOT, SYN_ROOT):
        for d in ("train", "val", "test"):
            cand = os.path.join(str(root), d, stem + ".txt")
            if os.path.exists(cand):
                return cand
    return None


def _load_gt_mask(img_path: str, lbl: str | None, size: int) -> np.ndarray:
    """Load GT mask: prefer _mask.png SAM ring, fall back to rasterising the label."""
    stem = os.path.splitext(os.path.basename(img_path))[0]
    mask_png = os.path.join(os.path.dirname(img_path), stem + MASK_SUFFIX)
    if os.path.exists(mask_png):
        return _load_sam_mask(mask_png, size)
    if lbl:
        return corners_to_mask(lbl, size)
    return np.zeros((size, size), dtype=np.uint8)


def _build_row(
    predictor: GatePredictor,
    img_path: str,
    *,
    K_undist: np.ndarray | None = None,
    D_undist: np.ndarray | None = None,
) -> dict:
    img_rgb, pred = _predict_with_undistort(predictor, img_path,
                                             K_undist, D_undist)
    lbl = _find_label(img_path)
    return {
        "img_rgb":   img_rgb,
        "gt_mask":   _load_gt_mask(img_path, lbl, predictor.img_size),
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
    p.add_argument("--undistort", type=str, default=None,
                   help="path to camera_info.txt; if set, undistort each image "
                        "(via Brown-Conrady cv2.undistort) before inference")
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

    K_und = D_und = None
    if args.undistort is not None:
        K_und, D_und = _read_camera_info(args.undistort)
        print(f"Undistorting with K=[{K_und[0,0]:.1f},{K_und[1,1]:.1f},"
              f"{K_und[0,2]:.1f},{K_und[1,2]:.1f}]  |D|={float(np.linalg.norm(D_und)):.3f}")

    rows = [_build_row(predictor, p_, K_undist=K_und, D_undist=D_und)
            for p_ in paths]
    out = args.out or ("inference_debug.png" if args.qag_debug else "inference.png")
    ckpt_meta = {**predictor.checkpoint_meta, "name": os.path.basename(args.checkpoint)}

    if args.qag_debug:
        plot_inference_grid_with_qag_steps(rows, ckpt_meta, out)
    else:
        plot_inference_grid(rows, ckpt_meta, out)


if __name__ == "__main__":
    main()
