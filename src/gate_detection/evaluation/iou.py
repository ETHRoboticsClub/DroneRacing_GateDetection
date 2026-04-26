"""Segmentation IoU evaluation on the synthetic test split."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gate_detection.data.labels import corners_to_mask
from gate_detection.inference.predictor import GatePredictor
from gate_detection.paths import SYN_ROOT
from .metrics import compute_iou


@dataclass
class IoUSample:
    img_rgb: np.ndarray
    gt_mask: np.ndarray
    pred_prob: np.ndarray
    pred_bin: np.ndarray
    iou: float
    fname: str


@dataclass
class IoUResults:
    ious: np.ndarray
    samples: list[tuple[str, str]]                      # (img_path, lbl_path)
    visuals: list[IoUSample] = field(default_factory=list)

    @property
    def mean(self) -> float:   return float(self.ious.mean())
    @property
    def median(self) -> float: return float(np.median(self.ious))
    @property
    def std(self) -> float:    return float(self.ious.std())

    def fraction_above(self, t: float) -> float:
        return float((self.ious >= t).mean())


def _load_test_samples(test_dir: str) -> list[tuple[str, str]]:
    out = []
    for f in sorted(os.listdir(test_dir)):
        if not f.endswith(".jpg"):
            continue
        stem = os.path.splitext(f)[0]
        lbl = os.path.join(test_dir, stem + ".txt")
        if os.path.exists(lbl):
            out.append((os.path.join(test_dir, f), lbl))
    return out


def evaluate_iou(
    predictor: GatePredictor,
    *,
    test_dir: str | os.PathLike = None,
    threshold: float = 0.5,
    n_visuals: int = 16,
) -> IoUResults:
    """Run GateNet on every test image and compute per-sample IoU.

    Stores up to ``n_visuals`` samples (raw image, GT mask, pred prob, pred
    binary, IoU, filename) for downstream visualisation.
    """
    test_dir = str(test_dir or (Path(SYN_ROOT) / "test"))
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"No test set at {test_dir}.\n"
            "Generate one first:  python scripts/prepare_test_set.py",
        )
    samples = _load_test_samples(test_dir)
    if not samples:
        raise ValueError(f"No labelled images found in {test_dir}.")

    ious: list[float] = []
    visuals: list[IoUSample] = []

    for img_path, lbl_path in samples:
        img_rgb, pred_prob = predictor.predict_image(img_path)
        pred_bin = (pred_prob >= threshold).astype(np.uint8)
        gt_mask = corners_to_mask(lbl_path, predictor.img_size)

        iou = compute_iou(pred_bin, gt_mask)
        ious.append(iou)

        if len(visuals) < n_visuals:
            visuals.append(IoUSample(
                img_rgb=img_rgb, gt_mask=gt_mask,
                pred_prob=pred_prob, pred_bin=pred_bin,
                iou=iou, fname=os.path.basename(img_path),
            ))

    return IoUResults(ious=np.array(ious), samples=samples, visuals=visuals)
