"""Metrics for segmentation and corner localisation."""

from __future__ import annotations

import numpy as np
import torch
from torchmetrics.functional.classification import binary_jaccard_index


def compute_iou(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    """Binary IoU via torchmetrics. Returns 1.0 when both masks are empty."""
    pred_t = torch.as_tensor(pred_bin, dtype=torch.int64)
    gt_t = torch.as_tensor(gt_bin, dtype=torch.int64)
    if torch.logical_or(pred_t.bool(), gt_t.bool()).sum().item() == 0:
        return 1.0
    return float(binary_jaccard_index(pred_t, gt_t))


def inner_corner_error(detected: np.ndarray | None,
                       gt_gates: list[np.ndarray]) -> float | None:
    """Mean L2 px error between detected inner corners and the closest GT gate.

    Both arrays follow CORNER_TYPES order, so inner corners are indices 4..7.
    Returns ``None`` if detection failed or no GT gate is available.
    """
    if detected is None or len(gt_gates) == 0:
        return None
    inner_det = detected[4:8]
    best = None
    for gt in gt_gates:
        err = float(np.mean(np.linalg.norm(inner_det - gt[4:8], axis=1)))
        if best is None or err < best:
            best = err
    return best
