"""Segmentation losses for GateNet (Dice + BCE with deep supervision)."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from kornia.losses import dice_loss as kornia_dice_loss


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss via Kornia, adapted for binary masks.

    GateNet emits a single foreground probability channel. Kornia's Dice loss
    expects class probabilities with ``C >= 2`` and integer class labels.
    """
    # Build a 2-channel (bg, fg) probability tensor from the single foreground
    # channel — kornia's API needs (N, C, H, W) class probs with C>=2 even
    # for the binary case. clamp guards against tiny sigmoid overflow under AMP.
    pred_fg = pred.float().clamp(0.0, 1.0)
    pred_bg = 1.0 - pred_fg
    pred_2c = torch.cat([pred_bg, pred_fg], dim=1)
    # kornia expects integer class indices, not one-hot; squeeze the
    # (N, 1, H, W) GT into (N, H, W) and cast to int64.
    target_idx = target.squeeze(1).to(dtype=torch.int64)
    return kornia_dice_loss(pred_2c, target_idx, eps=eps)


def gate_loss(
    outputs: Sequence[torch.Tensor],
    mask: torch.Tensor,
    deep_supervision_weights: Sequence[float] = (4, 2, 1, 1, 1),
    bce_weight: float = 2.0,
) -> torch.Tensor:
    """Deep-supervision combined loss across all GateNet output scales.

    For each output scale ``i``:
        L_i = Dice(pred_i, gt_i) + bce_weight * BCE(pred_i, gt_i)
    where ``gt_i`` is the GT mask resampled to ``pred_i``'s resolution
    (nearest-neighbour to keep the labels strictly binary).

    The total loss is ``sum_i w_i * L_i``.
    """
    total: torch.Tensor = torch.tensor(0.0, device=mask.device)
    for w, pred in zip(deep_supervision_weights, outputs):
        # Nearest-neighbour interpolation keeps the GT strictly binary at
        # every scale; bilinear would introduce 0.5 fractions that Dice
        # would treat as half-foreground pixels.
        gt = F.interpolate(mask.float(), size=pred.shape[2:], mode="nearest")
        d  = dice_loss(pred, gt)
        # F.binary_cross_entropy on probabilities (not logits) is not safe
        # under autocast — bf16/fp16 inputs can produce ±inf when log() hits
        # values rounded to 0 or 1. Disable autocast here and force fp32.
        # The logits variant (binary_cross_entropy_with_logits) doesn't
        # apply because GateNet's heads end in sigmoid.
        with torch.autocast(device_type=pred.device.type, enabled=False):
            b = F.binary_cross_entropy(pred.float(), gt.float())
        total = total + w * (d + bce_weight * b)
    return total
