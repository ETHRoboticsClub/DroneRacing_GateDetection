"""DeepLabV3+ model wrapper using segmentation_models_pytorch."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn

from .interface import SegmentationModelWrapper


def build_deeplabv3plus(
    *,
    encoder_name: str = "resnet34",
    encoder_weights: str | None = None,
    dropout_p: float = 0.0,
) -> SegmentationModelWrapper:
    """Build DeepLabV3+ as a single-head probability model."""
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation="sigmoid",
    )
    # NOTE: this wrapping changes the state-dict layout — keys go from
    # "segmentation_head.0.*" (no dropout) to "segmentation_head.1.0.*"
    # (with dropout). dropout_p must therefore be persisted in the
    # checkpoint's model_config and restored on load, otherwise the model
    # rebuilt for inference will not match the saved weights.
    if dropout_p > 0:
        model.segmentation_head = nn.Sequential(
            nn.Dropout2d(p=dropout_p),
            model.segmentation_head,
        )
    # Single output → ds_weights=(1.0,); the loss collapses to single-scale
    # Dice + BCE automatically when the output tuple has length 1.
    return SegmentationModelWrapper(model, ds_weights=(1.0,))
