"""Model factory for segmentation backbones."""

from __future__ import annotations

from .deeplabv3plus import build_deeplabv3plus
from .gatenet import build_gatenet
from .interface import ModelConfig, SegmentationModelWrapper


def build_model(cfg: ModelConfig) -> SegmentationModelWrapper:
    """Construct a wrapped segmentation model from :class:`ModelConfig`."""
    name = cfg.name.lower()
    if name == "gatenet":
        return build_gatenet(
            width_factor=cfg.width_factor,
            dropout_p=cfg.dropout_p,
        )
    if name == "deeplabv3plus":
        return build_deeplabv3plus(
            encoder_name=cfg.encoder_name,
            encoder_weights=cfg.encoder_weights,
            dropout_p=cfg.dropout_p,
        )
    raise ValueError(f"Unknown model name: {cfg.name}")
