"""Model factory for segmentation backbones."""

from __future__ import annotations

from .deeplabv3plus import build_deeplabv3plus
from .gatenet import build_gatenet
from .interface import ModelConfig, SegmentationModelWrapper


def build_model(cfg: ModelConfig) -> SegmentationModelWrapper:
    """Construct a wrapped segmentation model from :class:`ModelConfig`."""
    name = cfg.name.lower()
    if name == "gatenet":
        if cfg.keypoint_head:
            raise NotImplementedError(
                "keypoint_head is wired only for deeplabv3plus today; "
                "set model.name=deeplabv3plus or disable model.keypoint_head."
            )
        return build_gatenet(
            width_factor=cfg.width_factor,
            dropout_p=cfg.dropout_p,
        )
    if name == "deeplabv3plus":
        return build_deeplabv3plus(
            encoder_name=cfg.encoder_name,
            encoder_weights=cfg.encoder_weights,
            dropout_p=cfg.dropout_p,
            keypoint_head=cfg.keypoint_head,
            n_keypoints=cfg.n_keypoints,
            next_gate_head=cfg.next_gate_head,
        )
    raise ValueError(f"Unknown model name: {cfg.name}")
