"""Common interface wrappers for segmentation models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    """Serializable model configuration stored in checkpoints.

    Adding a field here means it MUST also round-trip through the
    trainer's checkpoint save and the predictor's checkpoint load —
    every architectural field that affects the state-dict layout has to be
    rebuildable from the saved metadata, otherwise inference breaks
    silently when a new field is introduced.
    """

    name: str = "gatenet"
    width_factor: int = 4
    encoder_name: str = "resnet34"
    encoder_weights: str | None = None
    dropout_p: float = 0.0
    # Multi-task keypoint head (plan P1-1). When True the model produces
    # an extra (B, n_keypoints, H, W) heatmap output alongside the
    # segmentation mask. Default False keeps every pre-P1-1 checkpoint
    # loadable unchanged.
    keypoint_head: bool = False
    n_keypoints: int = 4


class SegmentationModelWrapper(nn.Module):
    """Unified interface for all segmentation backbones in this project.

    ``forward`` returns a tuple of probability maps ordered from coarse to fine.
    For single-head models, this tuple has length 1. Backbones that also
    expose a keypoint head pack their two outputs into a dict
    ``{"seg": tuple|tensor, "keypoint": tensor}`` from their inner forward;
    this wrapper unpacks that into the same tuple-of-segmentation-maps
    contract so existing callers (``gate_loss``, IoU metric, predictor
    full-res tap at index ``[-1]``) keep working unchanged. Multi-task
    callers use :meth:`forward_multi` instead.
    """

    def __init__(self, model: nn.Module, *, ds_weights: tuple[float, ...]):
        super().__init__()
        self.model = model
        self.ds_weights = ds_weights

    @staticmethod
    def _to_seg_tuple(seg) -> tuple[torch.Tensor, ...]:
        if isinstance(seg, (tuple, list)):
            return tuple(seg)
        return (seg,)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        y = self.model(x)
        if isinstance(y, dict):
            return self._to_seg_tuple(y["seg"])
        return self._to_seg_tuple(y)

    def forward_multi(
        self, x: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor | None]:
        """Multi-task forward: returns ``(seg_tuple, keypoint_tensor_or_None)``.

        ``keypoint_tensor`` is ``None`` for backbones without a keypoint
        head; callers should branch on that and skip the keypoint loss /
        decode rather than special-case the model class.
        """
        y = self.model(x)
        if isinstance(y, dict):
            return self._to_seg_tuple(y["seg"]), y.get("keypoint")
        return self._to_seg_tuple(y), None
