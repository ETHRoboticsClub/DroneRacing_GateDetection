"""Common interface wrappers for segmentation models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    """Serializable model configuration stored in checkpoints."""

    name: str = "gatenet"
    width_factor: int = 4
    encoder_name: str = "resnet34"
    encoder_weights: str | None = None
    dropout_p: float = 0.0


class SegmentationModelWrapper(nn.Module):
    """Unified interface for all segmentation backbones in this project.

    ``forward`` returns a tuple of probability maps ordered from coarse to fine.
    For single-head models, this tuple has length 1.
    """

    def __init__(self, model: nn.Module, *, ds_weights: tuple[float, ...]):
        super().__init__()
        self.model = model
        self.ds_weights = ds_weights

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        y = self.model(x)
        if isinstance(y, (tuple, list)):
            return tuple(y)
        return (y,)
