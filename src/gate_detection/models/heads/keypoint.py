"""Keypoint head — small Conv head producing per-corner heatmaps.

Tapped off the DeepLabV3+ decoder feature map (default 256 channels at
H/4 × W/4 with output_stride=16) and bilinearly upsampled to the input
resolution so soft-argmax decoding can reach sub-pixel accuracy without an
extra deconv stack.

Architecture (small on purpose — multi-task backbone is the load-bearing
part, the head should not dominate):

    Conv2d(in_channels, hidden, 3×3, padding=1)
    → BatchNorm2d(hidden)
    → ReLU(inplace=True)
    → Conv2d(hidden, n_keypoints, 1×1)
    → bilinear upsample to ``(out_h, out_w)``

A final sigmoid bounds the output to ``[0, 1]`` so the per-channel peak
confidence is interpretable as a probability from epoch 1, not just
post-convergence. The targets rendered by ``losses/keypoint.render_heatmap``
also live in ``[0, 1]`` (Gaussian peak max = 1.0), so weighted MSE on
sigmoid output is well-conditioned. Soft-argmax is unaffected — sigmoid
is monotonic so the argmax location is preserved.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        n_keypoints: int = 4,
        hidden: int = 64,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, n_keypoints, kernel_size=1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        # Match the He-style init that smp's segmentation head uses for its
        # final 1×1 conv; mismatched init scales between the seg and kp
        # heads can starve one of gradient early in training.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        x = self.net(feat)
        if x.shape[-2:] != tuple(out_size):
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x
