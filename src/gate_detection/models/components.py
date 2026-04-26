"""Reusable U-Net building blocks."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two 3x3 conv → BN → ReLU layers (the standard U-Net block)."""

    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        # Dropout has no learnable parameters, so sharing the same module
        # across both positions is intentional (same dropout rate, no
        # weight tying concern). Identity when disabled keeps the Sequential
        # shape consistent regardless of dropout_p.
        d = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        # bias=False on conv: the following BN absorbs any constant offset,
        # so the conv bias would be redundant and slow down training.
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            d,
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            d,
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """Downscale (max-pool 2x) → DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch, dropout_p=dropout_p))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Transposed-conv upscale → add skip → DoubleConv.

    Skip connection is fused via addition (not concatenation) — this keeps the
    decoder narrow and is the choice in the original GateNet description.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2, bias=False),
            nn.BatchNorm2d(in_ch),
        )
        self.conv = DoubleConv(in_ch, out_ch, dropout_p=dropout_p)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            # Handle odd input sizes by resampling to skip resolution.
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = x + skip
        return self.conv(x)


class OutConv(nn.Module):
    """1x1 conv → sigmoid producing a single-channel probability map."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)
