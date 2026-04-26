"""GateNet — U-Net-style segmentation network with deep supervision.

Architecture (with channel divisor ``f``; default f=4):
    Encoder : inc-(64/f) → down1-(128/f) → down2-(256/f) → down3-(512/f) → down4-(512/f)
    Decoder : up1-(256/f) → up2-(128/f) → up3-(64/f) → up4-(64/f)

Forward returns five sigmoid probability maps (y0 … y4) at increasing
resolutions; only y4 (full input resolution) is used at deployment, while
y0…y3 supply the deep-supervision losses during training.
"""

from __future__ import annotations

import torch.nn as nn

from .components import DoubleConv, Down, OutConv, Up
from .interface import SegmentationModelWrapper


class GateNet(nn.Module):
    """U-Net segmentation network with deep supervision."""

    def __init__(self, f: int = 4, dropout_p: float = 0.0):
        super().__init__()
        c = [64 // f, 128 // f, 256 // f, 512 // f, 512 // f]

        # Encoder
        self.inc   = DoubleConv(3, c[0], dropout_p=dropout_p)
        self.down1 = Down(c[0], c[1], dropout_p=dropout_p)
        self.down2 = Down(c[1], c[2], dropout_p=dropout_p)
        self.down3 = Down(c[2], c[3], dropout_p=dropout_p)
        self.down4 = Down(c[3], c[4], dropout_p=dropout_p)

        # Decoder
        self.up1 = Up(c[4],      256 // f, dropout_p=dropout_p)
        self.up2 = Up(256 // f,  128 // f, dropout_p=dropout_p)
        self.up3 = Up(128 // f,   64 // f, dropout_p=dropout_p)
        self.up4 = Up( 64 // f,   64 // f, dropout_p=dropout_p)

        # Deep-supervision output heads (one per scale)
        self.outc0 = OutConv(c[4])
        self.outc1 = OutConv(256 // f)
        self.outc2 = OutConv(128 // f)
        self.outc3 = OutConv( 64 // f)
        self.outc4 = OutConv( 64 // f)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder (s* = skip features)
        s4 = self.inc(x)
        s3 = self.down1(s4)
        s2 = self.down2(s3)
        s1 = self.down3(s2)
        b  = self.down4(s1)

        y0 = self.outc0(b)

        d1 = self.up1(b,  s1); y1 = self.outc1(d1)
        d2 = self.up2(d1, s2); y2 = self.outc2(d2)
        d3 = self.up3(d2, s3); y3 = self.outc3(d3)
        d4 = self.up4(d3, s4); y4 = self.outc4(d4)

        return y0, y1, y2, y3, y4


def build_gatenet(*, width_factor: int = 4, dropout_p: float = 0.0) -> SegmentationModelWrapper:
    """Build GateNet wrapped in the shared segmentation-model interface."""
    model = GateNet(f=width_factor, dropout_p=dropout_p)
    return SegmentationModelWrapper(model, ds_weights=(4.0, 2.0, 1.0, 1.0, 1.0))
