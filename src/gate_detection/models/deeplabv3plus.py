"""DeepLabV3+ model wrapper using segmentation_models_pytorch.

Two build paths:

* ``keypoint_head=False`` (default) — single-head segmentation, identical
  behaviour to every checkpoint trained before plan P1-1.
* ``keypoint_head=True`` — segmentation head AND a parallel keypoint head
  (4-channel heatmap, one channel per inner gate corner), both tapped off
  the shared encoder + ASPP decoder. Forward of the inner module returns
  a dict ``{"seg": Tensor, "keypoint": Tensor}``; the
  ``SegmentationModelWrapper`` unpacks it for callers that only need the
  seg tuple, and ``forward_multi`` exposes both for the multi-task loss.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from .heads import KeypointHead
from .interface import SegmentationModelWrapper


# Decoder feature width passed explicitly to smp.DeepLabV3Plus so the
# keypoint head's in_channels is wired from a value we own — relying on
# smp's default is a silent-fallback per CLAUDE.md §6 (and the smp
# decoder does not expose `out_channels`, making any post-hoc assertion
# tautological).
_DEEPLABV3PLUS_DECODER_CHANNELS = 256


class _DeepLabV3PlusWithKeypoint(nn.Module):
    """Single forward producing both seg and keypoint outputs.

    Replicates the smp ``DeepLabV3Plus.forward`` body so we can tap the
    decoder feature map (input to ``segmentation_head``) for the keypoint
    head. Keeping the seg path identical to upstream means a checkpoint
    trained with this wrapper is still loadable as a plain DeepLabV3+
    after the keypoint state-dict slice is dropped — useful for falling
    back to single-task inference.
    """

    def __init__(self, smp_model: nn.Module, keypoint_head: KeypointHead):
        super().__init__()
        self.smp_model = smp_model
        self.keypoint_head = keypoint_head

    def forward(self, x: torch.Tensor) -> dict:
        # The smp DeepLabV3Plus.forward body, re-implemented with one extra
        # tap into the decoder output. Kept in sync with smp's own forward
        # at smp.base.SegmentationModel.forward (which is encoder → decoder
        # → segmentation_head, optionally classification_head). We keep
        # smp's input-shape guard so a non-mod-16 input (output_stride=16)
        # raises a useful error rather than silently misaligning.
        if hasattr(self.smp_model, "check_input_shape"):
            self.smp_model.check_input_shape(x)
        features = self.smp_model.encoder(x)
        decoder_out = self.smp_model.decoder(features)
        seg = self.smp_model.segmentation_head(decoder_out)
        kp = self.keypoint_head(decoder_out, out_size=x.shape[-2:])
        return {"seg": seg, "keypoint": kp}


def build_deeplabv3plus(
    *,
    encoder_name: str = "resnet34",
    encoder_weights: str | None = None,
    dropout_p: float = 0.0,
    keypoint_head: bool = False,
    n_keypoints: int = 4,
) -> SegmentationModelWrapper:
    """Build DeepLabV3+ as a single-head probability model.

    When ``keypoint_head=True`` the returned wrapper also exposes a
    4-channel keypoint heatmap (per-inner-corner Gaussian peaks) via
    :meth:`SegmentationModelWrapper.forward_multi`.
    """
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation="sigmoid",
        decoder_channels=_DEEPLABV3PLUS_DECODER_CHANNELS,
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
    if keypoint_head:
        head = KeypointHead(
            in_channels=_DEEPLABV3PLUS_DECODER_CHANNELS,
            n_keypoints=n_keypoints,
        )
        wrapped = _DeepLabV3PlusWithKeypoint(model, head)
        # Single seg output → ds_weights=(1.0,); the loss collapses to
        # single-scale Dice + BCE automatically.
        return SegmentationModelWrapper(wrapped, ds_weights=(1.0,))
    # Single output → ds_weights=(1.0,); the loss collapses to single-scale
    # Dice + BCE automatically when the output tuple has length 1.
    return SegmentationModelWrapper(model, ds_weights=(1.0,))
