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
    """Single forward producing seg + keypoint + (optional) next-gate-mask.

    Replicates the smp ``DeepLabV3Plus.forward`` body so we can tap the
    decoder feature map (input to ``segmentation_head``) for the keypoint
    head AND, when ``next_gate_head`` is provided, for the next-gate
    head (report 15). Keeping the seg path identical to upstream means
    a checkpoint trained with this wrapper is still loadable as a plain
    DeepLabV3+ after the extra heads' state-dict slices are dropped —
    useful for falling back to single-task inference.
    """

    def __init__(
        self,
        smp_model: nn.Module,
        keypoint_head: KeypointHead,
        next_gate_head: KeypointHead | None = None,
    ):
        super().__init__()
        self.smp_model = smp_model
        self.keypoint_head = keypoint_head
        # Optional next-gate-mask head (n_keypoints=1, sigmoid).
        # Stored as None when not configured so the state-dict layout
        # is identical to pre-report-15 keypoint checkpoints.
        self.next_gate_head = next_gate_head

    def forward(self, x: torch.Tensor) -> dict:
        # The smp DeepLabV3Plus.forward body, re-implemented with extra
        # taps off decoder_out. Kept in sync with smp's own forward at
        # smp.base.SegmentationModel.forward (encoder → decoder →
        # segmentation_head, optionally classification_head). The input-
        # shape guard from smp is preserved so a non-mod-16 input raises
        # a useful error rather than silently misaligning.
        if hasattr(self.smp_model, "check_input_shape"):
            self.smp_model.check_input_shape(x)
        features = self.smp_model.encoder(x)
        decoder_out = self.smp_model.decoder(features)
        seg = self.smp_model.segmentation_head(decoder_out)
        kp = self.keypoint_head(decoder_out, out_size=x.shape[-2:])
        out = {"seg": seg, "keypoint": kp}
        if self.next_gate_head is not None:
            out["next_gate"] = self.next_gate_head(
                decoder_out, out_size=x.shape[-2:],
            )
        return out


def build_deeplabv3plus(
    *,
    encoder_name: str = "resnet34",
    encoder_weights: str | None = None,
    dropout_p: float = 0.0,
    keypoint_head: bool = False,
    n_keypoints: int = 4,
    next_gate_head: bool = False,
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
        # Optional next-gate-mask head (report 15) — same architecture as
        # KeypointHead but emits a single sigmoid channel for the binary
        # "this region IS the next gate" mask. Tapped off the same
        # decoder feature map so the existing seg + kp heads can share
        # the encoder/decoder representation.
        ng_head = None
        if next_gate_head:
            ng_head = KeypointHead(
                in_channels=_DEEPLABV3PLUS_DECODER_CHANNELS,
                n_keypoints=1,
            )
        wrapped = _DeepLabV3PlusWithKeypoint(model, head, next_gate_head=ng_head)
        # Single seg output → ds_weights=(1.0,); the loss collapses to
        # single-scale Dice + BCE automatically.
        return SegmentationModelWrapper(wrapped, ds_weights=(1.0,))
    if next_gate_head:
        # Defensive — without keypoint_head=True there's no point training
        # the next-gate-mask head; the predictor would have no corner
        # heatmap to mask. CLAUDE.md §6: surface the inconsistency.
        raise ValueError(
            "next_gate_head=True requires keypoint_head=True — the "
            "next-gate mask is used to ROI-filter the kp heatmap at "
            "inference, so a model without a kp head can't consume it."
        )
    # Single output → ds_weights=(1.0,); the loss collapses to single-scale
    # Dice + BCE automatically when the output tuple has length 1.
    return SegmentationModelWrapper(model, ds_weights=(1.0,))
