"""Inference wrapper around GateNet.

The :class:`GatePredictor` handles the standard preprocessing
(BGR → RGB, resize to ``img_size``, normalise to ``[0,1]``, NCHW tensor)
and returns the full-resolution probability map ``y4`` from GateNet.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch

from gate_detection.config import get_device
from gate_detection.models import ModelConfig, SegmentationModelWrapper, build_model


def _normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop torch.compile wrapper prefixes (e.g. ``_orig_mod.``) when present.

    The trainer already strips ``_orig_mod.`` via ``_base_model`` before
    saving, so this is mostly defensive — covers checkpoints saved by older
    versions of the trainer or by external tooling.
    """
    prefix = "_orig_mod."
    if any(k.startswith(prefix) for k in state_dict):
        return {
            (k[len(prefix):] if k.startswith(prefix) else k): v
            for k, v in state_dict.items()
        }
    return state_dict


def _load_state_dict_flexible(
    model: SegmentationModelWrapper,
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Load either wrapped (model.*) or legacy raw-backbone checkpoints.

    ``SegmentationModelWrapper`` stores the actual backbone under ``self.model``,
    so wrapped state-dicts have a ``model.`` prefix on every key. Legacy
    GateNet checkpoints predate the wrapper and were saved as raw backbone
    state-dicts (keys like ``inc.net.0.weight``); detect by absence of the
    prefix and load directly into the inner backbone.
    """
    has_wrapper_prefix = any(k.startswith("model.") for k in state_dict)
    if has_wrapper_prefix:
        model.load_state_dict(state_dict)
        return

    # Legacy checkpoints saved raw backbone keys (e.g. "inc.*").
    model.model.load_state_dict(state_dict)


def _load_model_config(
    ckpt: dict,
    *,
    model_name: str,
    width_factor: int,
    encoder_name: str,
    encoder_weights: str | None,
    dropout_p: float,
) -> ModelConfig:
    """Resolve model config from checkpoint first, then explicit args.

    The saved ``model_config`` is authoritative — every architectural field
    that affects the state-dict layout (including ``dropout_p``, which gates
    the dropout-wrapped segmentation head) must round-trip from it.
    """
    saved = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}
    return ModelConfig(
        name=saved.get("name", model_name),
        width_factor=int(saved.get("width_factor", width_factor)),
        encoder_name=saved.get("encoder_name", encoder_name),
        encoder_weights=saved.get("encoder_weights", encoder_weights),
        dropout_p=float(saved.get("dropout_p", dropout_p)),
    )


def load_checkpoint(
    path: str | os.PathLike,
    *,
    model_name: str = "gatenet",
    width_factor: int = 4,
    encoder_name: str = "resnet34",
    encoder_weights: str | None = None,
    dropout_p: float = 0.0,
    device: str | None = None,
) -> tuple[SegmentationModelWrapper, dict]:
    """Load a segmentation checkpoint. Returns ``(model.eval(), checkpoint_dict)``."""
    device = device or get_device()
    payload = torch.load(str(path), map_location=device, weights_only=True)

    # Two on-disk formats:
    #   * dict with metadata: {"model": state_dict, "epoch": ..., "model_config": ...}
    #   * raw state_dict (rare; e.g. third-party weights or torch.save(model.state_dict()))
    if isinstance(payload, dict) and "model" in payload:
        ckpt = payload
        state_dict = ckpt["model"]
    else:
        state_dict = payload
        ckpt = {"model": state_dict}

    model_cfg = _load_model_config(
        ckpt,
        model_name=model_name,
        width_factor=width_factor,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        dropout_p=dropout_p,
    )
    model = build_model(model_cfg).to(device)

    state_dict = _normalize_state_dict_keys(state_dict)
    ckpt["model"] = state_dict
    ckpt.setdefault("model_config", {
        "name": model_cfg.name,
        "width_factor": model_cfg.width_factor,
        "encoder_name": model_cfg.encoder_name,
        "encoder_weights": model_cfg.encoder_weights,
        "dropout_p": model_cfg.dropout_p,
    })
    _load_state_dict_flexible(model, state_dict)
    model.eval()
    return model, ckpt


class GatePredictor:
    """Run GateNet on single images or batches of preprocessed RGB tensors."""

    def __init__(
        self,
        checkpoint: str | os.PathLike,
        *,
        img_size: int = 384,
        model_name: str = "gatenet",
        width_factor: int = 4,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = None,
        dropout_p: float = 0.0,
        device: str | None = None,
    ):
        self.device = device or get_device()
        self.img_size = img_size
        self.model, self.checkpoint_meta = load_checkpoint(
            checkpoint,
            model_name=model_name,
            width_factor=width_factor,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            dropout_p=dropout_p,
            device=self.device,
        )
        # Replay the data preprocessing the checkpoint was trained with.
        # Old checkpoints (no data_config field) default to imagenet_norm=False
        # — same behaviour as before this flag was introduced. We log the
        # fallback explicitly per CLAUDE.md §6 (no silent defaults).
        if "data_config" not in self.checkpoint_meta:
            print(
                f"[WARN] checkpoint at {checkpoint}: no data_config; "
                "defaulting imagenet_norm=False (legacy behaviour)",
                flush=True,
            )
        data_cfg = self.checkpoint_meta.get("data_config", {}) or {}
        self.imagenet_norm: bool = bool(data_cfg.get("imagenet_norm", False))

    @torch.no_grad()
    def predict_image(self, img_path: str | os.PathLike) -> tuple[np.ndarray, np.ndarray]:
        """Read an image from disk → (resized RGB uint8, prediction float32 [0,1])."""
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        img_t, img_rgb = self._preprocess_bgr(img_bgr)
        pred = self._predict_tensor(img_t).squeeze().cpu().numpy()
        return img_rgb, pred

    @torch.no_grad()
    def _predict_tensor(self, img_t: torch.Tensor) -> torch.Tensor:
        """Forward pass returning the full-resolution probability map (NHW or NCHW)."""
        return self.model(img_t)[-1]

    def _preprocess_bgr(self, img_bgr: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        """BGR ndarray → (NCHW tensor on device, RGB ``img_size×img_size`` uint8)."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_res = cv2.resize(img_rgb, (self.img_size, self.img_size))
        if self.imagenet_norm:
            # Match the training-time path in GateDataset.__getitem__ via the
            # single shared helper in data/transforms.py.
            from gate_detection.data.transforms import apply_imagenet_norm
            img_t = torch.from_numpy(apply_imagenet_norm(img_res)).permute(2, 0, 1).contiguous()
        else:
            img_t = torch.from_numpy(img_res).permute(2, 0, 1).float() / 255.0
        return img_t.unsqueeze(0).to(self.device), img_res
