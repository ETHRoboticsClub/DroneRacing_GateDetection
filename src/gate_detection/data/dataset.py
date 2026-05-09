"""Dataset for GateNet training/validation."""

from __future__ import annotations

import os
from typing import Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .labels import corners_to_mask, parse_gates
from .transforms import apply_imagenet_norm, train_augment, train_augment_kp
from gate_detection.losses.keypoint import render_heatmap

MASK_SUFFIX = "_mask.png"


def _load_sam_mask(mask_path: str, out_size: int) -> np.ndarray:
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros((out_size, out_size), dtype=np.uint8)
    if m.shape != (out_size, out_size):
        m = cv2.resize(m, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def _label_path_for(img_path: str) -> str:
    """Sibling .txt label path next to an image, regardless of mask mode."""
    stem = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(os.path.dirname(img_path), stem + ".txt")


def _first_gate_inner(
    lbl_path: str,
    img_size: int,
    n_keypoints: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (corners_px (K, 2), vis (K,)) for the first gate in ``lbl_path``.

    "First gate" = first parseable line in the label file, matching the
    convention used by ``benchmark.py`` (priors are taken from
    ``gt_gates[0]``). Multi-gate scenes only supervise the first.

    If the label has no gates, returns an all-zero corner array with
    vis=0 — the loss masks invisible channels so this becomes a no-op
    sample for the keypoint head while the seg head still trains on it.
    """
    if not lbl_path or not os.path.exists(lbl_path):
        return (
            np.zeros((n_keypoints, 2), dtype=np.float32),
            np.zeros(n_keypoints, dtype=np.int32),
        )
    gates = parse_gates(lbl_path)
    if not gates:
        return (
            np.zeros((n_keypoints, 2), dtype=np.float32),
            np.zeros(n_keypoints, dtype=np.int32),
        )
    g = gates[0]
    # parse_gates returns inner corners normalised; scale to pixel coords
    # at img_size — augmentation works in pixel coords for clarity.
    corners = (g["inner"] * img_size).astype(np.float32)
    vis = np.array(g["inner_vis"], dtype=np.int32)
    if corners.shape[0] != n_keypoints:
        # Defensive: parse_gates currently always returns 4-corner inner;
        # if a future label format changes that, surface it loudly rather
        # than silently producing the wrong number of channels.
        raise ValueError(
            f"label {lbl_path} produced {corners.shape[0]} inner corners; "
            f"keypoint head expects {n_keypoints}"
        )
    return corners, vis


class GateDataset(Dataset):
    """Loads (image, mask) pairs from one or more flat directories.

    Each input directory should contain matching ``<stem>.jpg`` files plus
    either ``<stem>.txt`` label files (default) or ``<stem>_mask.png`` SAM
    ring masks (when ``use_sam_masks=True``).

    Images are resized to ``img_size`` × ``img_size``.  When
    ``use_sam_masks=False`` masks are rasterised from the label corners;
    when ``use_sam_masks=True`` they are loaded from the pre-generated PNG
    and samples without a PNG are skipped (run
    ``scripts/generate_sam_masks.py`` to fill them in).

    Args:
        sample_dirs:   list of directories to scan.
        img_size:      target square image size.
        augment:       if True, apply :func:`train_augment` to each sample.
        use_sam_masks: if True, load SAM ring masks from ``_mask.png`` files.
        imagenet_norm: if True, apply per-channel ImageNet mean/std (must
            match the predictor's preprocessing — kept in sync via the
            shared ``apply_imagenet_norm`` helper).
        keypoint_heatmap: if True, also emit Gaussian heatmap targets for
            the first gate's inner corners. Sample tuple becomes
            ``(img_t, mask_t, kp_target_t, kp_vis_t)``.
        heatmap_sigma: Gaussian σ in pixels at ``img_size`` resolution.
            σ=3 px at 384² gives a 9-px-wide effective peak — wide enough
            for early-training gradients, narrow enough for sub-pixel
            soft-argmax to stay accurate.
        n_keypoints: number of corner channels (=4 for inner-only, the
            current convention; outer corners are synthesised at inference).
    """

    def __init__(
        self,
        sample_dirs: Iterable[str],
        img_size: int = 384,
        augment: bool = True,
        use_sam_masks: bool = False,
        imagenet_norm: bool = False,
        keypoint_heatmap: bool = False,
        heatmap_sigma: float = 3.0,
        n_keypoints: int = 4,
    ):
        self.img_size = img_size
        self.augment = augment
        self.use_sam_masks = use_sam_masks
        self.imagenet_norm = imagenet_norm
        self.keypoint_heatmap = keypoint_heatmap
        self.heatmap_sigma = heatmap_sigma
        self.n_keypoints = n_keypoints
        # second element is mask_path (SAM mode) or lbl_path (label mode)
        self.samples: list[tuple[str, str]] = []

        skipped = 0
        for d in sample_dirs:
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if not f.endswith(".jpg"):
                    continue
                stem = os.path.splitext(f)[0]
                img_p = os.path.join(d, f)
                if use_sam_masks:
                    mask_p = os.path.join(d, stem + MASK_SUFFIX)
                    if not os.path.exists(mask_p):
                        skipped += 1
                        continue
                    self.samples.append((img_p, mask_p))
                else:
                    lbl = os.path.join(d, stem + ".txt")
                    if os.path.exists(lbl):
                        self.samples.append((img_p, lbl))

        if use_sam_masks and skipped:
            print(
                f"  [GateDataset] skipped {skipped} samples without {MASK_SUFFIX} "
                "(run scripts/generate_sam_masks.py to fill in)"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, secondary_path = self.samples[idx]

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        if self.use_sam_masks:
            mask = _load_sam_mask(secondary_path, self.img_size)
        else:
            mask = corners_to_mask(secondary_path, self.img_size)

        if self.keypoint_heatmap:
            # In SAM-mask mode the dataset path stores `mask_path` as
            # `secondary_path`; the keypoint label still lives at the
            # sibling .txt. Resolve it explicitly.
            lbl_path = (
                secondary_path
                if secondary_path.endswith(".txt")
                else _label_path_for(img_path)
            )
            corners_px, vis = _first_gate_inner(
                lbl_path, self.img_size, self.n_keypoints,
            )
            if self.augment:
                img, mask, corners_px, vis = train_augment_kp(
                    img, mask, corners_px, vis,
                )
            kp_target = render_heatmap(
                corners_px, vis,
                size=self.img_size,
                sigma=self.heatmap_sigma,
            )  # (K, H, W) float32
            kp_target_t = torch.from_numpy(kp_target)            # (K, H, W)
            kp_vis_t    = torch.from_numpy(vis.astype(np.int32))  # (K,)
        else:
            if self.augment:
                img, mask = train_augment(img, mask)

        # Train/inference parity: if imagenet_norm is on, apply the SAME
        # helper used by the predictor (single source of truth in
        # transforms.apply_imagenet_norm). The flag is persisted in the
        # checkpoint metadata so the predictor can replay it.
        if self.imagenet_norm:
            img_t = torch.from_numpy(apply_imagenet_norm(img)).permute(2, 0, 1).contiguous()
        else:
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()

        if self.keypoint_heatmap:
            return img_t, mask_t, kp_target_t, kp_vis_t
        return img_t, mask_t
