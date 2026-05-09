"""Dataset for GateNet training/validation."""

from __future__ import annotations

import os
from typing import Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .labels import corners_to_mask
from .transforms import apply_imagenet_norm, train_augment

MASK_SUFFIX = "_mask.png"


def _load_sam_mask(mask_path: str, out_size: int) -> np.ndarray:
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros((out_size, out_size), dtype=np.uint8)
    if m.shape != (out_size, out_size):
        m = cv2.resize(m, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


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
    """

    def __init__(
        self,
        sample_dirs: Iterable[str],
        img_size: int = 384,
        augment: bool = True,
        use_sam_masks: bool = False,
        imagenet_norm: bool = False,
    ):
        self.img_size = img_size
        self.augment = augment
        self.use_sam_masks = use_sam_masks
        self.imagenet_norm = imagenet_norm
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, secondary_path = self.samples[idx]

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        if self.use_sam_masks:
            mask = _load_sam_mask(secondary_path, self.img_size)
        else:
            mask = corners_to_mask(secondary_path, self.img_size)

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
        return img_t, mask_t
