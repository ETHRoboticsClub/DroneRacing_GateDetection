"""Dataset for GateNet training/validation."""

from __future__ import annotations

import os
from typing import Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .labels import corners_to_mask
from .transforms import train_augment


class GateDataset(Dataset):
    """Loads (image, mask) pairs from one or more flat directories.

    Each input directory should contain matching ``<stem>.jpg`` and
    ``<stem>.txt`` files (the standard ``data/synthetic/<split>/`` layout).
    Images are resized to ``img_size`` × ``img_size``; masks are rasterised
    from the label corners at the same resolution.

    Args:
        sample_dirs: list of directories to scan.
        img_size:    target square image size.
        augment:     if True, apply :func:`train_augment` to each sample.
    """

    def __init__(self, sample_dirs: Iterable[str], img_size: int = 384, augment: bool = True):
        self.img_size = img_size
        self.augment = augment
        self.samples: list[tuple[str, str]] = []

        for d in sample_dirs:
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if not f.endswith(".jpg"):
                    continue
                stem = os.path.splitext(f)[0]
                lbl = os.path.join(d, stem + ".txt")
                if os.path.exists(lbl):
                    self.samples.append((os.path.join(d, f), lbl))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, lbl_path = self.samples[idx]

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = corners_to_mask(lbl_path, self.img_size)

        if self.augment:
            img, mask = train_augment(img, mask)

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        return img_t, mask_t
