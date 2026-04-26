"""Raw flight discovery on disk.

Walks ``data/raw/autonomous/`` and ``data/piloted/`` for flight-* directories,
deterministically splits them 80 / 20 into train / val using ``random.seed(42)``,
and finds candidate frames whose label file has all 4 inner corners visible.

The split is deliberately frozen for reproducibility — do not change the seed
or sort order without a corresponding bump to the curated data layout.
"""

from __future__ import annotations

import glob
import os
import random
from pathlib import Path

from .labels import all4_visible
from gate_detection.paths import DATA_ROOT


def flight_dirs(split: str, data_root: str | os.PathLike = DATA_ROOT) -> list[str]:
    """Return the train- or val-split flight directories (deterministic)."""
    data_root = str(data_root)
    all_dirs: list[str] = []
    # sorted() before shuffle is critical for reproducibility — glob's order
    # is filesystem-dependent (e.g. ext4 ≠ btrfs) so without it the same
    # seed would produce different splits on different machines.
    for subdir in (os.path.join("raw", "autonomous"), "piloted"):
        all_dirs.extend(sorted(glob.glob(os.path.join(data_root, subdir, "flight-*"))))
    random.seed(42)
    shuffled = all_dirs[:]
    random.shuffle(shuffled)
    cut = int(0.8 * len(shuffled))
    return shuffled[:cut] if split == "train" else shuffled[cut:]


def collect_candidates(dirs: list[str]) -> list[tuple[str, str]]:
    """Return (image_path, label_path) pairs whose label has all 4 inner corners visible."""
    out: list[tuple[str, str]] = []
    for fdir in dirs:
        # tii-racing layout: each flight dir contains "camera_<flight_name>/"
        # for jpgs and "label_<flight_name>/" for matching .txt labels.
        flight_name = os.path.basename(fdir)
        lbl_dir = os.path.join(fdir, f"label_{flight_name}")
        cam_dir = os.path.join(fdir, f"camera_{flight_name}")
        if not os.path.isdir(lbl_dir):
            continue
        for lbl in sorted(glob.glob(os.path.join(lbl_dir, "*.txt"))):
            if not all4_visible(lbl):
                continue
            stem = os.path.splitext(os.path.basename(lbl))[0]
            img = os.path.join(cam_dir, stem + ".jpg")
            if os.path.exists(img):
                out.append((img, lbl))
    return out


def find_flight_paths(flight_root: str | os.PathLike) -> tuple[str, str, str | None]:
    """Resolve (camera_dir, label_dir, sync_csv) under a flight directory.

    Raises ``FileNotFoundError`` if camera/label dirs are missing; the sync CSV
    is optional and returned as ``None`` when absent.
    """
    flight_root = str(flight_root)
    name = os.path.basename(os.path.normpath(flight_root))

    cam_dir = os.path.join(flight_root, f"camera_{name}")
    if not os.path.isdir(cam_dir):
        raise FileNotFoundError(f"Camera directory not found: {cam_dir}")

    lbl_dir = os.path.join(flight_root, f"label_{name}")
    if not os.path.isdir(lbl_dir):
        raise FileNotFoundError(f"Label directory not found: {lbl_dir}")

    csv_path = os.path.join(flight_root, f"{name}_cam_ts_sync.csv")
    return cam_dir, lbl_dir, (csv_path if os.path.isfile(csv_path) else None)
