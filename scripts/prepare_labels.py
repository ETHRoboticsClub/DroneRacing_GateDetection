"""Generate synthetic 29-col labels (with outer corners) for train / val / test.

Walks the raw flight directories under ``data/``, picks frames whose label has
all 4 inner corners visible, synthesises the outer corners, and copies image +
label into ``data/synthetic/<split>/``.

Re-running is safe: existing files are kept and only the missing quota is
filled. Test set sources from val-split flights only (held out from training).

    python scripts/prepare_labels.py                 # train (2000) + val (500)
    python scripts/prepare_labels.py --split test    # test (300)
    python scripts/prepare_labels.py --split train --n 800 --scale 1.4
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

from gate_detection.data.discovery import collect_candidates, flight_dirs
from gate_detection.data.labels import make_synthetic_label_lines
from gate_detection.paths import SYN_ROOT


_DEFAULT_QUOTA = {"train": 2000, "val": 500, "test": 300}


def _existing_stems(out_dir: Path) -> set[str]:
    if not out_dir.is_dir():
        return set()
    return {os.path.splitext(f)[0] for f in os.listdir(out_dir) if f.endswith(".txt")}


def _select_seed(split: str, have: int) -> int:
    """Match the original per-split seed schemes for reproducibility."""
    if split == "test":
        return 99 + have
    return have    # train/val share the same scheme


def topup_split(split: str, target: int, scale: float) -> None:
    out_dir = Path(SYN_ROOT) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = _existing_stems(out_dir)
    have = len(existing)
    need = target - have
    if need <= 0:
        print(f"  {split}: already have {have}/{target}, nothing to do.")
        return

    print(f"  {split}: have {have}/{target}, need {need} more.")

    # Test set is sourced from the val-split flights and must not collide
    # with anything already in the val split.
    if split == "test":
        candidates = collect_candidates(flight_dirs("val"))
        forbidden = existing | _existing_stems(Path(SYN_ROOT) / "val")
    else:
        candidates = collect_candidates(flight_dirs(split))
        forbidden = existing

    candidates = [(img, lbl) for img, lbl in candidates
                  if os.path.splitext(os.path.basename(lbl))[0] not in forbidden]
    if len(candidates) < need:
        print(f"  Warning: only {len(candidates)} candidates available.")

    random.seed(_select_seed(split, have))
    selected = random.sample(candidates, min(need, len(candidates)))

    for img_path, lbl_path in selected:
        stem = os.path.splitext(os.path.basename(lbl_path))[0]
        lines = make_synthetic_label_lines(lbl_path, scale=scale)
        with open(out_dir / f"{stem}.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
        shutil.copy2(img_path, out_dir / f"{stem}.jpg")

    actual = len([f for f in os.listdir(out_dir) if f.endswith(".txt")])
    print(f"  {split}: added {len(selected)} samples → total {actual}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic outer-corner labels")
    p.add_argument("--split", nargs="+", default=["train", "val"],
                   choices=["train", "val", "test"])
    p.add_argument("--n",     type=int, default=None,
                   help="target sample count for each chosen split (default: 2000/500/300)")
    p.add_argument("--scale", type=float, default=1.4,
                   help="outer-corner radial scale factor (default: 1.4)")
    args = p.parse_args()

    for split in args.split:
        target = args.n if args.n is not None else _DEFAULT_QUOTA[split]
        topup_split(split, target, args.scale)

    print(f"\nDone. Synthetic data in {SYN_ROOT}/")


if __name__ == "__main__":
    main()
