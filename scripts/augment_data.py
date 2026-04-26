"""Offline synthetic data augmentation.

For each curated 29-col sample in ``data/synthetic/<split>/``, generate
``--mult`` augmented copies into ``data/synthetic/<split>_aug/``.

    python scripts/augment_data.py                       # 5x train + val
    python scripts/augment_data.py --mult 10 --split train
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from gate_detection.data.synthesis import augment_sample
from gate_detection.paths import SYN_ROOT


def augment_split(split: str, mult: int) -> None:
    in_dir = Path(SYN_ROOT) / split
    out_dir = Path(SYN_ROOT) / f"{split}_aug"
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted({os.path.splitext(f)[0]
                    for f in os.listdir(in_dir) if f.endswith(".txt")})
    print(f"{split}: {len(stems)} curated samples × {mult} → target {len(stems) * mult}")

    generated = 0
    for stem in stems:
        img_p = in_dir / f"{stem}.jpg"
        lbl_p = in_dir / f"{stem}.txt"
        for j in range(mult):
            out_img = out_dir / f"{stem}_aug{j:02d}.jpg"
            out_lbl = out_dir / f"{stem}_aug{j:02d}.txt"
            if out_img.exists():
                generated += 1
                continue
            if augment_sample(str(img_p), str(lbl_p), str(out_img), str(out_lbl)):
                generated += 1

    print(f"  → {generated} augmented samples in {out_dir}/")


def main() -> None:
    p = argparse.ArgumentParser(description="Offline synthetic data augmentation")
    p.add_argument("--split", nargs="+", default=["train", "val"])
    p.add_argument("--mult",  type=int, default=5,
                   help="augmented copies per curated sample")
    args = p.parse_args()

    for s in args.split:
        augment_split(s, args.mult)


if __name__ == "__main__":
    main()
