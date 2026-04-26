"""Download background images for synthetic data augmentation.

Streams ``Andron00e/Places365-custom`` from HuggingFace and saves resized
640×480 JPEGs to ``data/raw/backgrounds/``.

    pip install datasets pillow
    python scripts/download_backgrounds.py
    python scripts/download_backgrounds.py --n 200 --out backgrounds/
"""

from __future__ import annotations

import argparse
import os

from gate_detection.paths import BG_ROOT


def download(n: int, out_dir: str) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: uv pip install datasets pillow")
        return

    os.makedirs(out_dir, exist_ok=True)

    print("Loading dataset index (streaming)...")
    ds = load_dataset(
        "Andron00e/Places365-custom",
        split="train", streaming=True, trust_remote_code=True,
    )

    saved = 0
    for sample in ds:
        if saved >= n:
            break
        img = sample.get("image") or sample.get("img")
        if img is None:
            continue
        path = os.path.join(out_dir, f"bg_{saved:04d}.jpg")
        img.convert("RGB").resize((640, 480)).save(path, "JPEG", quality=90)
        saved += 1
        if saved % 20 == 0:
            print(f"  {saved}/{n}")

    print(f"\nSaved {saved} backgrounds to {out_dir}/")


def main() -> None:
    p = argparse.ArgumentParser(description="Download background images")
    p.add_argument("--n",   type=int, default=300)
    p.add_argument("--out", type=str, default=str(BG_ROOT))
    args = p.parse_args()
    download(args.n, args.out)


if __name__ == "__main__":
    main()
