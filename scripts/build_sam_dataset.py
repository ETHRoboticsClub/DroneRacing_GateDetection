"""
Build a wide-coverage gate dataset by random-sampling frames from every flight
and symlinking the image + label into data/sam/{train,val,test}/.

Uses the same seed=42 train/val flight split as the rest of the pipeline.

A frame is eligible if every gate in its label has at least --min-visible
inner corners (visibility flag > 0).  Default is 2, which keeps close-approach
frames (the most important moments for racing) while excluding pathological
1-corner labels.

Naming: data/sam/<split>/<flight>__<orig_stem>.{jpg,txt}
        (double underscore so flight prefix never collides with stems)

Test set is drawn from the val-split flights, excluding stems already chosen
for val. Re-running is idempotent (existing symlinks are kept; only the missing
quota is filled).

Run `scripts/generate_sam_masks.py` afterwards to produce the SAM ring masks.

Usage:
    python scripts/build_sam_dataset.py
    python scripts/build_sam_dataset.py --train 4000 --val 1000 --test 500
    python scripts/build_sam_dataset.py --min-visible 4   # stricter (all 4 corners)
"""

import argparse
import glob
import os
import random

from gate_detection.data.discovery import flight_dirs
from gate_detection.paths import DATA_ROOT

OUT_ROOT = str(DATA_ROOT / "sam")
SEP = "__"


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------
def _min_inner_visible(label_path: str, n: int) -> bool:
    """True iff every gate in the label has >= n visible inner corners."""
    with open(label_path) as f:
        lines = [l for l in f if l.strip()]
    if not lines:
        return False
    for line in lines:
        vals = list(map(float, line.split()))
        if len(vals) < 17:
            return False
        vis_count = sum(1 for k in range(4) if vals[7 + k * 3] > 0)
        if vis_count < n:
            return False
    return True


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def collect_candidates(dirs: list[str], min_visible: int) -> list[tuple[str, str, str]]:
    """Return [(flight_name, abs_img_path, abs_lbl_path), ...] for eligible frames."""
    out = []
    for fdir in dirs:
        flight  = os.path.basename(fdir)
        lbl_dir = os.path.join(fdir, f"label_{flight}")
        cam_dir = os.path.join(fdir, f"camera_{flight}")
        if not os.path.isdir(lbl_dir) or not os.path.isdir(cam_dir):
            continue
        for lbl in sorted(glob.glob(os.path.join(lbl_dir, "*.txt"))):
            if not _min_inner_visible(lbl, min_visible):
                continue
            stem = os.path.splitext(os.path.basename(lbl))[0]
            img  = os.path.join(cam_dir, stem + ".jpg")
            if os.path.exists(img):
                out.append((flight, os.path.abspath(img), os.path.abspath(lbl)))
    return out


def _existing_stems(out_dir: str) -> set[str]:
    if not os.path.isdir(out_dir):
        return set()
    return {os.path.splitext(f)[0] for f in os.listdir(out_dir) if f.endswith(".jpg")}


# ---------------------------------------------------------------------------
# Sampling + symlinking
# ---------------------------------------------------------------------------
def populate_split(
    split: str,
    candidates: list[tuple[str, str, str]],
    target: int,
    exclude_stems: set[str] | None = None,
    seed: int = 42,
) -> set[str]:
    out_dir = os.path.join(OUT_ROOT, split)
    os.makedirs(out_dir, exist_ok=True)
    have     = _existing_stems(out_dir)
    excluded = (exclude_stems or set()) | have

    pool = []
    for flight, img, lbl in candidates:
        orig_stem = os.path.splitext(os.path.basename(img))[0]
        out_stem  = f"{flight}{SEP}{orig_stem}"
        if out_stem in excluded:
            continue
        pool.append((out_stem, img, lbl))

    need = target - len(have)
    if need <= 0:
        print(f"[{split}] already have {len(have)}/{target}, nothing to add")
        return have

    rng = random.Random(seed)
    rng.shuffle(pool)
    selected = pool[:need]
    if len(selected) < need:
        print(f"[{split}] WARNING: only {len(selected)} candidates available "
              f"(asked for {need} more)")

    new: set[str] = set()
    for out_stem, img, lbl in selected:
        link_img = os.path.join(out_dir, out_stem + ".jpg")
        link_lbl = os.path.join(out_dir, out_stem + ".txt")
        if not os.path.exists(link_img):
            os.symlink(img, link_img)
        if not os.path.exists(link_lbl):
            os.symlink(lbl, link_lbl)
        new.add(out_stem)

    total = have | new
    print(f"[{split}] symlinked {len(new)} new samples -> total {len(total)}/{target}")
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Sample + symlink the SAM dataset.")
    p.add_argument("--train",       type=int, default=4000)
    p.add_argument("--val",         type=int, default=1000)
    p.add_argument("--test",        type=int, default=500)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--min-visible", type=int, default=2,
                   help="minimum visible inner corners required per gate (1-4, default 2)")
    args = p.parse_args()

    if not 1 <= args.min_visible <= 4:
        p.error("--min-visible must be in [1, 4]")

    print(f"Output root: {OUT_ROOT}")
    print(f"Eligibility: every gate has >= {args.min_visible} visible inner corner(s)")

    print("Discovering train candidates ...")
    train_cands = collect_candidates(flight_dirs("train"), args.min_visible)
    print(f"  {len(train_cands)} eligible frames across {len(flight_dirs('train'))} flights")

    print("Discovering val candidates ...")
    val_cands = collect_candidates(flight_dirs("val"), args.min_visible)
    print(f"  {len(val_cands)} eligible frames across {len(flight_dirs('val'))} flights")

    train_stems = populate_split("train", train_cands, args.train, seed=args.seed)
    val_stems   = populate_split("val",   val_cands,   args.val,   seed=args.seed)
    _           = populate_split("test",  val_cands,   args.test,
                                 exclude_stems=val_stems,
                                 seed=args.seed + 1)

    print("\nDone. Now run:")
    print("    python scripts/generate_sam_masks.py")


if __name__ == "__main__":
    main()
