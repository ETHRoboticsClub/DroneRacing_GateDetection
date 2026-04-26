"""Interactive label viewer with keep / discard.

Controls:
    SPACE or K  → keep, next
    D           → discard (deletes .jpg + .txt from whichever split it's in), next
    B           → back (view previous, cannot un-discard)
    Q / ESC     → quit

    python scripts/viewer.py
    python scripts/viewer.py --shuffle
    python scripts/viewer.py --all --split train val
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np

from gate_detection.data.labels import parse_gates
from gate_detection.paths import SYN_ROOT


REVIEWED_LOG = Path(SYN_ROOT) / ".reviewed"


def load_reviewed() -> set[str]:
    if not REVIEWED_LOG.exists():
        return set()
    with open(REVIEWED_LOG) as f:
        return {l.strip() for l in f if l.strip()}


def mark_reviewed(stem: str) -> None:
    with open(REVIEWED_LOG, "a") as f:
        f.write(stem + "\n")


def load_samples(shuffle: bool, all_samples: bool,
                 splits: tuple[str, ...] = ("train", "val", "test")
                 ) -> list[tuple[str, str]]:
    reviewed = set() if all_samples else load_reviewed()
    entries: list[tuple[str, str]] = []
    for split in splits:
        split_dir = Path(SYN_ROOT) / split
        if not split_dir.is_dir():
            continue
        for f in sorted(os.listdir(split_dir)):
            if f.endswith(".txt"):
                stem = os.path.splitext(f)[0]
                if stem not in reviewed:
                    entries.append((stem, str(split_dir)))
    if shuffle:
        random.shuffle(entries)
    return entries


def _visible_corners(gate: dict, key: str, vis_key: str) -> np.ndarray:
    pts = gate[key]
    if pts is None:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array([p for p, v in zip(pts, gate[vis_key]) if v > 0],
                    dtype=np.float32)


def draw_frame(img_bgr: np.ndarray, gates, idx: int, total: int, stem: str) -> np.ndarray:
    H, W = img_bgr.shape[:2]
    vis = img_bgr.copy()

    for g in gates:
        for key, vk, color in [("inner", "inner_vis", (0, 255, 0)),
                               ("outer", "outer_vis", (0, 165, 255))]:
            pts_norm = _visible_corners(g, key, vk)
            if len(pts_norm) >= 2:
                pts = (pts_norm * [W, H]).astype(np.int32)
                cv2.polylines(vis, [pts], isClosed=len(pts) == 4, color=color, thickness=2)
                for p in pts:
                    cv2.circle(vis, tuple(p), 6, color, -1)
                    cv2.circle(vis, tuple(p), 6, (0, 0, 0), 1)

    cv2.rectangle(vis, (0, 0), (W, 38), (30, 30, 30), -1)
    cv2.putText(vis, f"[{idx + 1}/{total}]  {stem}",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(vis, (0, H - 30), (W, H), (30, 30, 30), -1)
    cv2.putText(vis, "SPACE/K=keep   D=discard   B=back   Q=quit",
                (8, H - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    return vis


def run(shuffle: bool, all_samples: bool, splits: tuple[str, ...]) -> None:
    samples = load_samples(shuffle, all_samples, splits)
    discarded = 0

    if not samples:
        print("No new samples to review. Use --all to review everything.")
        return
    print(f"{len(samples)} {'total' if all_samples else 'new'} samples. "
          "SPACE/K=keep  D=discard  B=back  Q=quit")

    cv2.namedWindow("viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("viewer", 900, 700)

    i = 0
    while i < len(samples):
        stem, split_dir = samples[i]
        img_p = os.path.join(split_dir, stem + ".jpg")
        lbl_p = os.path.join(split_dir, stem + ".txt")

        if not os.path.exists(img_p):
            samples.pop(i)
            continue

        img = cv2.imread(img_p)
        gates = [g for g in parse_gates(lbl_p) if g["outer"] is not None]
        split_name = os.path.basename(split_dir)
        frame = draw_frame(img, gates, i, len(samples), f"[{split_name}] {stem}")
        cv2.imshow("viewer", frame)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("b"):
            i = max(0, i - 1)
        elif key == ord("d"):
            os.remove(img_p)
            os.remove(lbl_p)
            samples.pop(i)
            discarded += 1
            mark_reviewed(stem)
            print(f"  Discarded: {stem}  ({discarded} total discarded)")
        else:
            mark_reviewed(stem)
            i += 1

    cv2.destroyAllWindows()
    print(f"\nSession done. Discarded {discarded} samples.")
    if discarded:
        print("  To top up:  python scripts/prepare_labels.py --split test")
        print("              python scripts/prepare_labels.py            (train/val)")


def main() -> None:
    p = argparse.ArgumentParser(description="Interactive label viewer")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--all",     action="store_true", help="review all, not just new")
    p.add_argument("--split",   nargs="+", default=["train", "val", "test"],
                   help="which splits to review (default: train val test)")
    args = p.parse_args()
    run(args.shuffle, args.all, tuple(args.split))


if __name__ == "__main__":
    main()
