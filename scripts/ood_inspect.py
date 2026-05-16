"""OOD inspection — render predictions on a folder of UNLABELLED images.

Goal (report 14, Tier 2.4): before competition data arrives, eyeball how
the current checkpoint behaves on out-of-distribution footage that was
NEVER seen in training or validation. Examples: YouTube drone-racing
clips, AlphaPilot 2019 footage, RPMRC datasets, or any directory of
arbitrary images of gates.

This script is intentionally minimal: load a ckpt, walk a directory,
run inference, save a grid of frame predictions and a per-frame CSV
of peak confidence. No metrics are computed because there are no labels —
the load-bearing question is *visual sanity*. The user then decides
whether the model fires on the right thing.

Usage:
    python scripts/ood_inspect.py \
        --checkpoint checkpoints/best_deeplabv3plus_keypoint_multi_balanced_v2.pth \
        --image-dir /path/to/some_ood_folder \
        --out docs/report/14_generalization_prep/ood_inspect_v2.png

CLAUDE.md §0 note: this is a sanity tool, NOT a benchmark. Detection
"works" by eye on N=24 frames means nothing statistically. Use this to
catch catastrophic failure modes (model fires on everything, model fires
on nothing, model fires on the wrong region of the frame) before
investing in a labelled OOD eval.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gate_detection.inference.predictor import GatePredictor


def list_images(image_dir: str, max_images: int | None) -> list[str]:
    """Sorted list of .jpg/.png/.jpeg files in ``image_dir``, capped at
    ``max_images`` (None = no cap)."""
    exts = (".jpg", ".jpeg", ".png")
    paths = sorted(
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(exts)
    )
    if max_images is not None:
        paths = paths[:max_images]
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True,
                   help="Path to a *_keypoint_multi_*.pth ckpt (model_config "
                        "must have keypoint_head=True for the keypoint path).")
    p.add_argument("--image-dir", required=True,
                   help="Directory of UNLABELLED images. The script does not "
                        "look for sibling .txt label files — by design.")
    p.add_argument("--out", default="ood_inspect.png",
                   help="Output grid PNG path.")
    p.add_argument("--csv", default=None,
                   help="Optional CSV path; one row per image with peak "
                        "confidence stats. Helps quantify aggregate "
                        "behaviour without labels.")
    p.add_argument("--max-images", type=int, default=24,
                   help="Cap the number of images rendered (grid gets "
                        "unreadable past ~36). Default 24.")
    p.add_argument("--grid-cols", type=int, default=4,
                   help="Columns in the output grid. Rows derived from "
                        "len(images) / cols.")
    p.add_argument("--img-size", type=int, default=384,
                   help="Inference resolution. Must match the ckpt's "
                        "training resolution (default 384).")
    p.add_argument("--model-name", default="deeplabv3plus")
    p.add_argument("--encoder-name", default="resnet34")
    p.add_argument("--dropout-p", type=float, default=0.1)
    p.add_argument("--threshold", type=float, default=0.30,
                   help="Per-channel KEYPOINT-confidence threshold below "
                        "which a frame is rendered as 'no detection'. "
                        "Distinct from the segmentation mask threshold "
                        "(``eval.threshold: 0.5`` in the YAML configs). "
                        "Matches the keypoint threshold used in "
                        "benchmark.py and fp_inspect.py.")
    args = p.parse_args()

    if not os.path.isdir(args.image_dir):
        raise FileNotFoundError(f"image-dir does not exist: {args.image_dir}")

    paths = list_images(args.image_dir, max_images=args.max_images)
    if not paths:
        raise RuntimeError(
            f"No .jpg/.jpeg/.png images found in {args.image_dir}; "
            f"nothing to inspect."
        )
    print(f"Found {len(paths)} images (capped at --max-images={args.max_images}).")

    # Predictor — pulls model + data config off the ckpt itself.
    pred = GatePredictor(
        args.checkpoint,
        img_size=args.img_size,
        model_name=args.model_name,
        encoder_name=args.encoder_name,
        encoder_weights=None,
        dropout_p=args.dropout_p,
        keypoint_head=True,
        n_keypoints=4,
    )
    if not pred.has_keypoint_head:
        raise RuntimeError(
            "ckpt has no keypoint_head; OOD inspect only renders the "
            "keypoint-quad path. Use a *_keypoint_multi_*.pth checkpoint."
        )
    print(
        f"Loaded {os.path.basename(args.checkpoint)}; "
        f"multi_gate_target={pred.multi_gate_target} "
        f"imagenet_norm={pred.imagenet_norm}"
    )

    # Inference loop. Collect per-image results.
    results: list[dict] = []
    for path in paths:
        rgb, seg, kp_xy, kp_conf = pred.predict_keypoints(path)
        if kp_xy is None or kp_conf is None:
            cm = 0.0
            cmax = float(seg.max())
            no_det = True
        else:
            cm = float(kp_conf.min())
            cmax = float(kp_conf.max())
            no_det = cm < args.threshold
        results.append({
            "path": path,
            "rgb": rgb,
            "seg": seg,
            "kp_xy": kp_xy,
            "kp_conf": kp_conf,
            "conf_min": cm,
            "conf_max": cmax,
            "seg_max": float(seg.max()),
            "seg_high_frac": float((seg > 0.5).mean()),
            "no_det": no_det,
        })

    # Grid render.
    n = len(results)
    cols = max(1, args.grid_cols)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.0 * rows))
    if rows * cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(
        f"OOD inspection — {os.path.basename(args.checkpoint)}\n"
        f"image-dir: {args.image_dir} (capped at {args.max_images}, "
        f"threshold {args.threshold:.2f})\n"
        f"Red quad = predicted gate. Title shows per-channel conf [min, max] "
        f"and seg max / high-frac. NO LABELS — visual sanity only.",
        fontsize=11,
    )

    for i, r in enumerate(results):
        ax = axes[i // cols, i % cols]
        ax.imshow(r["rgb"])
        ax.imshow(r["seg"], cmap="hot", alpha=0.30, vmin=0, vmax=1)
        if not r["no_det"] and r["kp_xy"] is not None:
            quad = np.vstack([r["kp_xy"], r["kp_xy"][0:1]])
            ax.plot(quad[:, 0], quad[:, 1], "-", color="red",
                    linewidth=2, alpha=0.85)
            for k in range(r["kp_xy"].shape[0]):
                ax.plot(r["kp_xy"][k, 0], r["kp_xy"][k, 1], "o",
                        color="red", markersize=7, markeredgewidth=0)
        fn = os.path.basename(r["path"])
        title = (
            f"{fn[:34]}\n"
            f"conf=[{r['conf_min']:.2f}, {r['conf_max']:.2f}] "
            f"seg max={r['seg_max']:.2f} hi={r['seg_high_frac']*100:.1f}%"
        )
        if r["no_det"]:
            title += "  [NO-DET]"
        ax.set_title(title, fontsize=8,
                     color="red" if r["no_det"] else "black")
        ax.axis("off")

    # Hide unused axes
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"saved -> {out_path} ({os.path.getsize(out_path)} bytes)")

    # Per-frame CSV (optional).
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "path", "conf_min", "conf_max",
                "seg_max", "seg_high_frac", "no_det",
            ])
            for r in results:
                w.writerow([
                    r["path"], f"{r['conf_min']:.4f}", f"{r['conf_max']:.4f}",
                    f"{r['seg_max']:.4f}", f"{r['seg_high_frac']:.4f}",
                    int(r["no_det"]),
                ])
        print(f"csv   -> {csv_path}")

    # Aggregate summary on stdout — useful for grepping a sanity number
    # out of a batch run without opening the PNG.
    fired = sum(1 for r in results if not r["no_det"])
    print(
        f"summary: {fired}/{n} frames fired (conf_min ≥ {args.threshold:.2f}); "
        f"mean conf_min = {np.mean([r['conf_min'] for r in results]):.3f}, "
        f"mean seg high-frac = "
        f"{np.mean([r['seg_high_frac'] for r in results]) * 100:.1f}%"
    )


if __name__ == "__main__":
    main()
