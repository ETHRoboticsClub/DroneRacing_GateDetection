"""Visualize the 4 FP fires from report 12's held-out 06p FP-rate test.

For each fire (a "no-gate" frame where the model fired despite the
labeller writing no .txt), show: input, predicted quad, GT corners
of ANY visible gate from sibling frames (in case there IS a gate
the labeller missed). The classification (true FP vs labeller-skipped
gate) is the load-bearing question.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gate_detection.inference.predictor import GatePredictor
from gate_detection.data.labels import parse_gates


def find_fp_fires(predictor, flight_dir, threshold=0.3, stride=10):
    """Replicate report-12's fp_rate.py: scan no-gate frames, return list
    of (img_path, kp_xy, kp_conf, conf_min) for the ones that fired."""
    cam_dir = os.path.join(flight_dir, [d for d in os.listdir(flight_dir) if d.startswith("camera_")][0])
    lbl_dir = os.path.join(flight_dir, [d for d in os.listdir(flight_dir) if d.startswith("label_")][0])
    out = []
    no_gate_frames = []
    for fn in sorted(os.listdir(cam_dir)):
        if not fn.endswith(".jpg"):
            continue
        lp = os.path.join(lbl_dir, fn.replace(".jpg", ".txt"))
        is_no_gate = (
            not os.path.exists(lp)
            or os.path.getsize(lp) == 0
            or not parse_gates(lp)
        )
        if is_no_gate:
            no_gate_frames.append(fn)
    print(f"  no-gate frames total: {len(no_gate_frames)}")
    sampled = no_gate_frames[::stride]
    print(f"  sampled at stride {stride}: {len(sampled)}")
    for fn in sampled:
        img_path = os.path.join(cam_dir, fn)
        rgb, seg, kp_xy, kp_conf = predictor.predict_keypoints(img_path)
        if kp_xy is None or kp_conf is None:
            continue
        cm = float(kp_conf.min())
        if cm >= threshold:
            out.append((img_path, kp_xy, kp_conf, cm, rgb, seg))
    return out, cam_dir, lbl_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",
                   default="checkpoints/best_deeplabv3plus_keypoint_multi.pth")
    p.add_argument("--flight",
                   default="datasets/drone-racing-dataset/data/piloted/flight-06p-ellipse")
    p.add_argument("--out", default="/tmp/fp_inspect_06p.png")
    args = p.parse_args()

    if "flight-06p-ellipse" not in args.flight:
        raise RuntimeError("Only held-out 06p-ellipse permitted (data hygiene)")

    pred = GatePredictor(
        args.checkpoint, img_size=384,
        model_name="deeplabv3plus", encoder_name="resnet34",
        encoder_weights=None, dropout_p=0.1,
        keypoint_head=True, n_keypoints=4,
    )
    print(f"Loaded {os.path.basename(args.checkpoint)}; multi_gate_target={pred.multi_gate_target}")

    fires, cam_dir, lbl_dir = find_fp_fires(pred, args.flight, threshold=0.3, stride=10)
    print(f"Found {len(fires)} FP fires (matches report 12's 4/166)")

    n = len(fires)
    if n == 0:
        print("No FP fires to display.")
        return

    # 2-row grid: top = the fire frame, bottom = ±5-frame neighbour
    # context (helps identify a labeller-skipped gate that's there
    # but not labelled).
    rows = 2
    cols = max(n, 1)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    fig.suptitle(
        "Held-out 06p-ellipse FP fires (multi-gate ckpt, threshold 0.3)\n"
        "Top row: the 'fired' no-gate frame + predicted quad (red). "
        "Bottom row: a labelled-gate neighbour frame ±10 frames away (showing what gates look like in this scene).\n"
        "Classification per frame in title.",
        fontsize=12,
    )

    all_imgs = sorted(f for f in os.listdir(cam_dir) if f.endswith(".jpg"))

    for col, (img_path, kp_xy, kp_conf, cm, rgb, seg) in enumerate(fires):
        # Top: the fire frame
        ax = axes[0, col]
        ax.imshow(rgb)
        ax.imshow(seg, cmap="hot", alpha=0.30, vmin=0, vmax=1)
        quad = np.vstack([kp_xy, kp_xy[0:1]])
        ax.plot(quad[:, 0], quad[:, 1], "-", color="red", linewidth=2, alpha=0.85)
        for k in range(4):
            ax.plot(kp_xy[k, 0], kp_xy[k, 1], "o", color="red",
                    markersize=9, markeredgewidth=0)
        fn = os.path.basename(img_path)
        seg_max = float(seg.max())
        seg_frac_high = float((seg > 0.5).mean())
        ax.set_title(
            f"{fn}\nconf=[{kp_conf.min():.2f}, {kp_conf.max():.2f}]   "
            f"seg max={seg_max:.2f} (>0.5 frac={seg_frac_high*100:.1f}%)",
            fontsize=9,
        )
        ax.axis("off")

        # Bottom: a labelled-gate neighbour frame nearby
        idx = all_imgs.index(fn) if fn in all_imgs else None
        neighbour = None
        if idx is not None:
            for offset in (10, -10, 20, -20, 30, -30, 50, -50):
                j = idx + offset
                if 0 <= j < len(all_imgs):
                    cand = all_imgs[j]
                    lp = os.path.join(lbl_dir, cand.replace(".jpg", ".txt"))
                    if os.path.exists(lp) and os.path.getsize(lp) > 0:
                        gates = parse_gates(lp)
                        if gates:
                            neighbour = (cand, gates, offset)
                            break

        ax2 = axes[1, col]
        if neighbour:
            ncand, ngates, noffset = neighbour
            import cv2
            nimg_bgr = cv2.imread(os.path.join(cam_dir, ncand))
            if nimg_bgr is None:
                # CLAUDE.md §6 — announce the fallback path rather than
                # crash on a corrupt jpg downstream.
                print(
                    f"[WARN] fp_inspect: cv2.imread failed for {ncand}; "
                    f"rendering blank neighbour cell",
                    flush=True,
                )
                ax2.set_title(f"neighbour {ncand} read failed", fontsize=9, color="red")
                ax2.axis("off")
                continue
            nrgb = cv2.cvtColor(nimg_bgr, cv2.COLOR_BGR2RGB)
            nrgb = cv2.resize(nrgb, (384, 384))
            ax2.imshow(nrgb)
            for g in ngates:
                ip = g["inner"] * 384
                for k in range(4):
                    if g["inner_vis"][k] > 0:
                        ax2.plot(ip[k, 0], ip[k, 1], "g+", markersize=12, markeredgewidth=2)
            ax2.set_title(f"neighbour {ncand} ({noffset:+d} frames, {len(ngates)} GT gates)", fontsize=9)
        else:
            ax2.set_title("no labelled neighbour ±50 frames", fontsize=9, color="red")
        ax2.axis("off")

    plt.tight_layout()
    plt.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"saved -> {args.out} ({os.path.getsize(args.out)} bytes)")


if __name__ == "__main__":
    main()
