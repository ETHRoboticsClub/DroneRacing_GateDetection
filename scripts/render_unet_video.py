"""Render consecutive frames through the LEGACY seg-only U-Net
(``best_gatenet.pth`` — the model deployed on zephyrus today).

Two reasons we need this video:
  1. The user asked: "run the same video with the unet checkpoint to
     see how it works" — i.e., visual sanity on what the deployed
     drone-side perception actually produces.
  2. It is the honest comparison point against the v2 keypoint
     model: the U-Net only emits a binary mask (no corner output),
     so the failure mode is qualitatively different — instead of
     'Frankenstein quads across gates' you see 'mask covers ALL
     visible gates at once'.

Layout (PANEL = 384 × 384):
    | Image | Mask overlay |
    plus a top bar:  "U-NET (seg-only, deployed on zephyrus)"

CLAUDE.md §0: this script does NOT do any corner extraction. The
zephyrus deployment uses ``quadgate.detect_corners`` post-hoc on the
mask via LSD line-segment fitting, which is itself a piece of legacy
code that has not been audited in this report. The video here is
purely "what does the U-Net output look like over time" — visible
inspection only, not a deployment benchmark.

Usage:
    python scripts/render_unet_video.py \\
        --checkpoint checkpoints/best_gatenet.pth \\
        --flight datasets/drone-racing-dataset/data/piloted/flight-06p-ellipse \\
        --start 3000 --count 500 --stride 2 --fps 15 \\
        --out docs/report/14_generalization_prep/unet_legacy_06p.mp4
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from gate_detection.inference.predictor import GatePredictor


PANEL = 384
BOTTOM_BAR = 28


def _resize_rgb(full_bgr: np.ndarray, size: int) -> np.ndarray:
    return cv2.cvtColor(cv2.resize(full_bgr, (size, size)), cv2.COLOR_BGR2RGB)


def _draw_title_bar(panel: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    box_h = 20 * len(lines) + 6
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (panel.shape[1], box_h), (0, 0, 0), -1)
    panel[:] = cv2.addWeighted(panel, 0.45, overlay, 0.55, 0)
    for i, (text, color) in enumerate(lines):
        cv2.putText(panel, text, (6, 16 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)


def _draw_mask_overlay(rgb: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """RGB frame with the seg probability heat-overlaid at 35% alpha."""
    seg_u8 = (np.clip(seg, 0, 1) * 255).astype(np.uint8)
    seg_color_bgr = cv2.applyColorMap(seg_u8, cv2.COLORMAP_HOT)
    seg_color_rgb = cv2.cvtColor(seg_color_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 0.65, seg_color_rgb, 0.35, 0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True,
                   help="Path to the LEGACY seg-only GateNet checkpoint "
                        "(no model_config in dict; no keypoint head). The "
                        "one deployed on zephyrus at "
                        "~/proj/ETHRC-Drone-Racing/src/perception/gate_detection/"
                        "checkpoints/best_gatenet.pth.")
    p.add_argument("--flight", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=500)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--out", required=True)
    p.add_argument("--img-size", type=int, default=PANEL)
    p.add_argument("--threshold", type=float, default=0.50,
                   help="Mask-binarisation threshold for the 'fired' count "
                        "(0.5 is the canonical seg threshold used in "
                        "eval.threshold of the YAML configs).")
    p.add_argument("--model-name", default="gatenet")
    p.add_argument("--width-factor", type=int, default=4)
    p.add_argument("--encoder-name", default="resnet34")
    args = p.parse_args()

    if args.fps <= 0 or args.count <= 0 or args.stride <= 0:
        raise ValueError("--fps / --count / --stride must all be > 0")

    flight_name = os.path.basename(args.flight.rstrip("/"))
    cam_subdir = next(
        d for d in os.listdir(args.flight) if d.startswith("camera_")
    )
    cam_dir = os.path.join(args.flight, cam_subdir)
    all_jpgs = sorted(f for f in os.listdir(cam_dir) if f.endswith(".jpg"))
    selected = all_jpgs[args.start : args.start + args.count * args.stride : args.stride]
    if not selected:
        raise RuntimeError("Frame selection produced 0 frames.")
    print(
        f"Rendering {len(selected)} frames from {cam_dir} "
        f"(stride={args.stride}, fps={args.fps}, "
        f"~{len(selected) / args.fps:.1f}s playback).",
        flush=True,
    )

    pred = GatePredictor(
        args.checkpoint,
        img_size=args.img_size,
        model_name=args.model_name,
        width_factor=args.width_factor,
        encoder_name=args.encoder_name,
        encoder_weights=None,
        dropout_p=0.0,
        keypoint_head=False,
        n_keypoints=4,
    )
    if pred.has_keypoint_head:
        print(
            "[WARN] checkpoint has a keypoint head; this script renders "
            "ONLY the seg channel anyway — for a quad-rendered video use "
            "render_consecutive_video.py instead.",
            flush=True,
        )
    print(
        f"Loaded {os.path.basename(args.checkpoint)}; "
        f"imagenet_norm={pred.imagenet_norm}",
        flush=True,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        args.out, cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps), (PANEL, PANEL + BOTTOM_BAR),
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"cv2.VideoWriter failed to open {args.out!r}; "
            f"install OpenCV with FFmpeg support."
        )

    fired = 0
    seg_maxes: list[float] = []
    seg_high_fracs: list[float] = []
    for i, fn in enumerate(selected):
        img_path = os.path.join(cam_dir, fn)
        rgb, seg = pred.predict_image(img_path)
        seg_max = float(seg.max())
        seg_high_frac = float((seg > args.threshold).mean())
        seg_maxes.append(seg_max)
        seg_high_fracs.append(seg_high_frac)
        if seg_max >= args.threshold:
            fired += 1

        panel = _draw_mask_overlay(rgb, seg)
        _draw_title_bar(panel, [
            (f"U-NET seg-only  frame {i + 1}/{len(selected)}", (255, 255, 255)),
            (f"seg max={seg_max:.2f}  >0.5 frac={seg_high_frac * 100:.1f}%"
             + ("" if seg_max >= args.threshold else "  [BELOW-THR]"),
             (255, 255, 255)),
        ])

        composite = np.zeros((PANEL + BOTTOM_BAR, PANEL, 3), dtype=np.uint8)
        composite[:PANEL, :] = panel
        caveat = "Legacy GateNet (3.3M params, no kp head) — deployed on zephyrus"
        cv2.rectangle(composite, (0, PANEL), (PANEL, PANEL + BOTTOM_BAR), (20, 20, 20), -1)
        cv2.putText(composite, caveat, (8, PANEL + 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)

        writer.write(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        if (i + 1) % 50 == 0:
            print(f"  rendered {i + 1}/{len(selected)}  fired={fired}", flush=True)

    writer.release()
    sz = os.path.getsize(args.out)
    print(f"saved -> {args.out} ({sz / 1e6:.2f} MB)", flush=True)
    print(
        f"summary: {fired}/{len(selected)} frames had seg_max >= {args.threshold} "
        f"({100 * fired / max(len(selected), 1):.1f}%)\n"
        f"  mean seg_max = {np.mean(seg_maxes):.3f}\n"
        f"  mean >0.5 fraction = {np.mean(seg_high_fracs) * 100:.2f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
