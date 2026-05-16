"""Render consecutive frames from a held-out flight as an MP4.

Goal — visual sanity for the two load-bearing questions on a flight the
v2 checkpoint has NEVER seen during training:

  1. Is the segmentation correct frame-by-frame?
  2. When multiple gates are visible, which one does the model pick?

The picked quad (red) sits ON TOP of any GT gate corners (green crosses)
so the user can directly compare "what the model chose" vs "what's in
the labels". The seg mask is overlaid as a hot colormap at 30 % alpha
so the underlying RGB is still readable.

Per CLAUDE.md §0 / report 13: there is NO inference-time "next gate"
selector. ``coherent_quad_decode`` picks the *most rectangular* visible
quad. The point of this video is to *see* whether that heuristic
agrees with where the drone should fly — not to claim it does.

Usage:
    python scripts/render_consecutive_video.py \
        --checkpoint checkpoints/best_deeplabv3plus_keypoint_multi_balanced_v2.pth \
        --flight datasets/drone-racing-dataset/data/piloted/flight-06p-ellipse \
        --start 0 --count 300 --stride 1 --fps 15 \
        --out docs/report/14_generalization_prep/consecutive_v2.mp4
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from gate_detection.data.labels import parse_gates
from gate_detection.inference.predictor import GatePredictor


def render_frame(
    img_rgb: np.ndarray,
    seg: np.ndarray,
    kp_xy: np.ndarray | None,
    kp_conf: np.ndarray | None,
    gt_gates: list[dict],
    threshold: float,
    frame_idx: int,
    total: int,
    fn: str,
    *,
    show_seg: bool = True,
    show_gt: bool = True,
) -> np.ndarray:
    """Return a (H, W, 3) BGR frame ready for cv2.VideoWriter."""
    out = img_rgb.copy()
    H = out.shape[0]

    # Seg overlay (hot colormap, 30 % alpha)
    if show_seg:
        seg_u8 = (np.clip(seg, 0, 1) * 255).astype(np.uint8)
        seg_color_bgr = cv2.applyColorMap(seg_u8, cv2.COLORMAP_HOT)
        seg_color_rgb = cv2.cvtColor(seg_color_bgr, cv2.COLOR_BGR2RGB)
        out = cv2.addWeighted(out, 0.70, seg_color_rgb, 0.30, 0)

    # GT gate corners (green crosses) — drawn first so the model's pick sits on top
    if show_gt:
        for g in gt_gates:
            ip = g["inner"] * H  # parse_gates returns normalised coords
            for k in range(4):
                if g["inner_vis"][k] > 0:
                    pt = (int(round(ip[k, 0])), int(round(ip[k, 1])))
                    cv2.drawMarker(out, pt, (0, 255, 0),
                                   markerType=cv2.MARKER_CROSS,
                                   markerSize=14, thickness=2)

    # Model's picked quad (red if confident, orange if below threshold)
    if kp_xy is not None and kp_conf is not None:
        conf_ok = float(kp_conf.min()) >= threshold
        color = (255, 0, 0) if conf_ok else (255, 165, 0)
        pts = kp_xy.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
        for k in range(kp_xy.shape[0]):
            cv2.circle(out, (int(kp_xy[k, 0]), int(kp_xy[k, 1])),
                       5, color, thickness=-1)
        conf_txt = (
            f"conf=[{float(kp_conf.min()):.2f}, "
            f"{float(kp_conf.max()):.2f}]"
            + ("" if conf_ok else "  BELOW-THR")
        )
    else:
        conf_txt = "no-det"

    # Header annotation strip — white text on a translucent black box.
    n_gt = len(gt_gates)
    header_lines = [
        f"{frame_idx + 1}/{total}  {fn}",
        f"GT gates: {n_gt}   {conf_txt}",
    ]
    box_h = 22 * len(header_lines) + 8
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (out.shape[1], box_h), (0, 0, 0), -1)
    out = cv2.addWeighted(out, 0.55, overlay, 0.45, 0)
    for i, line in enumerate(header_lines):
        cv2.putText(out, line, (8, 18 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    lineType=cv2.LINE_AA)

    # Legend at bottom-right (text adapts to what's visible)
    legend = "red quad = model pick"
    if show_gt:
        legend += "   green + = GT corners"
    (tw, th), _ = cv2.getTextSize(legend, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    pad = 6
    x0 = out.shape[1] - tw - pad * 2 - 6
    y0 = out.shape[0] - th - pad * 2 - 6
    legend_layer = out.copy()
    cv2.rectangle(legend_layer, (x0, y0),
                  (x0 + tw + pad * 2, y0 + th + pad * 2), (0, 0, 0), -1)
    out = cv2.addWeighted(out, 0.55, legend_layer, 0.45, 0)
    cv2.putText(out, legend, (x0 + pad, y0 + th + pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                lineType=cv2.LINE_AA)

    # VideoWriter wants BGR
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True,
                   help="Path to a *_keypoint_multi_*.pth ckpt.")
    p.add_argument("--flight", required=True,
                   help="Flight directory containing camera_* + label_* "
                        "subdirs (TII drone-racing-dataset layout).")
    p.add_argument("--start", type=int, default=0,
                   help="Index of the first frame to render (default 0).")
    p.add_argument("--count", type=int, default=300,
                   help="Number of frames to render (default 300).")
    p.add_argument("--stride", type=int, default=1,
                   help="Take every Nth frame from the sorted listing "
                        "(default 1). Source is 60 FPS — stride 6 + fps 10 "
                        "≈ 1× real-time playback.")
    p.add_argument("--fps", type=int, default=15,
                   help="Playback FPS for the encoded MP4 (default 15).")
    p.add_argument("--out", required=True,
                   help="Output MP4 path; parent created if needed.")
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--threshold", type=float, default=0.30,
                   help="Per-channel keypoint-conf threshold; below = "
                        "render the quad orange (not red).")
    p.add_argument("--model-name", default="deeplabv3plus")
    p.add_argument("--encoder-name", default="resnet34")
    p.add_argument("--dropout-p", type=float, default=0.1)
    p.add_argument("--no-seg", action="store_true",
                   help="Hide the segmentation-mask hot overlay. Use when "
                        "the only question is which gate was picked — the "
                        "underlying RGB then shows every gate clearly so "
                        "you can judge whether the model's pick is the one "
                        "the drone should fly through.")
    p.add_argument("--no-gt", action="store_true",
                   help="Hide the green GT-corner crosses. Pair with "
                        "--no-seg for a 'pick only' video.")
    args = p.parse_args()

    if not os.path.isdir(args.flight):
        raise FileNotFoundError(f"--flight not found: {args.flight}")

    # CLAUDE.md §0 — surface (not silence) the held-out-or-not question.
    flight_name = os.path.basename(args.flight.rstrip("/"))
    held_out_marker = ("06p-ellipse" in flight_name) or flight_name.endswith("a-lemniscate") or flight_name.endswith("a-ellipse") or flight_name.endswith("a-trackRATM")
    if not held_out_marker:
        print(
            f"[WARN] flight {flight_name!r} does not look held-out "
            f"(known held-out names: 06p-ellipse + 7× XXa-*). Continuing "
            f"— do NOT report numbers from a training flight.",
            flush=True,
        )

    cam_subdir = next(
        d for d in os.listdir(args.flight) if d.startswith("camera_")
    )
    lbl_subdir = next(
        d for d in os.listdir(args.flight) if d.startswith("label_")
    )
    cam_dir = os.path.join(args.flight, cam_subdir)
    lbl_dir = os.path.join(args.flight, lbl_subdir)

    all_jpgs = sorted(f for f in os.listdir(cam_dir) if f.endswith(".jpg"))
    selected = all_jpgs[args.start : args.start + args.count * args.stride : args.stride]
    if not selected:
        raise RuntimeError(
            f"Frame selection produced 0 frames; check --start={args.start} "
            f"vs total frames ({len(all_jpgs)}) and --count/--stride."
        )
    print(
        f"Rendering {len(selected)} frames from {cam_dir} "
        f"(stride={args.stride}, fps={args.fps}, ~{len(selected) / args.fps:.1f}s playback).",
        flush=True,
    )

    pred = GatePredictor(
        args.checkpoint, img_size=args.img_size,
        model_name=args.model_name, encoder_name=args.encoder_name,
        encoder_weights=None, dropout_p=args.dropout_p,
        keypoint_head=True, n_keypoints=4,
    )
    if not pred.has_keypoint_head:
        raise RuntimeError("ckpt has no keypoint head; can't render the picked quad.")
    print(
        f"Loaded {os.path.basename(args.checkpoint)}; "
        f"multi_gate_target={pred.multi_gate_target} "
        f"imagenet_norm={pred.imagenet_norm}",
        flush=True,
    )

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps), (args.img_size, args.img_size),
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"cv2.VideoWriter failed to open {out_path!r}. Codec mp4v "
            f"may be unavailable — install OpenCV with FFmpeg support."
        )

    fired = 0
    confs: list[float] = []
    gt_counts: list[int] = []
    for i, fn in enumerate(selected):
        img_path = os.path.join(cam_dir, fn)
        lp = os.path.join(lbl_dir, fn.replace(".jpg", ".txt"))
        gt_gates: list[dict] = (
            parse_gates(lp)
            if (os.path.exists(lp) and os.path.getsize(lp) > 0)
            else []
        )
        rgb, seg, kp_xy, kp_conf = pred.predict_keypoints(img_path)
        if kp_xy is not None and kp_conf is not None:
            confs.append(float(kp_conf.min()))
            if float(kp_conf.min()) >= args.threshold:
                fired += 1
        gt_counts.append(len(gt_gates))
        frame = render_frame(
            rgb, seg, kp_xy, kp_conf, gt_gates,
            args.threshold, i, len(selected), fn,
            show_seg=not args.no_seg,
            show_gt=not args.no_gt,
        )
        writer.write(frame)
        if (i + 1) % 50 == 0:
            print(f"  rendered {i + 1}/{len(selected)}", flush=True)

    writer.release()
    sz = os.path.getsize(out_path)
    print(f"saved -> {out_path} ({sz / 1e6:.2f} MB)", flush=True)
    print(
        f"summary: {fired}/{len(selected)} frames fired "
        f"(conf_min ≥ {args.threshold:.2f}); "
        f"mean conf_min = {np.mean(confs) if confs else 0:.3f}; "
        f"GT gates per frame: min={min(gt_counts)} max={max(gt_counts)} "
        f"mean={np.mean(gt_counts):.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
