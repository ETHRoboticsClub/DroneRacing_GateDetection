"""Three-way comparison video: baseline vs Path A (Vicon-cropped,
OFFLINE-ONLY proof-of-concept) vs Path B (track-by-detection with
quality gate, DEPLOYABLE).

Report 14 / "fixed trajectory" video. Renders the same 500 consecutive
frames from held-out flight-06p-ellipse three times, side-by-side, so
the failure-mode and the candidate fixes can be compared in one view.

Layout (each panel 384 × 384):
    | BASELINE | PATH A | PATH B |
    --------- ---------- ---------
    |  red    |  red   |  red   |   red quad = picked gate
    |  +stats |  +stats|  +stats|

Bottom strip: "Path A uses GROUND-TRUTH Vicon pose; for flight, swap
in VIO+map per MonoRace 2026."

CLAUDE.md §0 — lives depend on this. The script:
  * NEVER falls back silently. If Vicon pose is unavailable for a
    frame, Path A renders "no Vicon" in red and the panel skips the
    crop (uses the full-frame baseline instead).
  * Path A panel title is ALWAYS prefixed with [OFFLINE].
  * Path B panel title shows the tracker's per-frame status (ok,
    fresh_pick, fail_drop, handoff, etc.) so the reviewer / pilot
    can see *why* a frame was a non-detection vs why it was held.

Usage:
    python scripts/render_three_way_comparison.py \\
        --checkpoint checkpoints/best_deeplabv3plus_keypoint_multi_balanced_v2.pth \\
        --flight datasets/drone-racing-dataset/data/piloted/flight-06p-ellipse \\
        --calib-dir datasets/drone-racing-dataset/calibration \\
        --start 3000 --count 500 --stride 2 --fps 15 \\
        --out docs/report/14_generalization_prep/three_way_v2_06p_3000-4000.mp4
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from gate_detection.evaluation.pnp import (
    load_calibration,
    project_next_gate_uv,
)
from gate_detection.inference.predictor import GatePredictor
from gate_detection.inference.tracker import GateTracker, GateTrackerConfig
from gate_detection.inference.seg_cc_tracker import SegCCTracker, SegCCTrackerConfig
from gate_detection.inference.next_gate_tracker import (
    NextGateMaskTracker, NextGateMaskTrackerConfig,
)
from gate_detection.inference.hybrid_tracker import (
    HybridSegNextGateTracker, HybridTrackerConfig,
)
from gate_detection.losses.keypoint import coherent_quad_decode


# Display geometry
PANEL = 384
W3 = PANEL * 3
H = PANEL
BOTTOM_BAR = 28  # extra row at the bottom for the global caveat


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _draw_quad(ax: np.ndarray, quad: Optional[np.ndarray], color, thick: int = 2) -> None:
    if quad is None:
        return
    pts = quad.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(ax, [pts], isClosed=True, color=color, thickness=thick)
    for k in range(quad.shape[0]):
        cv2.circle(ax, (int(quad[k, 0]), int(quad[k, 1])), 5, color, -1)


def _draw_title_bar(panel: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    """Top translucent strip on a panel with up to 3 lines of text."""
    box_h = 20 * len(lines) + 6
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (panel.shape[1], box_h), (0, 0, 0), -1)
    panel[:] = cv2.addWeighted(panel, 0.45, overlay, 0.55, 0)
    for i, (text, color) in enumerate(lines):
        cv2.putText(panel, text, (6, 16 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)


def _resize_rgb(full_bgr: np.ndarray, size: int) -> np.ndarray:
    return cv2.cvtColor(cv2.resize(full_bgr, (size, size)), cv2.COLOR_BGR2RGB)


def _map_crop_to_display(
    corners_local: np.ndarray,
    crop_x0: int, crop_y0: int,
    full_W: int, full_H: int,
    display: int = PANEL,
) -> np.ndarray:
    """Map (K, 2) corners from crop-local pixels → display-panel pixels.

    The display panel is the FULL image resized (non-aspect-preserving)
    to ``display`` × ``display``, matching the baseline panel so the
    visual frame of reference is consistent across all three panels.
    """
    full_corners = corners_local + np.array([crop_x0, crop_y0], dtype=corners_local.dtype)
    sx = display / float(full_W)
    sy = display / float(full_H)
    return full_corners * np.array([sx, sy], dtype=corners_local.dtype)


# ---------------------------------------------------------------------------
# Path A — adaptive crop driven by Vicon-projected next-gate centre
# ---------------------------------------------------------------------------
def _path_a_predict(
    predictor: GatePredictor,
    full_bgr: np.ndarray,
    row: Optional[pd.Series],
    K: Optional[np.ndarray],
    R_cam_body: Optional[np.ndarray],
    t_cam_body: Optional[np.ndarray],
    crop_size: int = PANEL,
) -> tuple[Optional[np.ndarray], str, Optional[tuple[int, int]]]:
    """Run the Vicon-prior adaptive-crop path on one frame.

    Returns ``(quad_in_display_coords_or_None, status_str, (x0, y0)_or_None)``.
    The ``status_str`` is a short tag for the panel label so the viewer
    can see why a given frame has no quad: ``ok``, ``no_vicon``,
    ``no_gate_visible``, ``no_decode``.
    """
    if row is None or K is None or R_cam_body is None or t_cam_body is None:
        return None, "no_vicon", None
    proj = project_next_gate_uv(row, K, R_cam_body, t_cam_body)
    if proj is None:
        return None, "no_gate_visible", None
    _gid, u, v = proj
    full_H, full_W = full_bgr.shape[:2]
    if full_W < crop_size or full_H < crop_size:
        # Image smaller than crop — degenerate. Don't try Path A.
        return None, "img_too_small", None
    # CRITICAL FIX (reviewer #3): ``project_next_gate_uv`` filters
    # gates BEHIND the camera (g_cam.z <= 0) but NOT gates BESIDE the
    # camera where (u, v) is outside the image bounds. Without this
    # check the crop below would silently clamp and the model would
    # run on a region that does NOT contain the gate, yet Path A's
    # status would still read "ok" with a meaningless quad. That
    # would invalidate the upper-bound experiment (Path A's whole
    # point). Reject those frames explicitly per CLAUDE.md §6.
    if not (0 <= u < full_W and 0 <= v < full_H):
        return None, "off_frame", None
    # Center crop on (u, v), then clip to bounds. The clip preserves
    # the 384 × 384 size by shifting inward, so the model sees a
    # consistent input scale.
    x0 = int(round(u - crop_size / 2))
    y0 = int(round(v - crop_size / 2))
    x0 = max(0, min(full_W - crop_size, x0))
    y0 = max(0, min(full_H - crop_size, y0))
    crop_bgr = full_bgr[y0:y0 + crop_size, x0:x0 + crop_size]
    if crop_bgr.shape[:2] != (crop_size, crop_size):
        # CLAUDE.md §6 — surface the fallback rather than silently
        # returning a degenerate crop.
        print(
            f"[WARN] Path A: clipped crop shape {crop_bgr.shape[:2]} != "
            f"({crop_size}, {crop_size}); skipping this frame.",
            flush=True,
        )
        return None, "crop_clip_fail", None
    _rgb, _seg, heatmap = predictor.predict_keypoint_heatmap_from_bgr(crop_bgr)
    if heatmap is None:
        return None, "no_kp_head", (x0, y0)
    quad_local = coherent_quad_decode(heatmap)
    if quad_local is None:
        return None, "no_decode", (x0, y0)
    quad_display = _map_crop_to_display(
        quad_local, x0, y0, full_W, full_H, display=crop_size,
    )
    return quad_display, "ok", (x0, y0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--flight", required=True,
                   help="Held-out flight dir with camera_* + label_* "
                        "subdirs and a *_cam_ts_sync.csv.")
    p.add_argument("--calib-dir", default=None,
                   help="Directory with calib_*.json + drone_to_camera.json. "
                        "If omitted, Path A is disabled and the renderer "
                        "outputs only Baseline + Path B in those panel slots.")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=500)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--out", required=True)
    p.add_argument("--img-size", type=int, default=PANEL)
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--model-name", default="deeplabv3plus")
    p.add_argument("--encoder-name", default="resnet34")
    p.add_argument("--dropout-p", type=float, default=0.1)
    # Tracker knobs — see GateTrackerConfig docstrings.
    p.add_argument("--rect-thr", type=float, default=80.0,
                   help="Tracker rect_threshold (max acceptable score).")
    p.add_argument("--prox-weight", type=float, default=0.5)
    p.add_argument("--max-fail-streak", type=int, default=3)
    p.add_argument("--handoff-diag", type=float, default=280.0)
    p.add_argument("--handoff-persist", type=int, default=2)
    p.add_argument("--pathb-mode",
                   choices=["kp", "seg_cc", "next_gate", "hybrid"],
                   default="hybrid",
                   help="Which tracker variant powers the Path B panel. "
                        "'kp' = strict-lock on keypoint quads. "
                        "'seg_cc' = largest seg-CC selection (option #2). "
                        "'next_gate' = next-gate-head mask as ROI (option #1). "
                        "'hybrid' (default) = seg-CC coverage + next-gate-"
                        "head selection (report 15.3 — uses both heads).")
    args = p.parse_args()

    # Reviewer #4 / CLAUDE.md §6: explicit guards before the writer
    # silently fails with a misleading output.
    if args.fps <= 0:
        raise ValueError(f"--fps must be > 0; got {args.fps}")
    if args.count <= 0:
        raise ValueError(f"--count must be > 0; got {args.count}")
    if args.stride <= 0:
        raise ValueError(f"--stride must be > 0; got {args.stride}")

    flight_name = os.path.basename(args.flight.rstrip("/"))
    held_out = ("06p-ellipse" in flight_name) or flight_name.endswith("a-lemniscate") \
        or flight_name.endswith("a-ellipse") or flight_name.endswith("a-trackRATM")
    if not held_out:
        print(
            f"[WARN] flight {flight_name!r} does not look held-out; "
            f"continuing but do NOT report numbers from a training flight.",
            flush=True,
        )

    # ---------- Sources --------------------------------------------------
    cam_subdir = next(d for d in os.listdir(args.flight) if d.startswith("camera_"))
    cam_dir = os.path.join(args.flight, cam_subdir)
    csv_path = os.path.join(args.flight, f"{flight_name}_cam_ts_sync.csv")
    csv_df: Optional[pd.DataFrame] = None
    if os.path.isfile(csv_path):
        csv_df = pd.read_csv(csv_path, index_col="img_filename")
        print(f"Loaded CSV: {csv_path} ({len(csv_df)} rows)", flush=True)
    else:
        print(
            f"[WARN] no cam_ts_sync CSV at {csv_path}; Path A will render "
            f"'no_vicon' for every frame.",
            flush=True,
        )

    K = dist = R_cam_body = t_cam_body = None
    if args.calib_dir is not None:
        try:
            K, dist, R_cam_body, t_cam_body = load_calibration(flight_name, args.calib_dir)
            print(f"Loaded calibration from {args.calib_dir}; K[0,0]={K[0,0]:.1f}", flush=True)
        except Exception as e:
            print(
                f"[WARN] load_calibration failed ({e!r}); Path A disabled.",
                flush=True,
            )
            K = R_cam_body = t_cam_body = None
    else:
        print(
            "[WARN] --calib-dir not provided; Path A disabled "
            "(panel will show 'no_vicon' for every frame).",
            flush=True,
        )

    all_jpgs = sorted(f for f in os.listdir(cam_dir) if f.endswith(".jpg"))
    selected = all_jpgs[args.start : args.start + args.count * args.stride : args.stride]
    if not selected:
        raise RuntimeError(
            f"Frame selection produced 0 frames; check --start / --count / "
            f"--stride against total frames ({len(all_jpgs)})."
        )
    print(
        f"Rendering {len(selected)} frames from {cam_dir} "
        f"(stride={args.stride}, fps={args.fps}, "
        f"~{len(selected) / args.fps:.1f}s playback).",
        flush=True,
    )

    # ---------- Model + Tracker -----------------------------------------
    predictor = GatePredictor(
        args.checkpoint, img_size=args.img_size,
        model_name=args.model_name, encoder_name=args.encoder_name,
        encoder_weights=None, dropout_p=args.dropout_p,
        keypoint_head=True, n_keypoints=4,
    )
    if not predictor.has_keypoint_head:
        raise RuntimeError(
            "ckpt has no keypoint head; 3-way comparison needs one."
        )
    print(
        f"Loaded {os.path.basename(args.checkpoint)}; "
        f"multi_gate_target={predictor.multi_gate_target} "
        f"imagenet_norm={predictor.imagenet_norm}",
        flush=True,
    )

    if args.pathb_mode == "kp":
        tracker_cfg = GateTrackerConfig(
            conf_threshold=args.threshold,
            rect_threshold=args.rect_thr,
            proximity_weight=args.prox_weight,
            max_fail_streak=args.max_fail_streak,
            handoff_diagonal_threshold=args.handoff_diag,
            handoff_persist_frames=args.handoff_persist,
        )
        tracker = GateTracker(tracker_cfg)
    elif args.pathb_mode == "seg_cc":
        seg_tracker_cfg = SegCCTrackerConfig(
            conf_threshold=args.threshold,
            max_fail_streak=args.max_fail_streak,
            handoff_persist_frames=args.handoff_persist,
        )
        tracker = SegCCTracker(seg_tracker_cfg)
    elif args.pathb_mode == "next_gate":
        ng_tracker_cfg = NextGateMaskTrackerConfig(
            conf_threshold=args.threshold,
            max_fail_streak=args.max_fail_streak,
            handoff_persist_frames=args.handoff_persist,
        )
        tracker = NextGateMaskTracker(ng_tracker_cfg)
    else:
        # hybrid: seg-CC for coverage, next-gate-head for selection
        # (report 15.3). Best of both — requires a ckpt with
        # next_gate_head trained AND the seg head's general mask.
        hybrid_cfg = HybridTrackerConfig(
            conf_threshold=args.threshold,
            max_fail_streak=args.max_fail_streak,
            handoff_persist_frames=args.handoff_persist,
        )
        tracker = HybridSegNextGateTracker(hybrid_cfg)
    print(f"Path B tracker mode: {args.pathb_mode}", flush=True)

    # ---------- Video writer --------------------------------------------
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_size = (W3, H + BOTTOM_BAR)
    writer = cv2.VideoWriter(
        args.out, cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps), out_size,
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"cv2.VideoWriter failed to open {args.out!r}; "
            f"install OpenCV with FFmpeg support."
        )

    # ---------- Tallies -------------------------------------------------
    counts = {
        "baseline_fired": 0, "patha_fired": 0, "pathb_fired": 0,
        "patha_no_vicon": 0, "patha_no_gate_visible": 0, "patha_no_decode": 0,
        "patha_off_frame": 0,
        "pathb_handoffs": 0, "pathb_fail_drops": 0,
        "csv_missing_row": 0,
    }

    # ---------- Frame loop ----------------------------------------------
    for i, fn in enumerate(selected):
        img_path = os.path.join(cam_dir, fn)
        full_bgr = cv2.imread(img_path)
        if full_bgr is None:
            print(f"[WARN] cv2.imread failed for {fn}; skipping.", flush=True)
            continue
        full_H, full_W = full_bgr.shape[:2]
        rgb_display = _resize_rgb(full_bgr, PANEL)

        # One model forward on the full-frame-resized — shared by
        # baseline and Path B. predict_full additionally returns the
        # next-gate-mask head's output (None on legacy ckpts), which
        # the next_gate tracker mode consumes.
        _rgb, seg_full, heatmap_full, next_gate_full = predictor.predict_full(img_path)

        # Baseline: coherent_quad_decode on the same heatmap.
        baseline_quad = None
        baseline_conf = 0.0
        if heatmap_full is not None:
            per_channel_max = heatmap_full.reshape(heatmap_full.shape[0], -1).max(axis=-1)
            baseline_conf = float(per_channel_max.min())
            if baseline_conf >= args.threshold:
                baseline_quad = coherent_quad_decode(heatmap_full)
                if baseline_quad is not None:
                    counts["baseline_fired"] += 1

        # Path B: tracker on the model output. In 'kp' mode the
        # GateTracker takes the keypoint heatmap; in 'seg_cc' mode the
        # SegCCTracker takes (seg, kp_heatmap) — the seg drives CC
        # selection, the kp heatmap localises corners within the CC.
        pathb_quad = None
        pathb_status = None
        if heatmap_full is not None:
            if args.pathb_mode == "kp":
                pathb_quad, pathb_status = tracker.step(heatmap_full)
            elif args.pathb_mode == "seg_cc":
                pathb_quad, pathb_status = tracker.step(seg_full, heatmap_full)
            elif args.pathb_mode == "next_gate":
                # next_gate: requires the new head's output
                if next_gate_full is None:
                    print(
                        f"[WARN] frame {fn}: --pathb-mode=next_gate but the "
                        f"ckpt has no next-gate head; emitting no detection.",
                        flush=True,
                    )
                    pathb_quad, pathb_status = None, None
                else:
                    pathb_quad, pathb_status = tracker.step(
                        next_gate_full, heatmap_full,
                    )
            else:
                # hybrid: pass seg + ng + kp. Tracker handles ng=None
                # fallback to area-based selection internally.
                pathb_quad, pathb_status = tracker.step(
                    seg_full, next_gate_full, heatmap_full,
                )
            if pathb_quad is not None:
                counts["pathb_fired"] += 1
            if pathb_status is not None and pathb_status.handoff_triggered:
                counts["pathb_handoffs"] += 1
            if pathb_status is not None and pathb_status.reason == "fail_drop":
                counts["pathb_fail_drops"] += 1

        # Path A: Vicon-cropped pipeline (extra forward).
        row = None
        if csv_df is not None:
            if fn in csv_df.index:
                row = csv_df.loc[fn]
            else:
                # CLAUDE.md §6 — surface the missing CSV row rather
                # than silently rendering "no_vicon".
                counts["csv_missing_row"] += 1
                if counts["csv_missing_row"] <= 3:
                    print(
                        f"[WARN] frame {fn!r} not in {flight_name}_cam_ts_sync.csv "
                        f"index; Path A panel will render 'no_vicon'.",
                        flush=True,
                    )
                elif counts["csv_missing_row"] == 4:
                    print(
                        "[WARN] further CSV-miss warnings suppressed; "
                        "final count in summary.",
                        flush=True,
                    )
        patha_quad, patha_status, _crop_origin = _path_a_predict(
            predictor, full_bgr, row, K, R_cam_body, t_cam_body, crop_size=PANEL,
        )
        if patha_status == "ok" and patha_quad is not None:
            counts["patha_fired"] += 1
        elif patha_status == "no_vicon":
            counts["patha_no_vicon"] += 1
        elif patha_status == "no_gate_visible":
            counts["patha_no_gate_visible"] += 1
        elif patha_status == "no_decode":
            counts["patha_no_decode"] += 1
        elif patha_status == "off_frame":
            counts["patha_off_frame"] += 1

        # ---- Compose the three panels --------------------------------
        # All three panels share the same underlying RGB so the viewer's
        # eye doesn't have to re-anchor between panels.
        p_baseline = rgb_display.copy()
        p_a = rgb_display.copy()
        p_b = rgb_display.copy()

        _draw_quad(p_baseline, baseline_quad, color=(255, 0, 0))
        _draw_quad(p_a, patha_quad, color=(0, 200, 255))
        _draw_quad(p_b, pathb_quad, color=(0, 255, 100))

        _draw_title_bar(p_baseline, [
            (f"BASELINE  frame {i + 1}/{len(selected)}", (255, 255, 255)),
            (f"conf_min={baseline_conf:.2f}  "
             f"{'FIRED' if baseline_quad is not None else 'no-det'}",
             (255, 255, 255)),
        ])
        _draw_title_bar(p_a, [
            ("[OFFLINE-ONLY] PATH A  Vicon-cropped", (180, 230, 255)),
            (f"status={patha_status}  "
             f"{'FIRED' if patha_quad is not None else 'no-det'}",
             (180, 230, 255)),
        ])
        pb_reason = pathb_status.reason if pathb_status is not None else "n/a"
        pb_diag = (f"diag={pathb_status.picked_diagonal:.0f}"
                   if pathb_status and pathb_status.picked_diagonal is not None
                   else "diag=-")
        _draw_title_bar(p_b, [
            ("PATH B  Track + Quality Gate (DEPLOYABLE)", (180, 255, 200)),
            (f"reason={pb_reason}  fail={pathb_status.fail_count if pathb_status else 0}  "
             f"{pb_diag}",
             (180, 255, 200)),
        ])

        # Stitch panels into one frame + bottom caveat strip.
        composite = np.zeros((H + BOTTOM_BAR, W3, 3), dtype=np.uint8)
        composite[:H, 0:PANEL]         = p_baseline
        composite[:H, PANEL:2 * PANEL] = p_a
        composite[:H, 2 * PANEL:W3]    = p_b
        caveat = (
            "Path A uses GROUND-TRUTH Vicon pose; for flight, swap in "
            "VIO + pre-race map (Bahnam et al. 2026)."
        )
        cv2.rectangle(composite, (0, H), (W3, H + BOTTOM_BAR), (20, 20, 20), -1)
        cv2.putText(composite, caveat, (8, H + 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

        composite_bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
        writer.write(composite_bgr)

        if (i + 1) % 50 == 0:
            print(
                f"  rendered {i + 1}/{len(selected)}  "
                f"baseline_fired={counts['baseline_fired']}  "
                f"patha_fired={counts['patha_fired']}  "
                f"pathb_fired={counts['pathb_fired']}",
                flush=True,
            )

    writer.release()
    sz = os.path.getsize(args.out)
    print(f"saved -> {args.out} ({sz / 1e6:.2f} MB)", flush=True)
    n = len(selected)
    print(
        f"summary over {n} frames:\n"
        f"  Baseline fired:  {counts['baseline_fired']}/{n} "
        f"({100*counts['baseline_fired']/max(n,1):.1f}%)\n"
        f"  Path A   fired:  {counts['patha_fired']}/{n} "
        f"({100*counts['patha_fired']/max(n,1):.1f}%) "
        f"[no_vicon={counts['patha_no_vicon']}, "
        f"no_gate_visible={counts['patha_no_gate_visible']}, "
        f"off_frame={counts['patha_off_frame']}, "
        f"no_decode={counts['patha_no_decode']}]\n"
        f"  Path B   fired:  {counts['pathb_fired']}/{n} "
        f"({100*counts['pathb_fired']/max(n,1):.1f}%) "
        f"[handoffs={counts['pathb_handoffs']}, "
        f"fail_drops={counts['pathb_fail_drops']}]\n"
        f"  CSV rows missing: {counts['csv_missing_row']}/{n}",
        flush=True,
    )


if __name__ == "__main__":
    main()
