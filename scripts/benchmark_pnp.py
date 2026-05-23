"""Benchmark PnP pose estimation on a single flight directory.

Pipeline per frame
------------------
1. GateNet (segmentation) + QuAdGate (corner refinement) → detected 2-D
   inner corners in TL/TR/BR/BL order.
2. GT 2-D inner corners (label file, raw marker order 1..4).
3. Best-matching CSV gate: use Vicon (if available) to select the gate whose
   projected centre is closest to the label's 2-D bbox centre, filtered to
   gates with positive camera-frame z (gate in front).  Falls back to PnP
   reprojection-error matching when no CSV is present.
4. 3-D markers sorted to TL/TR/BR/BL via the GT 2-D sort permutation.
5. PnP on GT corners  → GT pose.
   PnP on det corners → detected pose.
6. Report reprojection errors and pose delta (det vs GT-corner pose).
7. Optionally save visualisation images (--visuals-dir).

Reported metrics
----------------
  reproj_gt       PnP reprojection error with GT 2-D corners          (px)
  reproj_det      PnP reprojection error with detected 2-D corners    (px)
  t_err           translation distance between det and GT-corner pose  (m)
  r_err           geodesic angle between det and GT-corner pose        (°)

Usage
-----
    # from the repo root:
    conda run -n ethrc python scripts/benchmark_pnp.py \\
        checkpoints/best_gatenet.pth \\
        data/piloted/flight-01p-ellipse

    python scripts/benchmark_pnp.py checkpoints/best_gatenet.pth \\
        data/piloted/flight-01p-ellipse \\
        --calib /path/to/calibration --stride 10 --out results/pnp_01.csv \\
        --visuals-dir results/visuals_01 --n-visuals 20
"""

from __future__ import annotations

import argparse
import os
import sys
import shutil
from pathlib import Path

# allow running directly without installing the package
_SRC = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import cv2
import json
import numpy as np
import pandas as pd

from gate_detection.config import load_config
from gate_detection.data.discovery import find_flight_paths
from gate_detection.data.labels import assign_corner_types, parse_gates
from gate_detection.evaluation.pnp import (
    PnPResult,
    best_gate_for_label,
    default_camera_matrix,
    load_calibration,
    markers_from_csv_row,
    rotation_error_deg,
    solve_pnp_inner,
    sort_inner_perm,
    sort_markers_perm,
    translation_error,
    vicon_camera_pose,
    vicon_gate_selection,
)
from gate_detection.inference.predictor import GatePredictor
from gate_detection.postprocessing.quadgate import detect_corners


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _synthesize_outer(inner_norm: np.ndarray, scale: float = 1.4) -> np.ndarray:
    c = inner_norm.mean(axis=0)
    return c + scale * (inner_norm - c)


def _build_prior(g: dict, img_size: int) -> np.ndarray:
    """(8,2) QuAdGate prior. Synthesises outer at 1.4× when absent (17-col)."""
    inner = g["inner"]
    outer = g["outer"] if g["outer"] is not None else _synthesize_outer(inner)
    return assign_corner_types(outer * img_size, inner * img_size)


# ---------------------------------------------------------------------------
# real-dataset (pose-CSV) helpers
# ---------------------------------------------------------------------------

def _quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Hamilton unit quaternion (x,y,z,w) → 3×3 rotation that maps body→world.

    Matches ROS / Eigen convention.
    """
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
        [  2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qw*qx)],
        [  2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)],
    ], dtype=np.float64)


def _read_camera_info(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse the camera_info.txt format used by data/real (pseudo-YAML with
    ``np.float64(...)`` wrappers around values).
    """
    import re
    text = open(path).read()
    text = re.sub(r"np\.float64\(([^)]+)\)", r"\1", text)

    def grab(key):
        m = re.search(rf"^{key}\s*:\s*\[([^\]]+)\]", text, re.MULTILINE)
        if not m:
            return None
        return [float(v) for v in m.group(1).split(",")]
    K = np.array(grab("K"), dtype=np.float64).reshape(3, 3)
    D = np.array(grab("D"), dtype=np.float64).reshape(-1, 1)
    if D.size < 5:
        D = np.concatenate([D.flatten(),
                             np.zeros(5 - D.size, dtype=np.float64)]).reshape(-1, 1)
    return K, D


def _nearest_pose_row(df: pd.DataFrame, ts_ns: int,
                      max_dt_ns: int = 50_000_000):
    """Return the pose CSV row whose timestamp is closest to *ts_ns* (within
    ``max_dt_ns`` nanoseconds), or ``None``.
    """
    ts_arr = df["timestamp_ns"].values
    idx    = int(np.searchsorted(ts_arr, ts_ns))
    cands  = []
    if idx > 0:
        cands.append(idx - 1)
    if idx < len(ts_arr):
        cands.append(idx)
    if not cands:
        return None
    best   = min(cands, key=lambda i: abs(int(ts_arr[i]) - ts_ns))
    if abs(int(ts_arr[best]) - ts_ns) > max_dt_ns:
        return None
    return df.iloc[best]


# Default marker offsets in the gate's local frame, matching the TII gate
# (1.52 m × 1.53 m square, normal along the gate-local x axis).  Override
# with --gate-half-y / --gate-half-z if your gate is a different size.
_DEFAULT_GATE_HALF = 0.76


def _gate_markers_world(gate_pose_row, half_y: float, half_z: float) -> np.ndarray:
    """4 marker world-frame positions from a gate-pose CSV row.

    Markers are placed at ±half_y × ±half_z in the gate's local frame
    (normal along local x), then transformed to world via the gate's pose.
    Returned in the order [(+y,+z), (-y,+z), (+y,-z), (-y,-z)].
    """
    p = np.array([gate_pose_row["pos_x"], gate_pose_row["pos_y"],
                  gate_pose_row["pos_z"]], dtype=np.float64)
    R = _quat_to_R(gate_pose_row["ori_x"], gate_pose_row["ori_y"],
                   gate_pose_row["ori_z"], gate_pose_row["ori_w"])
    offs = np.array([[0,  half_y,  half_z],
                     [0, -half_y,  half_z],
                     [0,  half_y, -half_z],
                     [0, -half_y, -half_z]], dtype=np.float64)
    return p + offs @ R.T   # (4,3) world coords


# ---------------------------------------------------------------------------
# visualisation
# ---------------------------------------------------------------------------

def _save_visual(
    img_rgb: np.ndarray,
    fname: str,
    gt_inner: np.ndarray | None,      # (4,2) sorted inner GT corners
    det_inner: np.ndarray | None,     # (4,2) detected inner corners
    pnp_gt: PnPResult | None,
    pnp_det: PnPResult | None,
    pts_3d: np.ndarray | None,        # (4,3) world-frame markers
    K: np.ndarray,
    dist: np.ndarray,
    save_path: str,
    stats: dict,
) -> None:
    """Save one visualisation image with overlaid corners and stats."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import cv2 as _cv2

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img_rgb)
        ax.axis("off")

        corner_labels = ["TL", "TR", "BR", "BL"]

        # GT label inner corners — green circles
        if gt_inner is not None:
            for i, (u, v) in enumerate(gt_inner):
                ax.plot(u, v, "go", ms=8, markeredgewidth=1.5,
                        markeredgecolor="darkgreen", zorder=5)
                ax.text(u + 5, v - 5, corner_labels[i], color="lime",
                        fontsize=7, zorder=6)

        # detected inner corners — cyan squares
        if det_inner is not None:
            for i, (u, v) in enumerate(det_inner):
                ax.plot(u, v, "cs", ms=8, markeredgewidth=1.5,
                        markeredgecolor="darkcyan", zorder=5)

        # PnP-reprojected GT markers — blue ×
        if pnp_gt is not None and pts_3d is not None:
            proj, _ = _cv2.projectPoints(
                pts_3d, pnp_gt.rvec, pnp_gt.tvec, K, dist
            )
            for u, v in proj.reshape(-1, 2):
                ax.plot(u, v, "bx", ms=10, mew=2, zorder=6)

        # PnP-reprojected det markers — orange ×
        if pnp_det is not None and pts_3d is not None:
            proj, _ = _cv2.projectPoints(
                pts_3d, pnp_det.rvec, pnp_det.tvec, K, dist
            )
            for u, v in proj.reshape(-1, 2):
                ax.plot(u, v, "x", color="orange", ms=10, mew=2, zorder=6)

        # legend
        handles = [
            mpatches.Patch(color="lime",   label="GT corners (label)"),
            mpatches.Patch(color="cyan",   label="Detected corners"),
            mpatches.Patch(color="blue",   label="PnP(GT) reprojection"),
            mpatches.Patch(color="orange", label="PnP(det) reprojection"),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=8,
                  framealpha=0.7)

        # stats text
        lines = [os.path.basename(fname)]
        if stats.get("reproj_gt") is not None:
            lines.append(f"reproj_gt  = {stats['reproj_gt']:.1f} px")
        if stats.get("reproj_det") is not None:
            lines.append(f"reproj_det = {stats['reproj_det']:.1f} px")
        if stats.get("t_err_m") is not None:
            lines.append(f"Δt (det-GT) = {stats['t_err_m']:.3f} m")
        if stats.get("r_err_deg") is not None:
            lines.append(f"Δr (det-GT) = {stats['r_err_deg']:.1f}°")
        txt = "\n".join(lines)
        ax.text(5, 5, txt, transform=ax.transData, fontsize=8,
                verticalalignment="top",
                bbox=dict(facecolor="black", alpha=0.55, pad=3))

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.tight_layout(pad=0.5)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"    [vis] failed to save {save_path}: {e}")


# ---------------------------------------------------------------------------
# per-frame processing
# ---------------------------------------------------------------------------

def _process_frame(fname, cam_dir, lbl_dir, csv_df, predictor, K, dist,
                   R_cam_body=None, t_cam_body=None,
                   vis_path=None):
    stem     = os.path.splitext(fname)[0]
    img_path = os.path.join(cam_dir, fname)
    lbl_path = os.path.join(lbl_dir, stem + ".txt")

    try:
        img_rgb, pred = predictor.predict_image(img_path)
    except FileNotFoundError:
        return None
    img_h, img_w = img_rgb.shape[:2]

    # GT inner corners (raw marker order, pixel coords). Require all 4 corners
    # to be fully visible (vis == 2): rules out frames where any corner is
    # occluded or clipped at the image edge — PnP on those is unreliable.
    label_gates = [g for g in parse_gates(lbl_path)
                   if all(v == 2 for v in g["inner_vis"])]
    if not label_gates:
        return None
    lg         = label_gates[0]
    raw_inner  = lg["inner"].astype(np.float64) * np.array([img_w, img_h])

    # also drop frames where any corner falls outside the image (label noise)
    if (raw_inner[:, 0] < 0).any() or (raw_inner[:, 0] > img_w).any() \
       or (raw_inner[:, 1] < 0).any() or (raw_inner[:, 1] > img_h).any():
        return None

    # CSV row
    row = csv_df.loc[fname] if (csv_df is not None and fname in csv_df.index) else None

    # match label → best CSV gate
    gate_id    = None
    pts_3d_raw = None
    if row is not None:
        # Use Vicon-based gate selection when drone pose is available
        label_center = raw_inner.mean(axis=0)
        if R_cam_body is not None and t_cam_body is not None:
            gate_id = vicon_gate_selection(row, K, R_cam_body, t_cam_body,
                                           label_center_2d=label_center)
        if gate_id is None:
            match = best_gate_for_label(raw_inner, row, K, dist)
            if match is not None:
                gate_id, _ = match
        if gate_id is not None:
            pts_3d_raw = markers_from_csv_row(row, gate_id)

    # sort 2-D and 3-D to TL/TR/BR/BL independently. The CSV's marker1..4 order
    # is per-gate arbitrary, so projecting them through the Vicon camera pose and
    # sorting in image space gives a correspondence-correct ordering. Falls back
    # to applying the 2-D perm to the 3-D array (only correct if the CSV uses
    # the [TL,TR,BR,BL] convention for that gate).
    perm_2d         = sort_inner_perm(raw_inner)
    gt_inner_sorted = raw_inner[perm_2d]

    pts_3d_sorted = None
    if pts_3d_raw is not None:
        perm_3d = None
        if row is not None and R_cam_body is not None and t_cam_body is not None:
            perm_3d = sort_markers_perm(pts_3d_raw, row, R_cam_body, t_cam_body, K, dist)
        if perm_3d is None:
            # CLAUDE.md sec.5: announce the fallback. Applying the 2-D image-space
            # perm to 3-D world points is correspondence-correct ONLY when the CSV
            # marker numbering coincides with [TL,TR,BR,BL]; otherwise PnP corres-
            # pondences mismatch and the reproj residual blows up (Report 42 #1).
            _why = ("no Vicon row" if row is None
                    else "no cam_T_body extrinsics" if (R_cam_body is None or t_cam_body is None)
                    else "sort_markers_perm returned None")
            print(f"[WARN] benchmark_pnp {fname}: vicon marker sort unavailable "
                  f"({_why}); falling back to 2-D image-space perm (only valid "
                  f"if CSV ordering is [TL,TR,BR,BL] for this gate)", flush=True)
            perm_3d = perm_2d
        pts_3d_sorted = pts_3d_raw[perm_3d]

    # GT PnP
    pnp_gt: PnPResult | None = None
    if pts_3d_sorted is not None:
        pnp_gt = solve_pnp_inner(gt_inner_sorted, pts_3d_sorted, K, dist)

    # QuAdGate detection (build prior from inner corners, works for 17-col)
    prior    = _build_prior(lg, predictor.img_size)
    detected = detect_corners(pred, prior)

    # detected PnP (scale corners from img_size → original resolution)
    pnp_det: PnPResult | None = None
    det_inner: np.ndarray | None = None
    if detected is not None and pts_3d_sorted is not None:
        sx, sy    = img_w / predictor.img_size, img_h / predictor.img_size
        det_inner = detected[4:8].copy().astype(np.float64)
        det_inner[:, 0] *= sx
        det_inner[:, 1] *= sy
        pnp_det = solve_pnp_inner(det_inner, pts_3d_sorted, K, dist)

    # pose delta (det vs GT-corner PnP)
    t_err = r_err = None
    if pnp_gt is not None and pnp_det is not None:
        t_err = translation_error(pnp_gt.tvec, pnp_det.tvec)
        r_err = rotation_error_deg(pnp_gt.rvec, pnp_det.rvec)

    # Absolute comparison: PnP-derived camera pose vs Vicon-derived camera pose.
    # Both are world→camera transforms (rvec, tvec).  Translation error is the L2
    # distance between camera origins in world frame.
    vicon_t_err = vicon_r_err = None
    if (row is not None and pnp_gt is not None
            and R_cam_body is not None and t_cam_body is not None):
        vicon_pose = vicon_camera_pose(row, R_cam_body, t_cam_body)
        if vicon_pose is not None:
            v_rvec, v_tvec = vicon_pose
            R_pnp, _   = cv2.Rodrigues(pnp_gt.rvec)
            R_vicon, _ = cv2.Rodrigues(v_rvec)
            cam_pnp_w   = (-R_pnp.T   @ pnp_gt.tvec).flatten()
            cam_vicon_w = (-R_vicon.T @ v_tvec).flatten()
            vicon_t_err = float(np.linalg.norm(cam_pnp_w - cam_vicon_w))
            vicon_r_err = rotation_error_deg(pnp_gt.rvec, v_rvec)

    # Same for the detected-corner PnP (this is the realistic "production" error
    # — what the drone would actually see from its onboard detection pipeline).
    vicon_t_err_det = vicon_r_err_det = None
    if (row is not None and pnp_det is not None
            and R_cam_body is not None and t_cam_body is not None):
        vicon_pose = vicon_camera_pose(row, R_cam_body, t_cam_body)
        if vicon_pose is not None:
            v_rvec, v_tvec = vicon_pose
            R_pnp, _   = cv2.Rodrigues(pnp_det.rvec)
            R_vicon, _ = cv2.Rodrigues(v_rvec)
            cam_pnp_w   = (-R_pnp.T   @ pnp_det.tvec).flatten()
            cam_vicon_w = (-R_vicon.T @ v_tvec).flatten()
            vicon_t_err_det = float(np.linalg.norm(cam_pnp_w - cam_vicon_w))
            vicon_r_err_det = rotation_error_deg(pnp_det.rvec, v_rvec)

    # distance drone → nearest gate (for distance-sliced reporting)
    dist_m = float("nan")
    if row is not None and gate_id is not None:
        dx = row.get("drone_x", float("nan"))
        gx = row.get(f"gate{gate_id}_int_x", float("nan"))
        gy_d = row.get("drone_y", float("nan")); gy_g = row.get(f"gate{gate_id}_int_y", float("nan"))
        gz_d = row.get("drone_z", float("nan")); gz_g = row.get(f"gate{gate_id}_int_z", float("nan"))
        if all(not pd.isna(v) for v in [dx, gx, gy_d, gy_g, gz_d, gz_g]):
            dist_m = float(np.sqrt((dx-gx)**2 + (gy_d-gy_g)**2 + (gz_d-gz_g)**2))

    result = dict(
        fname           = fname,
        detected        = detected is not None,
        csv_ok          = row is not None,
        gate_id         = gate_id,
        pnp_gt_ok       = pnp_gt  is not None,
        pnp_det_ok      = pnp_det is not None,
        reproj_gt       = pnp_gt.reproj_err  if pnp_gt  is not None else None,
        reproj_det      = pnp_det.reproj_err if pnp_det is not None else None,
        t_err_m         = t_err,
        r_err_deg       = r_err,
        vicon_t_err     = vicon_t_err,           # PnP(GT corners) vs Vicon
        vicon_r_err     = vicon_r_err,
        vicon_t_err_det = vicon_t_err_det,       # PnP(detected)   vs Vicon
        vicon_r_err_det = vicon_r_err_det,
        dist_m          = dist_m,
    )

    # optional visualisation
    if vis_path is not None:
        _save_visual(
            img_rgb, fname,
            gt_inner=gt_inner_sorted,
            det_inner=det_inner,
            pnp_gt=pnp_gt,
            pnp_det=pnp_det,
            pts_3d=pts_3d_sorted,
            K=K, dist=dist,
            save_path=vis_path,
            stats=result,
        )

    return result


# ---------------------------------------------------------------------------
# per-frame processing — REAL dataset (data/real format)
# ---------------------------------------------------------------------------

def _process_frame_real(
    fname: str,
    img_dir: str,
    pose_drone_df: pd.DataFrame,
    pose_gate_df: pd.DataFrame,
    predictor,
    K: np.ndarray,
    dist: np.ndarray,
    R_cam_body: np.ndarray | None,
    t_cam_body: np.ndarray | None,
    *,
    half_y: float = _DEFAULT_GATE_HALF,
    half_z: float = _DEFAULT_GATE_HALF,
    vis_path: str | None = None,
):
    """Process one frame from the real dataset.

    No labels are available, so only the detector-corner branch is run; the
    absolute comparison uses the time-matched drone pose from
    ``pose_drone.csv`` against the PnP-derived camera pose.
    """
    stem     = os.path.splitext(fname)[0]
    img_path = os.path.join(img_dir, fname)
    try:
        ts_ns = int(stem)
    except ValueError:
        return None

    drone_row = _nearest_pose_row(pose_drone_df, ts_ns)
    gate_row  = _nearest_pose_row(pose_gate_df,  ts_ns)
    if drone_row is None or gate_row is None:
        return None

    try:
        img_rgb, pred = predictor.predict_image(img_path)
    except FileNotFoundError:
        return None
    img_h, img_w = img_rgb.shape[:2]

    # 3-D markers in world frame (constant gate model, transformed by pose)
    pts_3d_raw = _gate_markers_world(gate_row, half_y, half_z)

    # build a "fake" CSV row with the columns vicon_camera_pose / sort_markers_perm
    # expect (drone_x, drone_y, drone_z, drone_rot[0..8]).  We compute R_w2b from
    # the drone quaternion (Hamilton, body→world → transpose for w2b).
    R_b2w = _quat_to_R(drone_row["ori_x"], drone_row["ori_y"],
                       drone_row["ori_z"], drone_row["ori_w"])
    R_w2b = R_b2w.T
    fake_row = pd.Series({
        "drone_x": drone_row["pos_x"],
        "drone_y": drone_row["pos_y"],
        "drone_z": drone_row["pos_z"],
        **{f"drone_rot[{i}]": R_w2b.flat[i] for i in range(9)},
    })

    # detector
    prior = np.zeros((8, 2), dtype=np.float32)   # no label prior — detector
                                                 # uses its segmentation output
    # Build a dummy "inner" to feed _build_prior: roughly the image centre square
    cx, cy = predictor.img_size // 2, predictor.img_size // 2
    s = predictor.img_size // 4
    fake_inner_norm = np.array([[cx-s, cy-s], [cx+s, cy-s],
                                [cx+s, cy+s], [cx-s, cy+s]],
                               dtype=np.float64) / predictor.img_size
    fake_g = {"inner": fake_inner_norm, "outer": None}
    prior   = _build_prior(fake_g, predictor.img_size)
    detected = detect_corners(pred, prior)

    pnp_det: PnPResult | None = None
    det_inner: np.ndarray | None = None
    pts_3d_sorted = None
    if detected is not None:
        sx, sy    = img_w / predictor.img_size, img_h / predictor.img_size
        det_inner = detected[4:8].copy().astype(np.float64)
        det_inner[:, 0] *= sx
        det_inner[:, 1] *= sy

        perm_2d  = sort_inner_perm(det_inner)
        det_inner_sorted = det_inner[perm_2d]

        if R_cam_body is not None and t_cam_body is not None:
            perm_3d = sort_markers_perm(pts_3d_raw, fake_row,
                                        R_cam_body, t_cam_body, K, dist)
        else:
            perm_3d = None
        if perm_3d is None:
            perm_3d = list(range(4))   # arbitrary; PnP may converge anyway
        pts_3d_sorted = pts_3d_raw[perm_3d]

        pnp_det = solve_pnp_inner(det_inner_sorted, pts_3d_sorted, K, dist)

    # absolute pose error vs the time-matched drone pose
    vicon_t_err_det = vicon_r_err_det = None
    if (pnp_det is not None and R_cam_body is not None and t_cam_body is not None):
        v_pose = vicon_camera_pose(fake_row, R_cam_body, t_cam_body)
        if v_pose is not None:
            v_rvec, v_tvec = v_pose
            R_pnp, _   = cv2.Rodrigues(pnp_det.rvec)
            R_vicon, _ = cv2.Rodrigues(v_rvec)
            cam_pnp_w   = (-R_pnp.T   @ pnp_det.tvec).flatten()
            cam_vicon_w = (-R_vicon.T @ v_tvec).flatten()
            vicon_t_err_det = float(np.linalg.norm(cam_pnp_w - cam_vicon_w))
            vicon_r_err_det = rotation_error_deg(pnp_det.rvec, v_rvec)

    # drone→gate distance
    g_pos = np.array([gate_row["pos_x"], gate_row["pos_y"], gate_row["pos_z"]])
    d_pos = np.array([drone_row["pos_x"], drone_row["pos_y"], drone_row["pos_z"]])
    dist_m = float(np.linalg.norm(g_pos - d_pos))

    result = dict(
        fname           = fname,
        detected        = detected is not None,
        csv_ok          = True,
        gate_id         = 1,
        pnp_gt_ok       = False,                 # no labels in real dataset
        pnp_det_ok      = pnp_det is not None,
        reproj_gt       = None,
        reproj_det      = pnp_det.reproj_err if pnp_det is not None else None,
        t_err_m         = None,
        r_err_deg       = None,
        vicon_t_err     = None,
        vicon_r_err     = None,
        vicon_t_err_det = vicon_t_err_det,
        vicon_r_err_det = vicon_r_err_det,
        dist_m          = dist_m,
    )

    if vis_path is not None:
        _save_visual(
            img_rgb, fname,
            gt_inner=None,
            det_inner=det_inner,
            pnp_gt=None,
            pnp_det=pnp_det,
            pts_3d=pts_3d_sorted,
            K=K, dist=dist,
            save_path=vis_path,
            stats=result,
        )
    return result


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _finite(lst):
    a = np.array([v for v in lst if v is not None], dtype=float)
    return a[np.isfinite(a)]


def _stat_line(label, a, unit):
    if len(a) == 0:
        return f"  {label:<28}: n/a"
    return (f"  {label:<28}: mean={a.mean():.3f}  "
            f"median={np.median(a):.3f}  "
            f"p95={np.percentile(a, 95):.3f}  {unit}")


def _pct_rows(a, thresholds, unit):
    rows = []
    for thr in thresholds:
        n = int((a <= thr).sum())
        rows.append(f"    ≤ {thr}{unit}  {n:>4d}/{len(a)}  ({100*n/len(a):.1f}%)")
    return rows


_DIST_BINS  = [0, 1, 2, 3, 5, 8, float("inf")]
_DIST_LBLS  = ["<1 m", "1–2 m", "2–3 m", "3–5 m", "5–8 m", ">8 m"]


def _print_summary(results, flight_name, stride, K) -> None:
    n      = len(results)
    n_det  = sum(1 for r in results if r["detected"])
    n_csv  = sum(1 for r in results if r["csv_ok"])
    n_gpnp = sum(1 for r in results if r["pnp_gt_ok"])
    n_dpnp = sum(1 for r in results if r["pnp_det_ok"])
    n_both = sum(1 for r in results if r["t_err_m"] is not None)
    rg   = _finite([r["reproj_gt"]   for r in results])
    rd   = _finite([r["reproj_det"]  for r in results])
    terr = _finite([r["t_err_m"]     for r in results])
    rerr = _finite([r["r_err_deg"]   for r in results])

    S = "─" * 66
    print(f"\n{S}")
    print(f"  Flight              : {flight_name}")
    print(f"  Frames evaluated    : {n}  (stride={stride})")
    print(f"  Frames with CSV     : {n_csv}")
    print(f"  Camera K            : "
          f"fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  "
          f"cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
    print()
    print(f"  ── QuAdGate detection ────────────────────────────────────────")
    print(f"  Detection rate      : {n_det}/{n}  ({100*n_det/max(n,1):.1f}%)")
    print()
    print(f"  ── PnP ───────────────────────────────────────────────────────")
    print(f"  GT  PnP success     : {n_gpnp}/{n_csv}")
    print(f"  Det PnP success     : {n_dpnp}/{n_det}")
    print(f"  Frames with Δ pose  : {n_both}")
    print()
    print(_stat_line("Reproj GT  (px)", rg, "px"))
    print(_stat_line("Reproj Det (px)", rd, "px"))

    if len(terr) > 0:
        print()
        print(f"  ── pose Δ (det vs GT-corner PnP)  [{n_both} frames] ──────────────")
        print(_stat_line("Translation (m)", terr, "m"))
        for l in _pct_rows(terr, [0.05, 0.10, 0.25, 0.50], " m"):
            print(l)
        print(_stat_line("Rotation    (°)", rerr, "°"))
        for l in _pct_rows(rerr, [2.0, 5.0, 10.0, 20.0], "°"):
            print(l)

    # absolute Vicon comparison: PnP-derived camera-in-world vs Vicon camera-in-world
    vt_gt  = _finite([r["vicon_t_err"]     for r in results])
    vr_gt  = _finite([r["vicon_r_err"]     for r in results])
    vt_det = _finite([r["vicon_t_err_det"] for r in results])
    vr_det = _finite([r["vicon_r_err_det"] for r in results])
    if len(vt_gt) > 0 or len(vt_det) > 0:
        print()
        print(f"  ── camera-pose error vs Vicon GT  (drone position accuracy) ──")
        if len(vt_gt) > 0:
            print(f"  [GT-corner PnP]  ({len(vt_gt)} frames)")
            print(_stat_line("Position (m)", vt_gt, "m"))
            for l in _pct_rows(vt_gt, [0.05, 0.10, 0.25, 0.50], " m"):
                print(l)
            print(_stat_line("Rotation (°)", vr_gt, "°"))
            for l in _pct_rows(vr_gt, [2.0, 5.0, 10.0, 20.0], "°"):
                print(l)
        if len(vt_det) > 0:
            print(f"  [Detected PnP]  ({len(vt_det)} frames)")
            print(_stat_line("Position (m)", vt_det, "m"))
            for l in _pct_rows(vt_det, [0.05, 0.10, 0.25, 0.50], " m"):
                print(l)
            print(_stat_line("Rotation (°)", vr_det, "°"))
            for l in _pct_rows(vr_det, [2.0, 5.0, 10.0, 20.0], "°"):
                print(l)

    # distance-sliced breakdown
    dists = _finite([r["dist_m"] for r in results])
    if len(dists) > 0 and len(terr) > 0:
        print()
        print(f"  ── translation Δ by drone→gate distance ──────────────────────")
        print(f"  {'Dist':>8}  {'N':>4}  {'Det rate':>10}  {'med t_err (m)':>14}")
        print(f"  {'─'*8}  {'─'*4}  {'─'*10}  {'─'*14}")
        for lo, hi, lbl in zip(_DIST_BINS[:-1], _DIST_BINS[1:], _DIST_LBLS):
            sub = [r for r in results
                   if not np.isnan(r.get("dist_m", float("nan")))
                   and lo <= r["dist_m"] < hi]
            if not sub:
                continue
            nd  = sum(1 for r in sub if r["detected"])
            te  = _finite([r["t_err_m"] for r in sub])
            med = f"{np.median(te):.3f}" if len(te) > 0 else "—"
            print(f"  {lbl:>8}  {len(sub):>4}  {nd/len(sub):>10.1%}  {med:>14}")

    print(f"{S}\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

_DEFAULT_CALIB = os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "gate_detection", "calibration",
))


def main() -> None:
    p = argparse.ArgumentParser(description="PnP pose benchmark on a single flight")
    p.add_argument("checkpoint", help="path to .pth checkpoint")
    p.add_argument("flight",     help="flight folder")
    p.add_argument("--dataset",  choices=["tii", "real"], default="tii",
                   help="dataset format: 'tii' (label files + sync CSV, default) "
                        "or 'real' (data/real format: pose_drone.csv + pose_gate.csv "
                        "+ images/ + camera_info.txt)")
    p.add_argument("--calib",   default=_DEFAULT_CALIB,
                   help="for 'tii': directory with calib_*.json + drone_to_camera.json; "
                        "for 'real': ignored (camera_info.txt is read from the flight "
                        "folder unless --camera-info is given)")
    p.add_argument("--camera-info", default=None,
                   help="(real) override path to camera_info.txt")
    p.add_argument("--extrinsic", default=None,
                   help="(real) path to a JSON with body→camera extrinsic "
                        "(translation x/y/z + rotation w/x/y/z); reuses the "
                        "TII drone_to_camera.json 'piloted' entry if not given")
    p.add_argument("--gate-half-y", type=float, default=_DEFAULT_GATE_HALF,
                   help="(real) half-width of the gate marker rectangle in metres "
                        f"(default {_DEFAULT_GATE_HALF})")
    p.add_argument("--gate-half-z", type=float, default=_DEFAULT_GATE_HALF,
                   help="(real) half-height of the gate marker rectangle in metres "
                        f"(default {_DEFAULT_GATE_HALF})")
    p.add_argument("--config",  default=None)
    p.add_argument("--stride",  type=int, default=30,
                   help="sample every N-th frame (default 30)")
    p.add_argument("--out",     default="pnp_benchmark.csv",
                   help="save per-frame CSV here")
    p.add_argument("--visuals-dir", default=None,
                   help="directory to save visualisation images (skip if not set)")
    p.add_argument("--n-visuals", type=int, default=16,
                   help="max number of visualisation images to save (default 16)")
    args = p.parse_args()

    cfg = load_config(args.config)
    # Snapshot the config beside the CSV so a run is reproducible from its
    # artefact alone (CLAUDE.md §0 — no result that cannot be re-run).
    _cfg_snap = Path(args.out).with_suffix("").as_posix() + ".config.yaml"
    Path(_cfg_snap).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, _cfg_snap)
    print(f"Config snapshot     → {_cfg_snap}")
    predictor = GatePredictor(
        args.checkpoint,
        img_size=cfg["data"]["img_size"],
        model_name=cfg["model"].get("name", "gatenet"),
        width_factor=cfg["model"]["width_factor"],
        encoder_name=cfg["model"].get("encoder_name", "resnet34"),
        encoder_weights=cfg["model"].get("encoder_weights", None),
        dropout_p=cfg["model"].get("dropout_p", 0.0),
    )

    flight_name = os.path.basename(os.path.normpath(args.flight))

    print(f"Checkpoint          : {os.path.basename(args.checkpoint)}"
          f"  (epoch {predictor.checkpoint_meta['epoch']},"
          f" val_loss={predictor.checkpoint_meta['val_loss']:.4f})")
    print(f"Dataset             : {args.dataset}")
    print(f"Flight              : {flight_name}")

    if args.dataset == "tii":
        _run_tii(args, flight_name, predictor)
    else:
        _run_real(args, flight_name, predictor)


# ---------------------------------------------------------------------------
# dataset-specific drivers
# ---------------------------------------------------------------------------

def _run_tii(args, flight_name: str, predictor) -> None:
    calib_dir   = os.path.realpath(args.calib)
    R_cam_body = t_cam_body = None
    try:
        K, dist, R_cam_body, t_cam_body = load_calibration(flight_name, calib_dir)
        print(f"Calibration         : {calib_dir}")
    except Exception as e:
        print(f"Warning: calibration not loaded ({e}). Using fallback K.")
        K, dist = default_camera_matrix(), np.zeros((5, 1))

    cam_dir, lbl_dir, csv_path = find_flight_paths(args.flight)
    csv_df = pd.read_csv(csv_path, index_col="img_filename") if csv_path else None

    all_imgs = sorted(f for f in os.listdir(cam_dir) if f.endswith(".jpg"))
    sampled  = all_imgs[::args.stride]
    print(f"Frames              : {len(sampled)} sampled from {len(all_imgs)} total\n")

    vis_dir   = args.visuals_dir
    n_vis_max = args.n_visuals
    n_vis_saved = 0

    results = []
    for fname in sampled:
        vis_path = None
        if vis_dir is not None and n_vis_saved < n_vis_max:
            vis_path = os.path.join(vis_dir, fname.replace(".jpg", ".png"))

        rec = _process_frame(
            fname, cam_dir, lbl_dir, csv_df, predictor, K, dist,
            R_cam_body=R_cam_body, t_cam_body=t_cam_body,
            vis_path=vis_path,
        )
        if rec is not None:
            results.append(rec)
            if vis_path is not None and rec.get("pnp_gt_ok"):
                n_vis_saved += 1

    if not results:
        print("No valid frames found.")
        return

    _print_summary(results, flight_name, args.stride, K)
    pd.DataFrame(results).to_csv(args.out, index=False)
    print(f"Per-frame results   → {args.out}")
    if vis_dir is not None:
        print(f"Visualisations      → {vis_dir}  ({n_vis_saved} images)")


def _run_real(args, flight_name: str, predictor) -> None:
    flight_dir = args.flight
    img_dir    = os.path.join(flight_dir, "images")
    drone_csv  = os.path.join(flight_dir, "pose_drone.csv")
    gate_csv   = os.path.join(flight_dir, "pose_gate.csv")
    cam_info   = args.camera_info or os.path.join(flight_dir, "camera_info.txt")

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"image directory not found: {img_dir}")
    if not (os.path.exists(drone_csv) and os.path.exists(gate_csv)):
        raise FileNotFoundError(
            f"pose_drone.csv and pose_gate.csv must exist in {flight_dir}")

    K, dist = _read_camera_info(cam_info)
    print(f"Camera info         : {cam_info}")
    print(f"  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}  "
          f"|D|={int(np.linalg.norm(dist))!s}")

    # body→camera extrinsic: from --extrinsic JSON, otherwise reuse TII piloted
    R_cam_body = t_cam_body = None
    if args.extrinsic and os.path.exists(args.extrinsic):
        e = json.load(open(args.extrinsic))
        t_cam_body = np.array([e["translation"]["x"], e["translation"]["y"],
                                e["translation"]["z"]], dtype=np.float64)
        q = e["rotation"]
        R_cam_body = _quat_to_R(q["x"], q["y"], q["z"], q["w"])
        print(f"Extrinsic           : {args.extrinsic}")
    else:
        try:
            calib_dir = os.path.realpath(args.calib)
            ext = json.load(open(os.path.join(calib_dir, "drone_to_camera.json")))
            t_cam_body = np.array([ext["translation"]["x"],
                                    ext["translation"]["y"],
                                    ext["translation"]["z"]], dtype=np.float64)
            q = ext["rotation"]["piloted"]
            R_cam_body = _quat_to_R(q["x"], q["y"], q["z"], q["w"])
            print(f"Extrinsic           : reusing TII 'piloted' entry from "
                  f"{calib_dir}/drone_to_camera.json")
        except Exception as e:
            print(f"Warning: no body→camera extrinsic loaded ({e}); "
                  f"absolute Vicon comparison will be skipped.")

    pose_drone = pd.read_csv(drone_csv).sort_values("timestamp_ns").reset_index(drop=True)
    pose_gate  = pd.read_csv(gate_csv ).sort_values("timestamp_ns").reset_index(drop=True)

    all_imgs = sorted(f for f in os.listdir(img_dir) if f.endswith((".png", ".jpg")))
    sampled  = all_imgs[::args.stride]
    print(f"Frames              : {len(sampled)} sampled from {len(all_imgs)} total")
    print(f"Gate marker rect    : ±{args.gate_half_y:.2f} m × ±{args.gate_half_z:.2f} m\n")

    vis_dir   = args.visuals_dir
    n_vis_max = args.n_visuals
    n_vis_saved = 0

    results = []
    for fname in sampled:
        vis_path = None
        if vis_dir is not None and n_vis_saved < n_vis_max:
            vis_path = os.path.join(vis_dir,
                                    os.path.splitext(fname)[0] + ".png")
        rec = _process_frame_real(
            fname, img_dir, pose_drone, pose_gate, predictor, K, dist,
            R_cam_body, t_cam_body,
            half_y=args.gate_half_y, half_z=args.gate_half_z,
            vis_path=vis_path,
        )
        if rec is not None:
            results.append(rec)
            if vis_path is not None and rec.get("pnp_det_ok"):
                n_vis_saved += 1

    if not results:
        print("No valid frames found.")
        return

    _print_summary(results, flight_name, args.stride, K)
    pd.DataFrame(results).to_csv(args.out, index=False)
    print(f"Per-frame results   → {args.out}")
    if vis_dir is not None:
        print(f"Visualisations      → {vis_dir}  ({n_vis_saved} images)")


if __name__ == "__main__":
    main()
