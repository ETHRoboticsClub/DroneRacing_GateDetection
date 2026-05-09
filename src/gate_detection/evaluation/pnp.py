"""PnP-based gate pose estimation utilities.

Typical flow
------------
1. Load calibration: ``load_calibration(flight_name, calib_dir)``
   → returns K (3×3), dist (5,1), R_cam_body (3×3), t_cam_body (3,)

2. For each frame, get raw label corners (marker order 1..4):
   ``raw_inner_2d = parse_gates(lbl_path)[0]["inner"] * [img_w, img_h]``

3. Find matching CSV gate: ``best_gate_for_label(raw_inner_2d, row, K, dist)``

4. Sort 3-D markers to TL/TR/BR/BL: ``sort_inner_perm(raw_inner_2d)``

5. Run PnP: ``solve_pnp_inner(pts_2d_sorted, pts_3d_sorted, K, dist)``

6. Compute Vicon GT pose: ``gt_pose_from_vicon(row, gate_id, R_cam_body, t_cam_body)``

7. Compare: ``pose_error(pnp_result, gt_rvec, gt_tvec)``
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# calibration loading
# ---------------------------------------------------------------------------

_CALIB_MAP = {
    "ellipse":     "calib_ap-ellipse-lemniscate.json",
    "lemniscate":  "calib_ap-ellipse-lemniscate.json",
    "trackratm-p": "calib_p-trackRATM.json",
    "trackratm-a": "calib_a-trackRATM.json",
}


def _track_type_key(flight_name: str) -> str:
    name = flight_name.lower()
    if "ellipse" in name:
        return "ellipse"
    if "lemniscate" in name:
        return "lemniscate"
    if "trackratm" in name or "trackRATM" in flight_name:
        # distinguish piloted (p) from autonomous (a)
        parts = name.replace("-", "").replace("_", "")
        if "trackratm" in parts:
            idx = parts.index("trackratm")
            # look for 'p' or 'a' right before 'trackratm' or 'flight'
            if any(f"flight{d}p" in parts for d in "0123456789"):
                return "trackratm-p"
            # fallback: piloted if 'p' appears in flight id segment
            before = name[:name.lower().find("track")]
            return "trackratm-p" if "p" in before.split("-") else "trackratm-a"
    # Fallback: no pattern matched. Per CLAUDE.md section 6, log the default
    # explicitly — silent miscalibration is exactly the failure mode this
    # rule is meant to catch (see docs/report/04_pnp_frame_diagnosis/).
    print(
        f"[WARN] _track_type_key('{flight_name}'): no track pattern matched "
        f"(expected 'ellipse' | 'lemniscate' | 'trackratm-*'); "
        f"defaulting to 'ellipse'. If the rig is mounted differently from the "
        f"ellipse calibration, the camera-from-world pose will be wrong; pass "
        f"--calib-key explicitly once that flag exists.",
        flush=True,
    )
    return "ellipse"


def load_calibration(
    flight_name: str,
    calib_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load camera intrinsics and extrinsics for the given flight.

    Returns
    -------
    K           : (3,3) camera matrix
    dist        : (5,1) distortion coefficients
    R_cam_body  : (3,3) rotation from body frame to camera frame
    t_cam_body  : (3,)  camera origin in body frame (metres)
    """
    key = _track_type_key(flight_name)
    calib_file = os.path.join(calib_dir, _CALIB_MAP[key])

    with open(calib_file) as f:
        c = json.load(f)
    K    = np.array(c["mtx"],  dtype=np.float64)
    dist = np.array(c["dist"], dtype=np.float64).reshape(-1, 1)

    # drone-to-camera extrinsic
    ext_file = os.path.join(calib_dir, "drone_to_camera.json")
    with open(ext_file) as f:
        e = json.load(f)

    t_cam_body = np.array([e["translation"]["x"],
                            e["translation"]["y"],
                            e["translation"]["z"]], dtype=np.float64)

    # quaternion for this track family
    track_family = "piloted" if key.endswith("-p") or key in ("ellipse", "lemniscate") \
                   else "lemniscate"
    # drone_to_camera.json keys: "lemniscate", "ellipse", "trackRATM", "piloted"
    if "ellipse" in key:
        qkey = "ellipse"
    elif "lemniscate" in key:
        qkey = "lemniscate"
    elif key == "trackratm-p":
        qkey = "trackRATM"
    elif key == "trackratm-a":
        qkey = "trackRATM"
    else:
        qkey = "piloted"

    # use "piloted" quaternion when available; fall back to closest match
    rot_dict = e["rotation"]
    qkey = qkey if qkey in rot_dict else "piloted"
    q = rot_dict[qkey]
    w, x, y, z = q["w"], q["x"], q["y"], q["z"]
    R_cam_body = _quat_to_rot(w, x, y, z)

    return K, dist, R_cam_body, t_cam_body


def _quat_to_rot(w, x, y, z) -> np.ndarray:
    """Unit quaternion (w,x,y,z) → 3×3 rotation matrix."""
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------

def markers_from_csv_row(row, gate_id: int) -> np.ndarray | None:
    """Read the 4 marker world-frame positions for *gate_id* (1-based).

    Returns ``(4,3)`` float64 [marker1..4] or ``None`` if any value is missing.
    """
    import pandas as pd
    pts = []
    for k in range(1, 5):
        cols = [f"gate{gate_id}_marker{k}_{ax}" for ax in ("x", "y", "z")]
        if not all(c in row.index for c in cols):
            return None
        vals = [row[c] for c in cols]
        if any(pd.isna(v) for v in vals):
            return None
        pts.append(vals)
    return np.array(pts, dtype=np.float64)


def sort_inner_perm(pts_2d: np.ndarray) -> list[int]:
    """Indices [iTL, iTR, iBR, iBL] that sort (4,2) image-frame corners.

    Uses the same x+y / x-y rule as ``assign_corner_types``.
    """
    s = pts_2d[:, 0] + pts_2d[:, 1]
    d = pts_2d[:, 0] - pts_2d[:, 1]
    return [int(np.argmin(s)), int(np.argmax(d)),
            int(np.argmax(s)), int(np.argmin(d))]


def project_markers_vicon(
    pts_3d: np.ndarray,
    row,
    R_cam_body: np.ndarray,
    t_cam_body: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray | None = None,
) -> np.ndarray | None:
    """Project (4,3) world-frame markers into pixels using the Vicon drone pose.

    Returns ``(4,2)`` pixel coordinates or ``None`` if Vicon data is missing.
    """
    pose = vicon_camera_pose(row, R_cam_body, t_cam_body)
    if pose is None:
        return None
    rvec, tvec = pose
    dist_f = np.zeros((5, 1), dtype=np.float64) if dist is None else dist
    proj, _ = cv2.projectPoints(pts_3d.astype(np.float64), rvec, tvec,
                                K.astype(np.float64), dist_f)
    return proj.reshape(-1, 2)


def sort_markers_perm(
    pts_3d: np.ndarray,
    row,
    R_cam_body: np.ndarray,
    t_cam_body: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray | None = None,
) -> list[int] | None:
    """Indices that sort (4,3) world-frame markers into TL/TR/BR/BL.

    Projects each marker through the Vicon camera pose, then applies the same
    image-space sort as ``sort_inner_perm`` so the 3-D point at output index k
    corresponds to the same corner role (TL/TR/BR/BL) as 2-D label index k
    after applying ``sort_inner_perm``.
    """
    pix = project_markers_vicon(pts_3d, row, R_cam_body, t_cam_body, K, dist)
    if pix is None:
        return None
    return sort_inner_perm(pix)


def best_gate_for_label(
    raw_inner_2d: np.ndarray,
    row,
    K: np.ndarray,
    dist: np.ndarray | None = None,
    n_gates: int = 4,
) -> tuple[int, float] | None:
    """Return (gate_id, reproj_err) for the CSV gate whose markers best explain
    the label corners via PnP (direct marker-order correspondence, no sorting).

    Only gates with tvec_z > 0 (gate in front of camera) are considered.
    Returns ``None`` if no valid match is found.
    """
    import pandas as pd
    K_f    = K.astype(np.float64)
    dist_f = np.zeros((5, 1), dtype=np.float64) if dist is None else np.asarray(dist)
    K_eye  = np.eye(3, dtype=np.float64)
    d_zero = np.zeros((5, 1), dtype=np.float64)
    # undistort once for all gate candidates
    p2d_norm = _undistort_pts(raw_inner_2d.astype(np.float64), K_f, dist_f)
    best_id, best_err = None, float("inf")

    for gid in range(1, n_gates + 1):
        pts_3d = markers_from_csv_row(row, gid)
        if pts_3d is None:
            continue
        ok, rvec, tvec = cv2.solvePnP(
            pts_3d.astype(np.float64), p2d_norm,
            K_eye, d_zero, flags=cv2.SOLVEPNP_SQPNP,
        )
        if not ok or float(tvec[2, 0]) < 0:
            continue
        # reprojection error in original distorted pixel space
        proj, _ = cv2.projectPoints(pts_3d, rvec, tvec, K_f, dist_f)
        err = float(np.mean(
            np.linalg.norm(proj.reshape(-1, 2) - raw_inner_2d, axis=1)
        ))
        if err < best_err:
            best_err, best_id = err, gid

    return (best_id, best_err) if best_id is not None else None


# ---------------------------------------------------------------------------
# GT pose from Vicon
# ---------------------------------------------------------------------------

def gt_pose_from_vicon(
    row,
    gate_id: int,
    R_cam_body: np.ndarray,
    t_cam_body: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute the GT pose of the gate centre in the camera frame.

    Uses Vicon drone and gate poses from the CSV row.

    Frame convention (verified empirically on flight-01p-ellipse):
      - drone_rot[0..8] stores the world→body rotation matrix (R_w2b).
      - drone_to_camera.json quaternion encodes camera→body rotation;
        so R_cam_body.T maps body vectors to camera vectors.
    Chain: world → body → camera:
      g_body = R_w2b @ (g_world - d_world)
      g_cam  = R_cam_body.T @ (g_body - t_cam_body)

    Returns (rvec, tvec) in camera frame, or None.
    """
    import pandas as pd

    # drone world pose
    d_cols = ["drone_x", "drone_y", "drone_z"]
    if not all(c in row.index for c in d_cols):
        return None
    if any(pd.isna(row[c]) for c in d_cols):
        return None
    d_pos = np.array([row[c] for c in d_cols], dtype=np.float64)

    # drone rotation: R_world_to_body (maps world→body)
    rot_cols = [f"drone_rot[{i}]" for i in range(9)]
    if not all(c in row.index for c in rot_cols):
        return None
    R_w2b = np.array([row[c] for c in rot_cols], dtype=np.float64).reshape(3, 3)

    # gate world position
    g_cols = [f"gate{gate_id}_int_{ax}" for ax in ("x", "y", "z")]
    if not all(c in row.index for c in g_cols):
        return None
    if any(pd.isna(row[c]) for c in g_cols):
        return None
    g_pos = np.array([row[c] for c in g_cols], dtype=np.float64)

    # gate rotation from world (stored as flat 3×3 in gate{N}_int_rot[0..8])
    grot_cols = [f"gate{gate_id}_int_rot[{i}]" for i in range(9)]
    R_gate_w = None
    if all(c in row.index for c in grot_cols):
        vals = [row[c] for c in grot_cols]
        if not any(pd.isna(v) for v in vals):
            R_gate_w = np.array(vals, dtype=np.float64).reshape(3, 3)

    # gate centre in body frame, then camera frame
    g_body = R_w2b @ (g_pos - d_pos)
    g_cam  = R_cam_body.T @ (g_body - t_cam_body)
    tvec = g_cam.reshape(3, 1)

    # gate rotation in camera frame
    if R_gate_w is not None:
        R_gate_cam = R_cam_body.T @ R_w2b @ R_gate_w.T
        rvec, _ = cv2.Rodrigues(R_gate_cam)
    else:
        rvec = np.zeros((3, 1), dtype=np.float64)

    return rvec, tvec


def vicon_camera_pose(
    row,
    R_cam_body: np.ndarray,
    t_cam_body: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute the Vicon GT camera pose in the same form as solvePnP output.

    solvePnP with world-frame 3D points returns (rvec, tvec) such that:
        p_cam = R @ p_world + tvec
    where R = world→camera rotation and tvec = world-origin in camera frame.

    This function returns the same (rvec, tvec) derived from Vicon drone pose.

    Frame convention (empirically verified):
        R_stored = drone_rot[0..8] = R_world→body  (maps world vectors → body)
        R_cam_body = camera→body rotation (quaternion in drone_to_camera.json)
        R_world→camera = R_cam_body.T @ R_w2b
        tvec = –R_cam_body.T @ (R_w2b @ d_pos + t_cam_body)
    """
    import pandas as pd

    d_cols = ["drone_x", "drone_y", "drone_z"]
    if not all(c in row.index for c in d_cols) or any(pd.isna(row[c]) for c in d_cols):
        return None
    rot_cols = [f"drone_rot[{i}]" for i in range(9)]
    if not all(c in row.index for c in rot_cols):
        return None

    d_pos = np.array([row[c] for c in d_cols], dtype=np.float64)
    R_w2b = np.array([row[c] for c in rot_cols], dtype=np.float64).reshape(3, 3)

    R_w2c = R_cam_body.T @ R_w2b
    tvec  = -(R_cam_body.T @ (R_w2b @ d_pos + t_cam_body)).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R_w2c)
    return rvec, tvec


def vicon_gate_selection(
    row,
    K: np.ndarray,
    R_cam_body: np.ndarray,
    t_cam_body: np.ndarray,
    label_center_2d: np.ndarray | None = None,
    n_gates: int = 4,
) -> int | None:
    """Return the gate_id (1-based) that is most likely visible in the image.

    Uses the Vicon drone pose to project each gate's centre into the camera
    frame and returns the gate with the smallest distance to *label_center_2d*
    (or the closest to the image centre if no label is given) among those
    with g_cam.z > 0.
    """
    import pandas as pd

    d_cols = ["drone_x", "drone_y", "drone_z"]
    if not all(c in row.index for c in d_cols) or any(pd.isna(row[c]) for c in d_cols):
        return None
    d_pos = np.array([row[c] for c in d_cols], dtype=np.float64)

    rot_cols = [f"drone_rot[{i}]" for i in range(9)]
    if not all(c in row.index for c in rot_cols):
        return None
    R_w2b = np.array([row[c] for c in rot_cols], dtype=np.float64).reshape(3, 3)

    cx = float(K[0, 2])
    cy = float(K[1, 2])
    ref = label_center_2d if label_center_2d is not None else np.array([cx, cy])

    best_id, best_dist = None, float("inf")
    for gid in range(1, n_gates + 1):
        g_cols = [f"gate{gid}_int_{ax}" for ax in ("x", "y", "z")]
        if not all(c in row.index for c in g_cols) or any(pd.isna(row[c]) for c in g_cols):
            continue
        g_pos = np.array([row[c] for c in g_cols], dtype=np.float64)
        g_body = R_w2b @ (g_pos - d_pos)
        g_cam  = R_cam_body.T @ (g_body - t_cam_body)
        if g_cam[2] <= 0:
            continue
        u = K[0, 0] * g_cam[0] / g_cam[2] + K[0, 2]
        v = K[1, 1] * g_cam[1] / g_cam[2] + K[1, 2]
        dist = float(np.hypot(u - ref[0], v - ref[1]))
        if dist < best_dist:
            best_dist, best_id = dist, gid
    return best_id


# ---------------------------------------------------------------------------
# PnP solver
# ---------------------------------------------------------------------------

@dataclass
class PnPResult:
    rvec: np.ndarray       # (3,1) Rodrigues rotation
    tvec: np.ndarray       # (3,1) translation  (same unit as 3-D points)
    reproj_err: float      # mean reprojection error on inliers (px)
    n_inliers: int


def _undistort_pts(pts_2d: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Undistort (N,2) pixel coords → (N,2) normalised coords (no K applied).

    Uses cv2.undistortPoints which inverts the full Brown-Conrady model.
    The returned points satisfy  x_norm = (u - cx)/fx  in ideal pinhole space.
    """
    return cv2.undistortPoints(
        pts_2d.astype(np.float64).reshape(-1, 1, 2),
        K.astype(np.float64),
        dist.astype(np.float64),
    ).reshape(-1, 2)


def solve_pnp_inner(
    pts_2d: np.ndarray,
    pts_3d: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray | None = None,
    *,
    ransac_threshold_px: float = 4.0,
) -> PnPResult | None:
    """Estimate camera-from-gate pose from 4 inner-corner correspondences.

    Uses the undistort-then-solve strategy:
      1. Undistort 2-D points via Brown-Conrady model → normalised coords.
      2. Solve PnP in normalised space (K=I, dist=0) so the model is exact.
      3. Refine with Levenberg-Marquardt in original distorted pixel space.
      4. Report reprojection error in original distorted pixel space.

    This avoids numerical instability from passing large distortion coefficients
    directly to the RANSAC/P3P solver.

    Parameters
    ----------
    pts_2d : (4,2)  detected corners in distorted pixel coords, TL/TR/BR/BL
    pts_3d : (4,3)  matching world-frame marker positions, same order
    K      : (3,3)  camera intrinsic matrix
    dist   : (5,1)  Brown-Conrady distortion coefficients (zeros if None)
    """
    if pts_2d is None or pts_3d is None or len(pts_2d) < 4:
        return None
    K_f    = K.astype(np.float64)
    p3d    = pts_3d.astype(np.float64)
    p2d    = pts_2d.astype(np.float64)
    dist_f = np.zeros((5, 1), dtype=np.float64) if dist is None else dist.astype(np.float64)
    K_eye  = np.eye(3, dtype=np.float64)
    d_zero = np.zeros((5, 1), dtype=np.float64)

    # undistort to normalised (ideal pinhole) coordinates
    p2d_norm = _undistort_pts(p2d, K_f, dist_f)

    # convert RANSAC threshold from pixels to normalised units (approx)
    fx_mean = (K_f[0, 0] + K_f[1, 1]) / 2.0
    thr_norm = ransac_threshold_px / fx_mean

    # P3P RANSAC in normalised space
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        p3d, p2d_norm, K_eye, d_zero,
        reprojectionError=thr_norm, confidence=0.99,
        iterationsCount=300, flags=cv2.SOLVEPNP_P3P,
    )
    if not ok or inliers is None or len(inliers) < 3:
        ok2, rvec2, tvec2 = cv2.solvePnP(
            p3d, p2d_norm, K_eye, d_zero, flags=cv2.SOLVEPNP_SQPNP,
        )
        if not ok2:
            return None
        rvec, tvec = rvec2, tvec2
        inliers = np.arange(4).reshape(-1, 1)

    # LM refinement in distorted pixel space (uses full K + dist model)
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            p3d, p2d, K_f, dist_f, rvec, tvec,
        )
    except cv2.error:
        pass  # refinement unavailable or diverged — keep initial result

    # reprojection error in original distorted pixel space
    idx = inliers.flatten()
    proj, _ = cv2.projectPoints(p3d[idx], rvec, tvec, K_f, dist_f)
    reproj_err = float(np.mean(
        np.linalg.norm(proj.reshape(-1, 2) - p2d[idx], axis=1)
    ))
    return PnPResult(rvec=rvec, tvec=tvec,
                     reproj_err=reproj_err, n_inliers=len(idx))


# ---------------------------------------------------------------------------
# error metrics
# ---------------------------------------------------------------------------

def translation_error(tvec1: np.ndarray, tvec2: np.ndarray) -> float:
    return float(np.linalg.norm(tvec1.flatten() - tvec2.flatten()))


def rotation_error_deg(rvec1: np.ndarray, rvec2: np.ndarray) -> float:
    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(rvec2)
    cos_angle = np.clip((np.trace(R1 @ R2.T) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def default_camera_matrix(img_w: int = 640, img_h: int = 480) -> np.ndarray:
    """Fallback pinhole K when no calibration file is available."""
    scale = img_w / 1280.0
    fx = fy = 820.0 * scale
    return np.array([[fx, 0, img_w/2], [0, fy, img_h/2], [0, 0, 1]], dtype=np.float64)
