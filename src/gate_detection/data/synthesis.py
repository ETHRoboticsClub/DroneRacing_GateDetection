"""Offline synthetic-data augmentation.

For each curated 29-col sample, produce several augmented copies:

    1. Composite the warped gate region onto a random indoor background.
    2. Apply scale + rotation + translation + perspective warp.
    3. HSV jitter on gate and background independently.
    4. Add Gaussian noise to the final composite.
    5. Update label corner coordinates through every transform.

This is conceptually distinct from :mod:`.transforms` (online augmentation
inside ``GateDataset``) and runs as a one-off preprocessing step.
"""

from __future__ import annotations

import glob
import os
import random

import cv2
import numpy as np

from .labels import gates_to_lines, parse_gates
from gate_detection.paths import BG_ROOT


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------
def transform_points(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply a 3x3 perspective matrix to (N, 2) points."""
    ones = np.ones((len(pts), 1), dtype=np.float32)
    ph = np.hstack([pts, ones])      # (N, 3)
    t = (M @ ph.T).T                 # (N, 3)
    return t[:, :2] / t[:, 2:3]


def random_perspective_matrix(W: int, H: int, strength: float) -> np.ndarray:
    """Random perspective matrix for the (W, H) image corners."""
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst = src + np.random.uniform(
        -strength * min(W, H),  strength * min(W, H), src.shape
    ).astype(np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def random_affine_matrix(
    W: int, H: int,
    *, scale_range=(0.85, 1.15), angle_deg=15.0, translate_frac=0.08,
) -> np.ndarray:
    """Random 3x3 (homogeneous) affine matrix: rotation, uniform scale, translation."""
    scale = random.uniform(*scale_range)
    angle = random.uniform(-angle_deg, angle_deg)
    tx = random.uniform(-translate_frac, translate_frac) * W
    ty = random.uniform(-translate_frac, translate_frac) * H
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return np.vstack([M, [0, 0, 1]])


# ---------------------------------------------------------------------------
# Background pool (BGR; lazy-loaded from BG_ROOT, cached process-wide)
# ---------------------------------------------------------------------------
_bg_pool: list[str] = []


def _get_bg_pool(bg_root=BG_ROOT) -> list[str]:
    global _bg_pool
    if not _bg_pool:
        _bg_pool = glob.glob(os.path.join(str(bg_root), "*.jpg"))
        if not _bg_pool:
            raise FileNotFoundError(
                f"No backgrounds found in {bg_root}/. "
                "Run: python scripts/download_backgrounds.py"
            )
        random.shuffle(_bg_pool)
    return _bg_pool


def random_background(W: int, H: int) -> np.ndarray:
    pool = _get_bg_pool()
    img = cv2.imread(random.choice(pool))
    if img is None:
        return np.zeros((H, W, 3), np.uint8)
    return cv2.resize(img, (W, H))


# ---------------------------------------------------------------------------
# HSV jitter (BGR variant — synthesis uses BGR throughout for cv2.imwrite)
# ---------------------------------------------------------------------------
def hsv_jitter_bgr(
    img: np.ndarray,
    h_range: float = 10.0,
    s_range: float = 0.2,
    v_range: float = 0.2,
) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-h_range, h_range)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(1 - s_range, 1 + s_range), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(1 - v_range, 1 + v_range), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------------------
# Single-sample augmentation
# ---------------------------------------------------------------------------
def augment_sample(img_path: str, lbl_path: str, out_img: str, out_lbl: str) -> bool:
    """Augment one curated sample → write composite image + 29-col label.

    Returns True on success, False if the source is unreadable, has no
    29-col gates, or every gate falls completely out of bounds after warp.
    """
    img = cv2.imread(img_path)
    if img is None:
        return False
    H, W = img.shape[:2]

    gates = [g for g in parse_gates(lbl_path) if g["outer"] is not None]
    if not gates:
        return False

    bg = hsv_jitter_bgr(random_background(W, H))

    # --- Compose random rigid + perspective warp ---
    # Compose into one 3×3 matrix so the image and label corners go through
    # the SAME transform (otherwise they'd drift). Affine first (centred at
    # image centre), perspective on top.
    M_aff = random_affine_matrix(W, H)
    M_persp = random_perspective_matrix(W, H, strength=random.uniform(0.0, 0.12))
    M_full = M_persp @ M_aff

    warped_gate = cv2.warpPerspective(
        img, M_full, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    )
    # HSV jitter the gate independently of the background — done AFTER warp
    # so the warp's reflective border doesn't break colour stats.
    warped_gate = hsv_jitter_bgr(warped_gate)

    # --- Gate mask (union over all gates), with slight dilation ---
    combined_mask = np.zeros((H, W), np.uint8)
    for g in gates:
        # Prefer outer polygon; fall back to inner if not enough outer corners.
        vis_outer = [(pt, v) for pt, v in zip(g["outer"], g["outer_vis"]) if v > 0]
        if len(vis_outer) < 3:
            vis_outer = [(pt, v) for pt, v in zip(g["inner"], g["inner_vis"]) if v > 0]
        if len(vis_outer) < 3:
            continue
        pts_norm = np.array([p for p, _ in vis_outer], np.float32)
        pts_px = pts_norm * [W, H]
        pts_warped = transform_points(pts_px, M_full).astype(np.int32)
        cv2.fillPoly(combined_mask, [pts_warped], 255)

    # Dilate the union mask by ~7 px to hide compositing fringes (resampling
    # at the gate boundary makes the warped image fuzzy by ~1-2 px; without
    # dilation the bg leaks through as a thin halo around every gate).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined_mask = cv2.dilate(combined_mask, kernel)

    # --- Composite: gate on top of background ---
    # Soft alpha-style composite (mask is 0/255 here, so this is binary; the
    # /255.0 keeps the formula generic if we ever switch to a soft mask).
    mask3 = combined_mask[:, :, None] / 255.0
    composite = (warped_gate * mask3 + bg * (1 - mask3)).astype(np.uint8)

    # Noise added LAST so it covers the seam between gate and background
    # uniformly — adding it before composite would only roughen one side.
    sigma = random.uniform(0, 12)
    noise = np.random.normal(0, sigma, composite.shape).astype(np.float32)
    composite = np.clip(composite.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # --- Update label coords through M_full; drop gates with no visible inner ---
    def warp_corners(pts_norm: np.ndarray, vis_list: list[int]):
        new_pts, new_vis = [], []
        for (nx, ny), v in zip(pts_norm, vis_list):
            if v > 0:
                px = np.array([[nx * W, ny * H]], np.float32)
                tp = transform_points(px, M_full)[0]
                nx2, ny2 = tp[0] / W, tp[1] / H
                in_bounds = 0.0 <= nx2 <= 1.0 and 0.0 <= ny2 <= 1.0
                new_pts.append([nx2, ny2])
                new_vis.append(2 if in_bounds else 0)
            else:
                new_pts.append([0.0, 0.0])
                new_vis.append(0)
        return np.array(new_pts, np.float32), new_vis

    new_gates = []
    for g in gates:
        new_inner, new_inner_vis = warp_corners(g["inner"], g["inner_vis"])
        new_outer, new_outer_vis = warp_corners(g["outer"], g["outer_vis"])
        if not any(v > 0 for v in new_inner_vis):
            continue
        new_gates.append({
            **g,
            "inner": new_inner, "inner_vis": new_inner_vis,
            "outer": new_outer, "outer_vis": new_outer_vis,
        })

    if not new_gates:
        return False

    cv2.imwrite(out_img, composite)
    with open(out_lbl, "w") as f:
        f.write("\n".join(gates_to_lines(new_gates)) + "\n")
    return True
