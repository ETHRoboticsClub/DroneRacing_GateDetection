"""QuAdGate: classical sub-pixel gate-corner extraction from a binary mask.

Pipeline
--------
1. Derotate the predicted mask using the estimated drone roll so that gate
   edges align with image axes.
2. Detect line segments with OpenCV's LSD.
3. Extend each segment by ×5/3 so that near-but-not-touching edges intersect.
4. Compute pairwise intersections of the extended lines → corner candidates.
5. Sample a 4-bit local descriptor (mask values in TL/TR/BR/BL of each
   candidate) and match against canonical prior descriptors per corner type.
6. Fit a 4-DOF (rotation + uniform scale + translation) RANSAC affine from
   priors → matched candidates and apply it to all 8 priors to fill in missed
   corners.
7. Un-derotate back to the original frame.

Conventions
-----------
* Image coordinates: ``+x = right``, ``+y = down``.
* After derotation, ``world-up`` aligns with image-up, which lets us encode
  per-corner descriptors as fixed bit patterns (see :data:`PRIOR_DESCRIPTORS`).
"""

from __future__ import annotations

import cv2
import numpy as np

from gate_detection.data.labels import CORNER_TYPES


# Canonical 4-bit local descriptors (TL, TR, BR, BL of each candidate).
# Bits encode whether the diagonal sample at (-h,-v), (+h,-v), (+h,+v),
# (-h,+v) lies inside the ring mask.
PRIOR_DESCRIPTORS: dict[str, np.ndarray] = {
    "outer_TL": np.array([0, 0, 1, 0]),
    "outer_TR": np.array([0, 0, 0, 1]),
    "outer_BR": np.array([1, 0, 0, 0]),
    "outer_BL": np.array([0, 1, 0, 0]),
    "inner_TL": np.array([1, 1, 0, 1]),
    "inner_TR": np.array([1, 1, 1, 0]),
    "inner_BR": np.array([0, 1, 1, 1]),
    "inner_BL": np.array([1, 0, 1, 1]),
}


# ---------------------------------------------------------------------------
# Geometric primitives (private)
# ---------------------------------------------------------------------------
def _derotate(mask: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the mask in-plane so world-up aligns with image-up. Returns (rot, M)."""
    h, w = mask.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    rot = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
    return rot, M


def _detect_lines(mask_u8: np.ndarray) -> np.ndarray:
    """Run LSD on a binary mask. Returns ``(N, 4)`` array of [x1, y1, x2, y2]."""
    # LSD args (cv2.LSD_REFINE_STD): (scale=0.8, sigma_scale=0.8, quant=25.0,
    # ang_th=30.0). Tighter ang_th (vs the OpenCV default 22.5) suppresses
    # spurious diagonal segments along the staircased polygon edges.
    lsd = cv2.createLineSegmentDetector(
        cv2.LSD_REFINE_STD, 0.8, 0.8, 25.0, 30.0,
    )
    lines, *_ = lsd.detect(mask_u8)
    if lines is None or len(lines) == 0:
        return np.empty((0, 4), dtype=np.float32)
    return lines.reshape(-1, 4).astype(np.float32)


def _extend_lines(lines: np.ndarray, factor: float = 5 / 3) -> np.ndarray:
    """Symmetrically extend each segment so total length grows by ``factor``.

    LSD usually under-shoots gate edges by a small margin (the predicted mask
    has soft endpoints). Without extension, edges that *should* meet at a
    corner intersect outside the segments themselves, so step 4 of the
    pipeline would miss the candidate. ``5/3`` matches the MonoRace paper.
    """
    cx = (lines[:, 0] + lines[:, 2]) / 2
    cy = (lines[:, 1] + lines[:, 3]) / 2
    hx = (lines[:, 2] - lines[:, 0]) / 2
    hy = (lines[:, 3] - lines[:, 1]) / 2
    return np.stack(
        [cx - factor * hx, cy - factor * hy, cx + factor * hx, cy + factor * hy],
        axis=1,
    )


def _clip_lines(lines: np.ndarray, w: int, h: int) -> np.ndarray:
    """Liang-Barsky clip of each segment to ``[0, w) × [0, h)``."""
    clipped = []
    for x1, y1, x2, y2 in lines:
        dx, dy = x2 - x1, y2 - y1
        t0, t1 = 0.0, 1.0
        for p, q in [(-dx, x1), (dx, (w - 1) - x1),
                     (-dy, y1), (dy, (h - 1) - y1)]:
            if p == 0:
                # Segment is parallel to this edge; q < 0 means it's entirely
                # outside the half-plane → flag for rejection (sentinel > t1).
                if q < 0:
                    t0 = 2.0
                    break
            else:
                t = q / p
                if p < 0:
                    t0 = max(t0, t)
                else:
                    t1 = min(t1, t)
        if t0 <= t1:
            clipped.append([x1 + t0 * dx, y1 + t0 * dy,
                            x1 + t1 * dx, y1 + t1 * dy])
    return np.array(clipped, dtype=np.float32) if clipped else np.empty((0, 4), dtype=np.float32)


def _intersect_lines(lines: np.ndarray, img_shape: tuple[int, int]) -> list[tuple]:
    """Pairwise intersections of all line pairs (treated as infinite).

    Returns ``[(pt, d1, d2), ...]`` with unit direction vectors of the two
    contributing lines. Only intersections inside the image are kept.
    """
    h, w = img_shape
    out: list[tuple] = []
    n = len(lines)
    for i in range(n):
        x1, y1, x2, y2 = lines[i]
        for j in range(i + 1, n):
            x3, y3, x4, y4 = lines[j]
            # Standard 2D line-line intersection determinant; |denom| ≈ 0
            # means the two lines are (near-)parallel and have no unique
            # intersection — skip rather than divide.
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-8:
                continue
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            px = x1 + t * (x2 - x1)
            py = y1 + t * (y2 - y1)
            if not (0 <= px < w and 0 <= py < h):
                continue
            # Direction vectors are kept (and normalised) so the descriptor
            # sampler can distinguish "horizontal-aligned" from
            # "vertical-aligned" edges at this candidate corner.
            d1 = np.array([x2 - x1, y2 - y1], dtype=np.float64)
            d2 = np.array([x4 - x3, y4 - y3], dtype=np.float64)
            d1 /= np.linalg.norm(d1) + 1e-12
            d2 /= np.linalg.norm(d2) + 1e-12
            out.append((np.array([px, py]), d1, d2))
    return out


def _sample_descriptor(
    mask: np.ndarray, pt: np.ndarray, d1: np.ndarray, d2: np.ndarray,
    step: float = 5.0,
) -> np.ndarray:
    """4-bit descriptor: mask values at TL, TR, BR, BL of ``pt``.

    The two line directions ``d1, d2`` are split into a horizontal-aligned
    direction (``h_dir`` → +x) and a vertical-aligned direction (``v_dir`` → +y);
    descriptor positions are then ``±step·h_dir ± step·v_dir`` around ``pt``.
    """
    h, w = mask.shape
    # Whichever direction is more x-aligned becomes "horizontal"; the other
    # becomes "vertical". After this branch we know h_dir lies along ±x and
    # v_dir along ±y, so canonicalise their sign so h_dir → +x, v_dir → +y.
    # That makes the (TL, TR, BR, BL) sampling pattern match the image axes
    # regardless of which order LSD reported the two edges in.
    if abs(d1[0]) >= abs(d2[0]):
        h_dir, v_dir = d1.copy(), d2.copy()
    else:
        h_dir, v_dir = d2.copy(), d1.copy()
    if h_dir[0] < 0:
        h_dir = -h_dir
    if v_dir[1] < 0:
        v_dir = -v_dir

    def sample(off):
        p = pt + off
        xi, yi = int(round(p[0])), int(round(p[1]))
        if 0 <= xi < w and 0 <= yi < h:
            return 1.0 if mask[yi, xi] > 0 else 0.0
        return 0.0

    return np.array([
        sample(-step * h_dir - step * v_dir),  # TL
        sample(+step * h_dir - step * v_dir),  # TR
        sample(+step * h_dir + step * v_dir),  # BR
        sample(-step * h_dir + step * v_dir),  # BL
    ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def detect_corners_debug(
    mask: np.ndarray,
    priors: np.ndarray,
    attitude_angle: float = 0.0,
    max_dist: float = 100.0,
    ransac_threshold: float = 5.0,
    max_translation: float = 150.0,
) -> dict:
    """Run the QuAdGate pipeline, returning every intermediate output.

    Args:
        mask: H×W float prediction in [0, 1] or {0, 1}; binarised internally at 0.5.
        priors: ``(8, 2)`` array of prior corner positions in :data:`CORNER_TYPES`
                order, in the original (un-derotated) image frame.
        attitude_angle: degrees to rotate the mask so world-up = image-up.
        max_dist: max prior↔candidate distance (px) to consider a match.
        ransac_threshold: RANSAC reprojection threshold for the affine fit.
        max_translation: reject the affine if |tx| or |ty| exceeds this.

    Returns:
        dict with keys ``mask_rot, M_rot, priors_rot, lines, lines_ext,
        cand_pts, cand_descs, matches, M_affine, result``.  ``result`` is
        the final ``(8, 2)`` corners in the original frame, or ``None`` on
        any pipeline failure.
    """
    out = dict(mask_rot=None, M_rot=None, priors_rot=None,
               lines=None, lines_ext=None, cand_pts=None, cand_descs=None,
               matches={}, M_affine=None, result=None)

    # Binarise the (possibly soft) sigmoid output so LSD has clean edges.
    mask_bin = (mask > 0.5).astype(np.uint8) * 255

    # 1. Derotate — also rotate the priors with the same matrix so they live
    # in the same frame as the LSD candidates we're about to detect.
    mask_rot, M_rot = _derotate(mask_bin, attitude_angle)
    out["mask_rot"] = mask_rot
    out["M_rot"] = M_rot

    priors_h = np.hstack([priors.astype(np.float64), np.ones((len(priors), 1))])
    priors_rot = (M_rot @ priors_h.T).T
    out["priors_rot"] = priors_rot

    h, w = mask_rot.shape

    # 2. LSD lines + extension. Need ≥2 lines to ever produce an intersection.
    lines = _detect_lines(mask_rot)
    out["lines"] = lines
    if len(lines) < 2:
        return out
    lines_ext = _clip_lines(_extend_lines(lines), w, h)
    out["lines_ext"] = lines_ext

    # 3. Intersection candidates + descriptors
    candidates = _intersect_lines(lines_ext, (h, w))
    if not candidates:
        return out
    cand_pts = np.array([c[0] for c in candidates])
    cand_descs = np.array([_sample_descriptor(mask_rot, *c) for c in candidates])
    out["cand_pts"] = cand_pts
    out["cand_descs"] = cand_descs

    # 4. Match each prior to nearest descriptor-compatible candidate.
    # Note: a single candidate may be picked by more than one prior — that's
    # tolerated because RANSAC in step 5 will drop the duplicates as outliers.
    matched_priors, matched_cands = [], []
    matches: dict[str, np.ndarray] = {}
    for prior_idx, ctype in enumerate(CORNER_TYPES):
        prior_pt = priors_rot[prior_idx]
        prior_desc = PRIOR_DESCRIPTORS[ctype]

        # Two filters: (a) the candidate's local mask pattern matches the
        # corner type's canonical descriptor exactly, and (b) it lies within
        # max_dist of the prior. Among survivors, take the closest.
        desc_match = np.all(cand_descs == prior_desc, axis=1)
        dists = np.linalg.norm(cand_pts - prior_pt, axis=1)
        valid = np.where(desc_match & (dists <= max_dist))[0]
        if len(valid) == 0:
            continue
        best = valid[np.argmin(dists[valid])]
        matched_priors.append(prior_pt)
        matched_cands.append(cand_pts[best])
        matches[ctype] = cand_pts[best]
    out["matches"] = matches

    # estimateAffinePartial2D needs ≥3 correspondences; with fewer we can't
    # solve for rotation + scale + translation, so bail out.
    if len(matched_priors) < 3:
        return out

    # 5. RANSAC 4-DOF affine (translation + rotation + uniform scale)
    src = np.array(matched_priors, dtype=np.float32)
    dst = np.array(matched_cands,  dtype=np.float32)
    M_affine, _ = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold,
    )
    out["M_affine"] = M_affine
    if M_affine is None:
        return out
    # Sanity check: a sane refinement only nudges the priors by a few px.
    # A huge translation almost always means RANSAC locked onto the wrong
    # gate (e.g. another gate visible in the same frame).
    if abs(M_affine[0, 2]) > max_translation or abs(M_affine[1, 2]) > max_translation:
        return out

    # 6. Apply affine to all priors → refined positions in derotated frame.
    # Even priors whose descriptor match failed in step 4 get a refined
    # location here, which is the whole point of fitting a global affine.
    priors_rot_h = np.hstack([priors_rot, np.ones((len(priors_rot), 1))])
    refined_rot = (M_affine @ priors_rot_h.T).T

    # 7. Un-derotate to original image frame
    M_rot_inv = cv2.invertAffineTransform(M_rot)
    refined_h = np.hstack([refined_rot, np.ones((len(refined_rot), 1))])
    refined = (M_rot_inv @ refined_h.T).T

    # Reject if any corner falls outside the image
    if np.any(refined < 0) or np.any(refined[:, 0] >= w) or np.any(refined[:, 1] >= h):
        return out

    out["result"] = refined.astype(np.float32)
    return out


def detect_corners(
    mask: np.ndarray,
    priors: np.ndarray,
    attitude_angle: float = 0.0,
    max_dist: float = 100.0,
    ransac_threshold: float = 5.0,
    max_translation: float = 150.0,
) -> np.ndarray | None:
    """Run QuAdGate and return only the final ``(8, 2)`` corners (or ``None``)."""
    return detect_corners_debug(
        mask, priors, attitude_angle, max_dist, ransac_threshold, max_translation,
    )["result"]
