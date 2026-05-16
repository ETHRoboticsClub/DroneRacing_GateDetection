"""Seg-CC based gate tracker (report 14, option #2).

The keypoint-only tracker repeatedly picks the wrong gate in multi-gate
scenes because the keypoint heatmap is multimodal by design (the model
was trained with max-across-gates targets; see
``losses/keypoint.py:121-127``). The "which gate" signal was
deliberately removed from the keypoint loss to avoid contradictory
gradients (report 09's failed attempt).

The **segmentation** mask does not have that ambiguity. Each gate is a
visually distinct ring of "gate"-class pixels — two visible gates =
two connected components in the binary mask. The LARGEST CC is the
closest gate (the next one to fly through).

Pipeline per frame:
  1. Binarise ``seg_prob > seg_threshold``.
  2. ``cv2.connectedComponentsWithStats`` → list of CCs.
  3. Drop CCs with area < ``min_area`` (noise).
  4. Pick a CC (largest on fresh start; matched-by-centroid otherwise).
  5. Within the chosen CC's bbox, take the **per-channel peak** of the
     keypoint heatmap as the four corners. Verify the same spatial-
     order constraint coherent_quad_decode uses.
  6. Emit the (4, 2) quad and a status dict.

Strict-lock policy (mirrors :class:`GateTracker`):
  - First frame: lock to largest CC.
  - Subsequent frames: stay locked unless (a) match disappears for
    ``max_fail_streak`` frames OR (b) another CC's area exceeds the
    locked CC's by ``switch_area_margin``× OR (c) handoff fires
    (chosen CC fills ≥ ``handoff_area_frac`` of the image).

CLAUDE.md §1 — research basis:
  AlphaPilot (Foehn et al. 2022, arXiv:2005.12813) Sec. III-C assigns
  detected keypoints to gates via a learned permutation matrix over
  the mask, sidestepping the same multimodal-ambiguity problem.
  We use the simpler "peak-in-bbox" rule because we only need to
  pick ONE gate's corners per frame (the locked one).

CLAUDE.md §0 — lives depend on this. This is option #2 in the
report 14 next-steps list; the user requested it after the
keypoint-only strict-lock tracker plateaued at 81-83 % fire-rate
with persistent multi-gate failure cases on flight-06p-ellipse.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from gate_detection.inference.tracker import GateTrackerStatus


# ---------------------------------------------------------------------------
# Free functions: CC enumeration + per-channel peak extraction
# ---------------------------------------------------------------------------
def find_gate_components(
    seg_prob: np.ndarray,
    *,
    threshold: float = 0.5,
    min_area: int = 200,
    max_area_frac: float = 0.40,
) -> list[dict]:
    """Connected components of the binary mask above ``threshold``.

    Returns a list of dicts ``{"label", "area", "centroid", "bbox"}``
    sorted DESCENDING by area (largest = closest gate first).
    bbox is ``(x0, y0, w, h)`` in image pixels.

    Filters: CCs with area outside ``[min_area, max_area_frac × H × W]``
    are dropped (reviewer round-3 Item F — merged-gates Frankensteins).
    """
    binary = (seg_prob > threshold).astype(np.uint8)
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8,
    )
    H, W = seg_prob.shape
    max_area = int(max_area_frac * H * W)
    out: list[dict] = []
    for lbl in range(1, n):  # 0 is background
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if area > max_area:
            # Likely two-or-more gates merged via 8-connectivity.
            # Don't pick this — would re-introduce Frankenstein-quad
            # extraction over the union bbox. Silent skip is fine
            # here; caller logs aggregate counts.
            continue
        out.append({
            "label": lbl,
            "area": area,
            "centroid": (
                float(centroids[lbl, 0]),
                float(centroids[lbl, 1]),
            ),
            "bbox": (
                int(stats[lbl, cv2.CC_STAT_LEFT]),
                int(stats[lbl, cv2.CC_STAT_TOP]),
                int(stats[lbl, cv2.CC_STAT_WIDTH]),
                int(stats[lbl, cv2.CC_STAT_HEIGHT]),
            ),
        })
    out.sort(key=lambda d: -d["area"])
    return out


def _peak_in_window(
    channel: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    padding: int = 8,
) -> Optional[tuple[float, float, float]]:
    """Argmax of ``channel`` within the (padded) bbox. Returns
    ``(x, y, peak_value)`` or ``None`` on a degenerate window."""
    H, W = channel.shape
    x0, y0, w, h = bbox
    x0p = max(0, x0 - padding)
    y0p = max(0, y0 - padding)
    x1p = min(W, x0 + w + padding)
    y1p = min(H, y0 + h + padding)
    if x1p <= x0p or y1p <= y0p:
        return None
    win = channel[y0p:y1p, x0p:x1p]
    if win.size == 0:
        return None
    flat_idx = int(win.argmax())
    py = flat_idx // win.shape[1]
    px = flat_idx % win.shape[1]
    return float(x0p + px), float(y0p + py), float(win[py, px])


def extract_quad_in_bbox(
    kp_heatmap: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    padding: int = 8,
) -> tuple[Optional[np.ndarray], np.ndarray]:
    """For each of the K=4 keypoint channels (TL, TR, BR, BL), take
    the argmax WITHIN the (padded) bbox. Returns ``(quad_or_None,
    confs)``. Quad is ``None`` if any channel has no peak in range,
    OR if the spatial-order constraint (TR.x > TL.x, BL.y > TL.y,
    BR.x > BL.x, BR.y > TR.y) is violated. ``confs`` is the per-
    channel peak value (also returned on None — caller may want it
    for diagnostics)."""
    if kp_heatmap.shape[0] != 4:
        raise ValueError(
            f"extract_quad_in_bbox expects K=4 channels; got "
            f"{kp_heatmap.shape[0]}"
        )
    corners = np.zeros((4, 2), dtype=np.float32)
    confs = np.zeros(4, dtype=np.float32)
    for k in range(4):
        peak = _peak_in_window(kp_heatmap[k], bbox, padding=padding)
        if peak is None:
            return None, confs
        corners[k, 0] = peak[0]
        corners[k, 1] = peak[1]
        confs[k] = peak[2]
    TL, TR, BR, BL = corners
    if not (TR[0] > TL[0] and BL[1] > TL[1]
            and BR[0] > BL[0] and BR[1] > TR[1]):
        return None, confs
    return corners, confs


# ---------------------------------------------------------------------------
# Tracker config + class
# ---------------------------------------------------------------------------
@dataclass
class SegCCTrackerConfig:
    """Knobs for :class:`SegCCTracker`. Defaults align with
    GateTrackerConfig where the metric matches; the area-based knobs
    are new and use diagonal²-equivalents for portability."""

    seg_threshold: float = 0.5
    """Binarisation threshold on the seg probability map. Matches
    eval.threshold in the YAML configs — single source of truth."""

    min_area: int = 200
    """Minimum CC area (px²) to consider as a gate. 200 px² on a
    384×384 image is ≈ 14×14 — smaller than any plausible real gate
    even at maximum distance. Filters out salt-and-pepper noise."""

    max_area_frac: float = 0.40
    """Maximum CC area as a fraction of the image area. Any CC larger
    than this is treated as a 'merged-gates Frankenstein' (two gates
    overlap in image space → connectivity-8 merges them into one CC).
    The largest-CC rule would then pick this merged blob and run
    corner extraction over it, re-introducing the Frankenstein-quad
    problem. 0.40 = 40 % of image is a defensible cut: even a
    near-passage real gate's RING is ~10-15 % of image (the seg
    mask is the ring, not the inner rect); ≥40 % means two-or-more
    gates have merged. Reviewer-round-3 Item F."""

    conf_threshold: float = 0.30
    """Per-channel keypoint peak floor. If any of the 4 corner
    channels has its in-bbox argmax below this, the frame is a
    non-detection. Matches the value used elsewhere."""

    match_radius_px: float = 60.0
    """Centroid-match radius for tracking. A new-frame CC within
    this distance of the locked CC's prior centroid is considered
    the same gate. Beyond this radius the lock is treated as lost."""

    switch_area_margin: float = 1.69
    """Override: switch to a different CC if its area is at least
    ``switch_area_margin`` × the locked CC's area. 1.69 = 1.3² —
    the area equivalent of the diagonal-1.3 margin used by
    :class:`GateTracker`. A genuinely closer gate at 30 % larger
    diagonal has ~70 % larger area; this margin lets through that
    case while rejecting noise-level fluctuations."""

    max_fail_streak: int = 3
    """Consecutive no-match-or-no-decode frames before the lock is
    dropped. 3 frames at 60 FPS ≈ 50 ms — survives brief occlusion
    but doesn't carry a vanished gate forever."""

    handoff_area_frac: float = 0.08
    """The chosen CC's area as a fraction of the full image area —
    threshold for "gate about to be passed". CORRECTED from a
    previously-wrong 0.30 default after reviewer round-3 Item E:
    a real-gate RING (outer rect ⊖ inner rect) at close approach
    is only ~10-15 % of the image, NOT 30 % (that figure would be
    correct for a filled gate-shape, but the seg mask we train on
    is the ring). With handoff_area_frac=0.30 the trigger never
    fired in the 500-frame held-out bench (handoffs=0). 0.08 fires
    on gates that are ~1.5-2 m away and growing; pair with
    ``handoff_persist_frames`` debounce."""

    handoff_persist_frames: int = 2
    """Debounce on the handoff trigger — must see ``handoff_area_frac``
    for this many consecutive frames before firing."""


class SegCCTracker:
    """Strict-lock tracker over segmentation-mask connected components.

    Usage::

        tracker = SegCCTracker()
        for frame in stream:
            rgb, seg_prob, kp_heatmap = predictor.predict_keypoint_heatmap(frame)
            quad, status = tracker.step(seg_prob, kp_heatmap)
            if quad is not None:
                ...  # use the chosen gate's quad
    """

    def __init__(self, config: Optional[SegCCTrackerConfig] = None):
        self.config = config or SegCCTrackerConfig()
        self.locked_centroid: Optional[tuple[float, float]] = None
        self.locked_area: Optional[int] = None
        self.fail_count: int = 0
        self.passed_count: int = 0

    def reset(self) -> None:
        self.locked_centroid = None
        self.locked_area = None
        self.fail_count = 0
        self.passed_count = 0

    def step(
        self,
        seg_prob: np.ndarray,
        kp_heatmap: Optional[np.ndarray],
    ) -> tuple[Optional[np.ndarray], GateTrackerStatus]:
        cfg = self.config
        status = GateTrackerStatus()

        if kp_heatmap is None:
            # No keypoint head on this ckpt — can't extract sub-pixel
            # corners. Caller should fall back to a classical pipeline.
            status.reason = "no_kp_head"
            return None, status

        H, W = seg_prob.shape
        img_area = float(H * W)

        ccs = find_gate_components(
            seg_prob,
            threshold=cfg.seg_threshold,
            min_area=cfg.min_area,
            max_area_frac=cfg.max_area_frac,
        )
        status.n_candidates = len(ccs)
        if not ccs:
            status.reason = "no_candidates"
            return self._soft_fail(status)

        # ----- pick: largest if fresh; matched-by-centroid otherwise --
        if self.locked_centroid is None:
            chosen = ccs[0]
            status.reason = "fresh_pick"
        else:
            matched = None
            matched_dist = float("inf")
            for cc in ccs:
                d = float(np.hypot(
                    cc["centroid"][0] - self.locked_centroid[0],
                    cc["centroid"][1] - self.locked_centroid[1],
                ))
                if d < matched_dist:
                    matched_dist = d
                    matched = cc
            if matched is None or matched_dist > cfg.match_radius_px:
                status.reason = "lock_lost"
                return self._soft_fail(status)
            status.picked_proximity = matched_dist

            # Override: another CC has area >= margin × matched.area
            biggest_other = None
            for cc in ccs:
                if cc is matched:
                    continue
                if cc["area"] >= cfg.switch_area_margin * matched["area"]:
                    if biggest_other is None or cc["area"] > biggest_other["area"]:
                        biggest_other = cc
            if biggest_other is not None:
                chosen = biggest_other
                status.reason = "switch_to_bigger"
            else:
                chosen = matched
                status.reason = "ok"

        # ----- extract corners within the chosen CC's bbox -----------
        # Padding tightened from 8→4 (reviewer round-3 Item A): with
        # padding=8 the search window can encroach on a nearby gate's
        # corner peaks and silently extract the WRONG gate's corners
        # (caught by my own synthetic smoke test before reviewer
        # codified the concern). 4 px still allows for slight
        # CC-edge bleed (the seg ring's outer edge isn't always
        # pixel-perfect against the actual gate corner) but is small
        # enough that a gate ~8+ px away from the chosen CC's bbox
        # can't cross-contaminate.
        quad, confs = extract_quad_in_bbox(
            kp_heatmap, chosen["bbox"], padding=4,
        )
        if quad is None:
            status.reason = "no_decode"
            return self._soft_fail(status)
        if float(confs.min()) < cfg.conf_threshold:
            status.reason = "low_conf"
            return self._soft_fail(status)

        # ----- diagnostics ------------------------------------------
        status.picked_diagonal = 0.5 * (
            float(np.linalg.norm(quad[2] - quad[0])) +
            float(np.linalg.norm(quad[3] - quad[1]))
        )
        TL, TR, BR, BL = quad
        top_w = float(TR[0] - TL[0])
        bot_w = float(BR[0] - BL[0])
        left_h = float(BL[1] - TL[1])
        right_h = float(BR[1] - TR[1])
        status.picked_rect_score = abs(top_w - bot_w) + abs(left_h - right_h)

        # ----- commit ------------------------------------------------
        self.locked_centroid = chosen["centroid"]
        self.locked_area = chosen["area"]
        self.fail_count = 0

        # ----- handoff debounce -------------------------------------
        area_frac = chosen["area"] / img_area
        if area_frac >= cfg.handoff_area_frac:
            self.passed_count += 1
            if self.passed_count >= cfg.handoff_persist_frames:
                status.handoff_triggered = True
                status.reason = "handoff"
                self.locked_centroid = None
                self.locked_area = None
                self.passed_count = 0
                self.fail_count = 0
        else:
            self.passed_count = 0

        return quad, status

    def _soft_fail(
        self, status: GateTrackerStatus,
    ) -> tuple[None, GateTrackerStatus]:
        self.fail_count += 1
        status.fail_count = self.fail_count
        if self.fail_count >= self.config.max_fail_streak:
            self.locked_centroid = None
            self.locked_area = None
            self.fail_count = 0
            self.passed_count = 0
        return None, status
