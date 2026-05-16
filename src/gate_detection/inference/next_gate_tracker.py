"""Next-gate-mask tracker (report 15, option #1).

This is the deployable runtime side of the next-gate-head retrain.
Instead of using the seg mask (which covers ALL visible gates) OR
the raw keypoint heatmap (which has peaks at every visible gate's
corners and no "which gate" signal), this tracker uses the model's
LEARNED next-gate-mask output as the gate-selection ROI.

Per frame:
  1. Run model → ``(seg, kp_heatmap, next_gate_mask)``.
  2. Binarise the next-gate-mask at ``mask_threshold``.
  3. Find the largest CC in that mask (the model said "this is the
     next gate"). With proper training the mask covers ONLY the
     next gate — there should be only one CC. We pick the largest
     defensively in case the head emits two blobs.
  4. Within that CC's bbox, take the per-channel argmax of the kp
     heatmap → (4, 2) corners.
  5. Verify spatial-order constraint.
  6. Apply the same strict-lock / handoff / fail-streak policy as
     :class:`SegCCTracker` but driven by the LEARNED mask, not the
     general seg mask.

Cited basis (CLAUDE.md §1):
  AlphaPilot Sec. III-C (Foehn et al. 2022, arXiv:2005.12813) assigns
  detected keypoints to gates via a learned permutation matrix over
  the seg mask — sidesteps the multi-modal-ambiguity problem. The
  next-gate-mask head is a more specialised version: not "which
  gate is this corner from?" but "where IS the next gate?". Same
  philosophy, simpler output.

  MonoRace (Bahnam et al. 2026, arXiv:2601.15222) crops via an EKF
  prior to "isolate the next gate". The next-gate-mask is the
  no-VIO analogue — the model learns to do the localisation
  internally from the multi-gate Vicon supervision.

CLAUDE.md §0 — lives depend on this. Conservative tracker: holds
the lock during noisy frames, drops cleanly when the head says
"no next gate", and refuses to emit a stale detection on a frame
the corners couldn't be extracted from.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from gate_detection.inference.seg_cc_tracker import (
    extract_quad_in_bbox,
    find_gate_components,
)
from gate_detection.inference.tracker import GateTrackerStatus


@dataclass
class NextGateMaskTrackerConfig:
    """Knobs for :class:`NextGateMaskTracker`. The defaults mirror
    SegCCTrackerConfig where the metric is the same; the mask-side
    thresholds are tuned to the BCE-supervised next-gate head."""

    mask_threshold: float = 0.5
    """Binarisation threshold on the next-gate-mask head.

    Restored to the standard 0.5 after the v3.2 retrain with Dice+BCE.
    Empirically verified on piloted held-out frames: max sigmoid is now
    0.73-0.95 on multi-gate scenes (vs v3.1's 0.05-0.23). The head's
    positive predictions are calibrated; the 0.5 cut works as designed.
    Future tuners may re-sweep on competition footage."""

    min_area: int = 100
    """Minimum CC area (px²) in the next-gate-mask. Lower than the
    SegCCTracker default (200) because the next-gate-mask only ever
    covers ONE gate (not all of them), so small distant gates have
    smaller masks. 100 px² ≈ 10×10 — still rejects salt-and-pepper
    noise from the head's sigmoid."""

    conf_threshold: float = 0.30
    """Per-channel keypoint peak floor (inside the mask's bbox).
    Same value as SegCCTracker / GateTracker — single source of truth."""

    match_radius_px: float = 60.0
    """Centroid-match radius for tracking across frames."""

    max_fail_streak: int = 3

    handoff_area_frac: float = 0.08
    """Same as SegCCTracker — set from the empirical analysis of
    ring areas at close approach (~10-15 % of image)."""

    handoff_persist_frames: int = 2


class NextGateMaskTracker:
    """Strict-lock tracker driven by the next-gate-mask head output."""

    def __init__(self, config: Optional[NextGateMaskTrackerConfig] = None):
        self.config = config or NextGateMaskTrackerConfig()
        self.locked_centroid: Optional[tuple[float, float]] = None
        self.fail_count: int = 0
        self.passed_count: int = 0

    def reset(self) -> None:
        self.locked_centroid = None
        self.fail_count = 0
        self.passed_count = 0

    def step(
        self,
        next_gate_mask: Optional[np.ndarray],
        kp_heatmap: Optional[np.ndarray],
    ) -> tuple[Optional[np.ndarray], GateTrackerStatus]:
        cfg = self.config
        status = GateTrackerStatus()

        if next_gate_mask is None:
            status.reason = "no_next_gate_head"
            return None, status
        if kp_heatmap is None:
            status.reason = "no_kp_head"
            return None, status

        H, W = next_gate_mask.shape
        img_area = float(H * W)

        # CCs of the next-gate mask — with a well-trained head there
        # should be exactly ONE CC (the next gate). We tolerate >1 in
        # case the head outputs noisy components, and pick the largest.
        ccs = find_gate_components(
            next_gate_mask.astype(np.float32),
            threshold=cfg.mask_threshold,
            min_area=cfg.min_area,
            # No max_area cap: by construction the next-gate-mask
            # only ever covers ONE gate, so a "huge" CC is just a
            # close-approach gate, not a Frankenstein.
            max_area_frac=1.0,
        )
        status.n_candidates = len(ccs)
        if not ccs:
            status.reason = "no_candidates"
            return self._soft_fail(status)

        if self.locked_centroid is None:
            chosen = ccs[0]
            status.reason = "fresh_pick"
        else:
            matched, matched_dist = None, float("inf")
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
            chosen = matched
            status.reason = "ok"

        # Extract corners within the chosen CC's bbox (per-channel argmax).
        quad, confs = extract_quad_in_bbox(
            kp_heatmap, chosen["bbox"], padding=4,
        )
        if quad is None:
            status.reason = "no_decode"
            return self._soft_fail(status)
        if float(confs.min()) < cfg.conf_threshold:
            status.reason = "low_conf"
            return self._soft_fail(status)

        # Diagnostics
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

        # Commit
        self.locked_centroid = chosen["centroid"]
        self.fail_count = 0

        # Handoff
        area_frac = chosen["area"] / img_area
        if area_frac >= cfg.handoff_area_frac:
            self.passed_count += 1
            if self.passed_count >= cfg.handoff_persist_frames:
                status.handoff_triggered = True
                status.reason = "handoff"
                self.locked_centroid = None
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
            self.fail_count = 0
            self.passed_count = 0
        return None, status
