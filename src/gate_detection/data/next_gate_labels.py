"""Next-gate-mask label generation for the report-15 retrain.

Goal: given a training sample (full-resolution-equivalent crop +
``<stem>.txt`` label), produce a (H, W) binary mask covering ONLY
the next gate the drone is approaching, as determined by Vicon
ground truth.

Procedure per sample:
  1. Parse the crop filename → flight name + original frame name.
  2. Load that flight's ``<flight>_cam_ts_sync.csv`` (cached).
  3. Load that flight's calibration (K, R_cam_body, t_cam_body) via
     ``gate_detection.evaluation.pnp.load_calibration`` (cached per
     track-family key).
  4. Project all visible gate centres into the image plane via
     ``project_next_gate_uv`` → pick the gate ahead of the drone
     closest to image centre. This is the "next gate ID".
  5. The label file's ``parse_gates`` returns the gates in file
     order — we don't know if that order matches the CSV's gate-ID
     order, so we MATCH by centroid: which label-file gate has its
     inner-corner centroid closest to the Vicon-projected (u, v)?
  6. Render a ring mask covering ONLY that matched gate
     (outer-polygon fill ⊖ inner-polygon erase, same as
     ``corners_to_mask`` but for one gate only).

Caching: the CSVs and calibration files are loaded LAZILY on first
access per worker (each DataLoader worker is a separate process, so
each builds its own cache). 28 flights × O(MB) CSV → at most ~150 MB
of RAM per worker, well within budget for 16 workers.

CLAUDE.md §0 — no plausible-looking code: this module is purely a
data pipeline. Reviewers should check (a) the centroid-match against
parse_gates ordering is correct, (b) the Vicon projection re-uses
``project_next_gate_uv`` so we don't duplicate the math, (c) on
missing CSV / Vicon-NaN / no-gate-visible the function returns an
all-zero mask AND logs a [WARN] per CLAUDE.md §6.

CLAUDE.md §1 — cited basis: this is the labelled-supervision
equivalent of MonoRace's (Bahnam et al. 2026, arXiv:2601.15222
§Adaptive Cropping) "target gate = nearest visible gate ahead of
drone" rule. They use the same rule at inference (via EKF); we use
it offline to GENERATE training labels.
"""
from __future__ import annotations

import os
import re
from typing import Optional

import cv2
import numpy as np


# Cache containers — keyed by flight name. Each DataLoader worker
# gets its own copy via fork; entries are populated on demand.
_CSV_CACHE: dict[str, "object"] = {}
_CALIB_CACHE: dict[str, tuple] = {}

# Reusable filename parser; matches both `flight-01p-ellipse` and
# `flight-01a-ellipse` prefixes, with anything after as the frame
# identifier.
_FLIGHT_RE = re.compile(r"^(flight-[0-9]+[apAP]?-[a-zA-Z0-9]+)__(.+)$")
_WARN_BUDGET = 5  # per worker, then suppress to keep logs sane


def _parse_crop_filename(crop_basename: str) -> Optional[tuple[str, str]]:
    """Extract (flight_name, original_frame_filename) from a SAM-crop's
    filename. Returns None if the filename doesn't match the
    flight-XX__<frame>.<ext> convention."""
    name, ext = os.path.splitext(crop_basename)
    m = _FLIGHT_RE.match(name)
    if m is None:
        return None
    flight = m.group(1)
    frame_stem = m.group(2)
    # CSV is indexed by the original .jpg filename, so re-attach .jpg
    return flight, f"{frame_stem}.jpg"


def _get_csv_row(
    flight: str,
    frame_filename: str,
    csv_root: str,
):
    """Return the per-frame CSV row for (flight, frame). None on miss.

    CSV layout: ``<csv_root>/<piloted|autonomous>/<flight>/<flight>_cam_ts_sync.csv``.
    """
    import pandas as pd

    if flight not in _CSV_CACHE:
        sub = "piloted" if "p-" in flight or flight.endswith("p") else "autonomous"
        csv_path = os.path.join(csv_root, sub, flight, f"{flight}_cam_ts_sync.csv")
        if not os.path.isfile(csv_path):
            # Try the other subdir as a fallback (some flight names
            # don't disambiguate p/a in the name).
            other = "autonomous" if sub == "piloted" else "piloted"
            alt = os.path.join(csv_root, other, flight, f"{flight}_cam_ts_sync.csv")
            if os.path.isfile(alt):
                csv_path = alt
            else:
                _CSV_CACHE[flight] = None
                return None
        try:
            _CSV_CACHE[flight] = pd.read_csv(csv_path, index_col="img_filename")
        except Exception as e:
            print(
                f"[WARN] next_gate_labels: failed to load {csv_path!r}: {e}; "
                f"caching None for flight={flight!r}.",
                flush=True,
            )
            _CSV_CACHE[flight] = None
    df = _CSV_CACHE[flight]
    if df is None:
        return None
    if frame_filename not in df.index:
        return None
    return df.loc[frame_filename]


def _get_calibration(flight: str, calib_dir: str):
    """Cached load of (K, dist, R_cam_body, t_cam_body) per flight."""
    if flight in _CALIB_CACHE:
        return _CALIB_CACHE[flight]
    from gate_detection.evaluation.pnp import load_calibration
    try:
        K, dist, R_cam_body, t_cam_body = load_calibration(flight, calib_dir)
        _CALIB_CACHE[flight] = (K, dist, R_cam_body, t_cam_body)
    except Exception as e:
        print(
            f"[WARN] next_gate_labels: load_calibration({flight!r}, "
            f"{calib_dir!r}) failed: {e}; caching None.",
            flush=True,
        )
        _CALIB_CACHE[flight] = None
    return _CALIB_CACHE[flight]


def find_next_gate_label_index(
    crop_basename: str,
    label_gates: list[dict],
    img_size: int,
    *,
    csv_root: str,
    calib_dir: str,
    warn_counter: list[int] | None = None,
) -> Optional[int]:
    """Return the index into ``label_gates`` of the Vicon-determined
    next gate, or ``None`` if any link in the lookup chain fails
    (no parseable filename, no CSV row, no calibration, projection
    behind camera / off-frame, or no label-gate-centroid match).

    The lookup is the same as ``render_next_gate_mask`` performs
    internally; factored out so the dataset's kp-target builder can
    reuse it without re-rendering a mask. CLAUDE.md §4 — single
    source of truth.
    """
    parsed = _parse_crop_filename(crop_basename)
    if parsed is None:
        _warn(warn_counter, f"could not parse crop filename {crop_basename!r}")
        return None
    flight, frame_filename = parsed

    row = _get_csv_row(flight, frame_filename, csv_root)
    if row is None:
        _warn(warn_counter, f"no CSV row for {crop_basename!r}")
        return None

    cal = _get_calibration(flight, calib_dir)
    if cal is None:
        _warn(warn_counter, f"no calibration for flight {flight!r}")
        return None
    K, _dist, R_cam_body, t_cam_body = cal

    from gate_detection.evaluation.pnp import project_next_gate_uv
    proj = project_next_gate_uv(row, K, R_cam_body, t_cam_body)
    if proj is None:
        _warn(warn_counter, f"no next-gate projection for {crop_basename!r}")
        return None
    _gid, u_full, v_full = proj

    full_w = 2.0 * float(K[0, 2])
    full_h = 2.0 * float(K[1, 2])
    u_scaled = u_full * img_size / max(full_w, 1.0)
    v_scaled = v_full * img_size / max(full_h, 1.0)

    matched_idx = _match_label_gate_by_centroid(
        label_gates, (u_scaled, v_scaled), img_size,
    )
    if matched_idx is None:
        _warn(
            warn_counter,
            f"no label-gate matched the projection for {crop_basename!r}",
        )
        return None
    return matched_idx


def _match_label_gate_by_centroid(
    label_gates: list[dict],
    target_uv: tuple[float, float],
    img_size: int,
) -> Optional[int]:
    """Among the label-file gates, return the index of the gate whose
    inner-corner centroid (in image pixels at ``img_size``) is closest
    to ``target_uv``. ``target_uv`` is the Vicon-projected next-gate
    centre at FULL-RES coordinates — we scale-match it to img_size
    space by assuming the label coords were normalised to [0, 1] of
    the original frame, which both parse_gates and the trainer use.
    """
    if not label_gates:
        return None
    u, v = target_uv
    best_idx, best_d = -1, float("inf")
    for i, g in enumerate(label_gates):
        # parse_gates returns inner coords normalised; scale to img_size.
        inner_px = g["inner"] * img_size
        cx, cy = float(inner_px[:, 0].mean()), float(inner_px[:, 1].mean())
        d = (cx - u) ** 2 + (cy - v) ** 2
        if d < best_d:
            best_d = d
            best_idx = i
    return best_idx if best_idx >= 0 else None


def render_next_gate_mask(
    crop_basename: str,
    label_gates: list[dict],
    img_size: int,
    *,
    csv_root: str,
    calib_dir: str,
    ring_thickness_px: int = 6,
    target_scale_to_img_size: bool = True,
    warn_counter: list[int] | None = None,
) -> tuple[np.ndarray, bool]:
    """Build a binary mask covering ONLY the Vicon-determined next gate.

    Returns ``(mask, has_signal)``. ``mask`` is a uint8
    ``(img_size, img_size)`` array. ``has_signal`` is True iff the
    label was constructed from a valid Vicon → projection → label
    match (i.e. the BCE loss should be applied to this sample);
    False iff any link in the chain failed (logged with [WARN] per
    CLAUDE.md §6, with a per-worker suppression budget).

    Caller (trainer) MUST honour ``has_signal``: applying BCE on a
    has_signal=False sample with an all-zero mask would push the
    head toward "predict 0 everywhere" on legitimate-gate frames,
    which is the bug reviewer round-1 Item H caught. The fix is to
    skip those samples in the loss (see ``trainer.py``'s per-batch
    masked BCE).
    """
    blank = np.zeros((img_size, img_size), dtype=np.uint8)
    # Single source of truth for the Vicon → projection → centroid
    # match — both this function and the dataset's kp-target builder
    # use ``find_next_gate_label_index``. CLAUDE.md §4.
    matched_idx_candidate = find_next_gate_label_index(
        crop_basename,
        label_gates,
        img_size=img_size,
        csv_root=csv_root,
        calib_dir=calib_dir,
        warn_counter=warn_counter,
    )
    if matched_idx_candidate is None:
        return blank, False
    g = label_gates[matched_idx_candidate]
    # Render the ring mask for ONLY this gate (outer fill ⊖ inner erase),
    # matching the same algorithm corners_to_mask uses but restricted
    # to one gate.
    def _visible_pts(corners: np.ndarray, vis: list[int]) -> list[list[int]]:
        pts = []
        for (x, y), v in zip(corners, vis):
            if v > 0:
                px = int(np.clip(x * img_size, 0, img_size - 1))
                py = int(np.clip(y * img_size, 0, img_size - 1))
                pts.append([px, py])
        return pts

    inner_pts = _visible_pts(g["inner"], g["inner_vis"])
    mask = blank.copy()
    if g["outer"] is not None:
        outer_pts = _visible_pts(g["outer"], g["outer_vis"])
        if len(outer_pts) >= 3:
            cv2.fillPoly(mask, [np.array(outer_pts, np.int32)], 1)
        if len(inner_pts) >= 3:
            cv2.fillPoly(mask, [np.array(inner_pts, np.int32)], 0)
    elif len(inner_pts) >= 3:
        # 17-col fallback: fill inner only, then erode to create a ring.
        cv2.fillPoly(mask, [np.array(inner_pts, np.int32)], 1)
        # Erode inward by ring_thickness_px to make a band, not a
        # filled rect — keeps the target consistent with the v2 model's
        # seg-mask training (which is also a ring).
        k = max(1, ring_thickness_px)
        kernel = np.ones((k, k), dtype=np.uint8)
        eroded = cv2.erode(mask, kernel)
        mask = (mask & ~eroded).astype(np.uint8)
    # If after all the geometry the mask is still empty (e.g. the
    # matched gate had < 3 visible corners), the sample has no usable
    # signal — return has_signal=False so the trainer's per-sample
    # mask skips it. Otherwise we'd push the head toward "no gate"
    # on a frame where Vicon says there IS one (caught by the
    # post-fix smoke test).
    if int(mask.sum()) == 0:
        _warn(warn_counter, f"matched gate has no visible corners in {crop_basename!r}")
        return mask, False
    return mask, True


def _warn(counter: list[int] | None, msg: str) -> None:
    """Emit [WARN] up to ``_WARN_BUDGET`` times then suppress."""
    if counter is None:
        print(f"[WARN] next_gate_labels: {msg}", flush=True)
        return
    counter[0] += 1
    if counter[0] <= _WARN_BUDGET:
        print(f"[WARN] next_gate_labels: {msg}", flush=True)
        if counter[0] == _WARN_BUDGET:
            print(
                f"[WARN] next_gate_labels: further warnings suppressed in "
                f"this worker (budget={_WARN_BUDGET}).",
                flush=True,
            )
