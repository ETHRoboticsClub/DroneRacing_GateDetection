"""End-to-end benchmark of GateNet + QuAdGate on a single flight."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from gate_detection.data.discovery import find_flight_paths
from gate_detection.data.labels import parse_pixel_corners
from gate_detection.inference.predictor import GatePredictor
from gate_detection.postprocessing.quadgate import detect_corners
from .metrics import inner_corner_error


# Outer-corner synthesis scale used to expand inner corners to the (8, 2)
# CORNER_TYPES layout that QuAdGate's prior expects. Matches
# ``data/labels.py:parse_pixel_corners(synth_outer=True)`` and
# ``scripts/benchmark_pnp.py:79-81`` so model-prior mode produces the same
# outer-corner geometry the GT-prior mode does.
_OUTER_SYNTH_SCALE = 1.4


def _build_prior_from_inner(inner_xy: np.ndarray) -> np.ndarray:
    """Expand ``(4, 2)`` inner corners → ``(8, 2)`` CORNER_TYPES prior.

    Outer corners are placed at ``centroid + 1.4 × (inner − centroid)``,
    matching the inner-only label conversion convention used elsewhere in
    the repo. Order: ``[outer_TL, outer_TR, outer_BR, outer_BL,
    inner_TL, inner_TR, inner_BR, inner_BL]`` — the head emits inner
    corners in TL/TR/BR/BL order so the same indexing applies to outers.
    """
    centroid = inner_xy.mean(axis=0)
    outer_xy = centroid + _OUTER_SYNTH_SCALE * (inner_xy - centroid)
    return np.vstack([outer_xy, inner_xy]).astype(np.float64)


@dataclass
class FlightSample:
    img_rgb: np.ndarray
    pred_prob: np.ndarray
    detected: np.ndarray | None
    gt_gates: list[np.ndarray]
    cerr: float | None
    fname: str


@dataclass
class FlightBenchmark:
    flight_name: str
    detected_mask: np.ndarray              # bool per evaluated frame
    corner_errors: list[float | None]      # one entry per frame
    distances: np.ndarray                  # NaN where no CSV / row missing
    visuals: list[FlightSample] = field(default_factory=list)
    fps: float = 0.0
    n_total: int = 0
    prior_mode: str = "gt"                 # which QuAdGate prior was used


# CORNER_TYPES order from data/labels.py:33-36 — must match parse_pixel_corners.
# Used by prior_mode="none" to build a default centred-gate prior.
def _default_prior(img_size: int, *, inner_frac: float = 0.125, outer_scale: float = 1.4) -> np.ndarray:
    """Build an (8, 2) default prior at the image centre.

    Layout matches CORNER_TYPES = [outer_TL, outer_TR, outer_BR, outer_BL,
                                   inner_TL, inner_TR, inner_BR, inner_BL]
    """
    cx = cy = img_size / 2.0
    inner_h = img_size * inner_frac          # default 48 px at 384²
    outer_h = inner_h * outer_scale          # 1.4× → ~67 px at 384²
    return np.array([
        [cx - outer_h, cy - outer_h],
        [cx + outer_h, cy - outer_h],
        [cx + outer_h, cy + outer_h],
        [cx - outer_h, cy + outer_h],
        [cx - inner_h, cy - inner_h],
        [cx + inner_h, cy - inner_h],
        [cx + inner_h, cy + inner_h],
        [cx - inner_h, cy + inner_h],
    ], dtype=np.float64)


def _drone_gate_distance(row, n_gates: int = 4) -> float:
    """Min Euclidean distance from drone to any of the (up to 4) tracked gates."""
    import pandas as pd  # local import — only needed when CSV is present

    dx, dy, dz = row["drone_x"], row["drone_y"], row["drone_z"]
    best = float("inf")
    for g in range(1, n_gates + 1):
        gx, gy, gz = f"gate{g}_int_x", f"gate{g}_int_y", f"gate{g}_int_z"
        if gx in row.index and not pd.isna(row[gx]):
            d = ((dx - row[gx]) ** 2 + (dy - row[gy]) ** 2 + (dz - row[gz]) ** 2) ** 0.5
            best = min(best, d)
    return best if best != float("inf") else float("nan")


def benchmark_flight(
    predictor: GatePredictor,
    flight_root: str | os.PathLike,
    *,
    stride: int = 30,
    n_visuals: int = 16,
    prior_mode: str = "gt",
    noisy_sigma_px: float = 30.0,
    seed: int = 42,
    keypoint_conf_threshold: float = 0.3,
) -> FlightBenchmark:
    """Run GateNet + QuAdGate on every ``stride``-th frame of one flight.

    The optional ``<flight>_cam_ts_sync.csv`` (with ``drone_x/y/z`` and
    ``gateN_int_x/y/z`` columns) enables a distance-sliced report; if absent,
    distances are NaN and the slicing step is skipped by the caller.

    ``prior_mode`` controls what is fed to QuAdGate as the prior corner set:
      - ``"gt"``        : ground-truth corners from the label file (the
                          original benchmarking convenience; over-states
                          blind detection).
      - ``"noisy"``     : GT corners + isotropic Gaussian noise with std
                          ``noisy_sigma_px`` per coordinate. Models EKF
                          prior drift.
      - ``"none"``      : a centred default prior independent of the label
                          — forces QuAdGate to do candidate-only matching
                          as it would in a deployed cold-start situation.
      - ``"model"``     : the predictor's keypoint head supplies the inner
                          corners; outer corners are synthesised at
                          1.4 × (inner − centroid). The deployment-relevant
                          number — no GT, no manual init.
      - ``"model_direct"``: same keypoint head, but skip QuAdGate refinement
                          entirely and report the model's corners as the
                          detected corners. Isolates the head's accuracy
                          from any QuAdGate amplification or damage.
      - ``"ekf"``       : previous frame's detected corners carried forward
                          as the current prior (the simplest motion model:
                          identity), with **hybrid fallback** — if QuAdGate
                          rejects the EKF prior (no match), retry with the
                          keypoint head's prior; if that also fails, fall
                          through to the centred default. Approximates a
                          deployed EKF + perception system: the EKF supplies
                          a tight prior when fresh, and the perception stack
                          recovers when the EKF state goes stale.
    Model and ekf modes require a checkpoint trained with
    ``model.keypoint_head=true``; a runtime check raises ``RuntimeError``
    otherwise — silent fall-through would mis-report 0 % detection on
    every frame.
    """
    valid_modes = {"gt", "noisy", "none", "model", "model_direct", "ekf"}
    if prior_mode not in valid_modes:
        raise ValueError(f"prior_mode must be one of {valid_modes}; got {prior_mode!r}")
    if prior_mode in {"model", "model_direct", "ekf"} and not getattr(predictor, "has_keypoint_head", False):
        raise RuntimeError(
            f"prior_mode={prior_mode!r} requires a checkpoint trained with "
            "model.keypoint_head=true; the loaded checkpoint has no keypoint head."
        )

    cam_dir, lbl_dir, csv_path = find_flight_paths(flight_root)
    flight_name = os.path.basename(os.path.normpath(str(flight_root)))

    # Deterministic noise across runs — uses one np.random.RandomState seeded
    # at function entry so re-running with the same seed gives the same per-
    # frame jitter (CLAUDE.md §0: "every experimental claim needs a fixed seed").
    rng = np.random.RandomState(seed)
    img_size = predictor.img_size
    none_prior = _default_prior(img_size) if prior_mode == "none" else None

    csv_df = None
    if csv_path:
        import pandas as pd
        csv_df = pd.read_csv(csv_path, index_col="img_filename")

    all_imgs = sorted(f for f in os.listdir(cam_dir) if f.endswith(".jpg"))
    sampled = all_imgs[::stride]

    detected_mask: list[bool] = []
    corner_errors: list[float | None] = []
    distances: list[float] = []
    visuals: list[FlightSample] = []

    # State carried between frames in ekf mode — the previous frame's
    # detected (8, 2) corners become the next frame's prior. Stays
    # None until the first successful detection; never reset back to
    # None on a missed frame so a single failure doesn't lose state.
    last_detected: np.ndarray | None = None

    t0 = time.perf_counter()
    for fname in sampled:
        stem = os.path.splitext(fname)[0]
        img_path = os.path.join(cam_dir, fname)
        lbl_path = os.path.join(lbl_dir, stem + ".txt")

        # Single forward pass either way. predict_keypoints returns
        # (rgb, seg, kp_xy_or_None, kp_conf_or_None); for non-keypoint
        # checkpoints the kp_xy and kp_conf are None and the call falls
        # through to the same seg path predict_image takes.
        try:
            img_rgb, pred, kp_xy, kp_conf = predictor.predict_keypoints(img_path)
        except FileNotFoundError:
            # OpenCV occasionally fails to decode flight jpgs; skip rather
            # than abort the whole benchmark.
            continue

        # Build the QuAdGate prior according to prior_mode (see function
        # docstring for the trade-offs). Only the FIRST gate's priors are
        # passed; multi-gate scenes are implicitly evaluated against the
        # closest-matching GT gate by inner_corner_error below.
        # synth_outer=True so 17-col TII labels (which ship inner-only) parse
        # into the (8, 2) format QuAdGate's prior expects. Without this every
        # gate is silently skipped — see labels.py:parse_pixel_corners docstring.
        gt_gates = parse_pixel_corners(
            lbl_path, predictor.img_size, synth_outer=True,
        )
        prior = None
        skip_quadgate = False
        if prior_mode == "gt":
            prior = gt_gates[0] if gt_gates else None
        elif prior_mode == "noisy":
            prior = (gt_gates[0] + rng.randn(*gt_gates[0].shape) * noisy_sigma_px) if gt_gates else None
        elif prior_mode == "none":
            prior = none_prior
        elif prior_mode in {"model", "model_direct"}:
            # Confidence gate: ``min`` (not ``mean``) over the 4 channels
            # because _build_prior_from_inner uses every corner — one bad
            # corner pollutes the whole (8, 2) prior. A frame where any
            # inner corner has a flat heatmap (max < threshold) is treated
            # as "no detection" rather than emitting a garbage prior.
            if kp_xy is None or float(kp_conf.min()) < keypoint_conf_threshold:
                prior = None
            else:
                prior = _build_prior_from_inner(kp_xy)
            if prior_mode == "model_direct":
                # Skip QuAdGate; the model's corners ARE the detected output.
                skip_quadgate = True
        elif prior_mode == "ekf":
            # Carry-forward identity-motion EKF (handled below as a
            # multi-tier retry — set prior=None here, the ekf branch
            # below runs detect_corners up to twice).
            prior = None  # actual selection happens in the EKF retry block

        if prior_mode == "ekf":
            # Tier 1: previous frame's detected corners (the EKF state).
            detected = None
            if last_detected is not None:
                detected = detect_corners(pred, last_detected)
            # Tier 2: keypoint-head prior — runs when EKF was unavailable
            # OR when QuAdGate rejected the EKF prior. Confidence-gated
            # the same way as prior_mode="model".
            if detected is None and kp_xy is not None and float(kp_conf.min()) >= keypoint_conf_threshold:
                detected = detect_corners(pred, _build_prior_from_inner(kp_xy))
            # Tier 3: centred default. QuAdGate-from-cold-start; almost
            # always fails per report 06's "none" mode (0 % within 5 px),
            # but emits a deterministic last-resort attempt before giving up.
            if detected is None:
                detected = detect_corners(pred, _default_prior(img_size))
        elif skip_quadgate:
            detected = prior  # already (8, 2) in CORNER_TYPES order
        else:
            detected = detect_corners(pred, prior) if prior is not None else None
        detected_mask.append(detected is not None)
        cerr = inner_corner_error(detected, gt_gates)
        corner_errors.append(cerr)

        # Update EKF state only on success — missing a frame should not
        # reset the prior. (No-op for non-ekf modes; the variable is
        # only read inside the ekf branch above.)
        if detected is not None:
            last_detected = np.asarray(detected, dtype=np.float64)

        dist = float("nan")
        if csv_df is not None and fname in csv_df.index:
            dist = _drone_gate_distance(csv_df.loc[fname])
        distances.append(dist)

        if len(visuals) < n_visuals:
            visuals.append(FlightSample(
                img_rgb=img_rgb, pred_prob=pred,
                detected=detected, gt_gates=gt_gates,
                cerr=cerr, fname=fname,
            ))

    elapsed = time.perf_counter() - t0
    return FlightBenchmark(
        flight_name=flight_name,
        detected_mask=np.array(detected_mask, dtype=bool),
        corner_errors=corner_errors,
        distances=np.array(distances, dtype=float),
        visuals=visuals,
        fps=(len(corner_errors) / elapsed) if elapsed > 0 else float("nan"),
        n_total=len(detected_mask),
        prior_mode=prior_mode,
    )
