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
) -> FlightBenchmark:
    """Run GateNet + QuAdGate on every ``stride``-th frame of one flight.

    The optional ``<flight>_cam_ts_sync.csv`` (with ``drone_x/y/z`` and
    ``gateN_int_x/y/z`` columns) enables a distance-sliced report; if absent,
    distances are NaN and the slicing step is skipped by the caller.
    """
    cam_dir, lbl_dir, csv_path = find_flight_paths(flight_root)
    flight_name = os.path.basename(os.path.normpath(str(flight_root)))

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

    t0 = time.perf_counter()
    for fname in sampled:
        stem = os.path.splitext(fname)[0]
        img_path = os.path.join(cam_dir, fname)
        lbl_path = os.path.join(lbl_dir, stem + ".txt")

        try:
            img_rgb, pred = predictor.predict_image(img_path)
        except FileNotFoundError:
            # OpenCV occasionally fails to decode flight jpgs; skip rather
            # than abort the whole benchmark.
            continue

        # Use GT corners as the QuAdGate prior. This is a benchmarking
        # convenience — at deployment the prior would come from the EKF /
        # previous-frame estimate. Using GT measures the segmentation +
        # corner refinement quality without confounding from prior drift.
        # Only the FIRST gate's priors are passed; multi-gate scenes are
        # implicitly evaluated against the closest-matching GT gate by
        # inner_corner_error below.
        gt_gates = parse_pixel_corners(lbl_path, predictor.img_size)
        detected = detect_corners(pred, gt_gates[0]) if gt_gates else None
        detected_mask.append(detected is not None)
        cerr = inner_corner_error(detected, gt_gates)
        corner_errors.append(cerr)

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
    )
