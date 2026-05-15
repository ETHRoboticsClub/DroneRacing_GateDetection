"""Measure false-positive rate on no-gate frames.

Definition: a frame is "no-gate" if its sibling .txt label file either
(a) does not exist, or (b) is zero-bytes, OR (c) has no parseable gate
lines. A model "fires" on the frame if predict_keypoints returns a
non-None kp_xy AND kp_conf.min() >= threshold across all 4 channels.

FP rate = #fired / #no-gate-frames-tested.

Caveats:
* Labellers may have missed real gates → some "no-gate" frames could
  still contain a gate. FP rate is therefore an UPPER BOUND on the
  true false-positive rate when measured against the labels alone.
* The model_direct path with multi-gate-target ckpt uses
  coherent_quad_decode and may return None (no valid spatial-ordered
  combo) — that's treated as "did not fire", which is correct.
* Frames are sampled at stride=10 from each flight's no-gate set to
  speed up the measurement; with 33 to 1660 no-gate frames per flight
  this gives 3-166 samples each.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from gate_detection.inference.predictor import GatePredictor
from gate_detection.data.labels import parse_gates


def find_no_gate_frames(flight_dir: str) -> list[tuple[str, str]]:
    """Return list of (img_path, reason) tuples for frames with no gate label.

    Reasons: "no_txt" (no sibling .txt), "empty_txt" (0-byte .txt),
    or "no_gate_line" (.txt parses to empty gates list).
    """
    cam_dir = os.path.join(flight_dir, [d for d in os.listdir(flight_dir) if d.startswith("camera_")][0])
    lbl_dir = os.path.join(flight_dir, [d for d in os.listdir(flight_dir) if d.startswith("label_")][0])

    out: list[tuple[str, str]] = []
    for fn in sorted(os.listdir(cam_dir)):
        if not fn.endswith(".jpg"):
            continue
        img_path = os.path.join(cam_dir, fn)
        lbl_path = os.path.join(lbl_dir, fn.replace(".jpg", ".txt"))
        if not os.path.exists(lbl_path):
            out.append((img_path, "no_txt"))
            continue
        if os.path.getsize(lbl_path) == 0:
            out.append((img_path, "empty_txt"))
            continue
        gates = parse_gates(lbl_path)
        if not gates:
            out.append((img_path, "no_gate_line"))
    return out


def measure_fp(
    checkpoint: str,
    flight_dir: str,
    threshold: float = 0.3,
    stride: int = 10,
    multi_gate_target_arg: bool = False,
) -> dict:
    """Run predict_keypoints on no-gate frames; count "fires".

    Returns dict with: n_no_gate (total), n_tested (sampled), n_fired,
    fp_rate, n_per_reason (dict of reason → count).
    """
    pred = GatePredictor(
        checkpoint, img_size=384, model_name="deeplabv3plus",
        encoder_name="resnet34", encoder_weights=None, dropout_p=0.1,
        keypoint_head=True, n_keypoints=4,
    )
    print(f"  loaded {os.path.basename(checkpoint)}; multi_gate_target={pred.multi_gate_target}")

    no_gate = find_no_gate_frames(flight_dir)
    print(f"  no-gate frames: {len(no_gate)}")
    if not no_gate:
        return {"n_no_gate": 0, "n_tested": 0, "n_fired": 0, "fp_rate": 0.0, "n_per_reason": {}, "examples_fired": []}

    sampled = no_gate[::stride]
    fired_examples: list[tuple[str, str, float]] = []  # (img_path, reason, kp_conf.min())
    fired_count = 0
    reason_counts: dict[str, int] = {}
    fired_conf_min_dist: list[float] = []
    not_fired_conf_min_dist: list[float] = []

    for img_path, reason in sampled:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        rgb, seg, kp_xy, kp_conf = pred.predict_keypoints(img_path)
        if kp_xy is None or kp_conf is None:
            continue
        m = float(kp_conf.min())
        if m >= threshold:
            fired_count += 1
            fired_conf_min_dist.append(m)
            if len(fired_examples) < 6:
                fired_examples.append((img_path, reason, m))
        else:
            not_fired_conf_min_dist.append(m)

    return {
        "n_no_gate": len(no_gate),
        "n_tested": len(sampled),
        "n_fired": fired_count,
        "fp_rate": fired_count / max(len(sampled), 1),
        "n_per_reason": reason_counts,
        "examples_fired": fired_examples,
        "fired_conf_min_mean": float(np.mean(fired_conf_min_dist)) if fired_conf_min_dist else 0.0,
        "not_fired_conf_min_mean": float(np.mean(not_fired_conf_min_dist)) if not_fired_conf_min_dist else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--stride", type=int, default=10)
    args = p.parse_args()

    flights = [
        "datasets/drone-racing-dataset/data/piloted/flight-01p-ellipse",
        "datasets/drone-racing-dataset/data/piloted/flight-06p-ellipse",
        "datasets/drone-racing-dataset/data/piloted/flight-13p-trackRATM",
    ]
    ckpts = [
        ("single-gate (report 07)", "checkpoints/best_deeplabv3plus_keypoint.pth"),
        ("multi-gate  (report 10)", "checkpoints/best_deeplabv3plus_keypoint_multi.pth"),
    ]

    print(f"== FP-rate test == threshold={args.threshold}  stride={args.stride}\n")
    for name, ckpt in ckpts:
        print(f"=== Checkpoint: {name} — {ckpt} ===")
        for fl in flights:
            print(f"  --- {os.path.basename(fl)} ---")
            r = measure_fp(ckpt, fl, threshold=args.threshold, stride=args.stride)
            print(f"    n_no_gate_total       : {r['n_no_gate']}")
            print(f"    n_tested (stride={args.stride})    : {r['n_tested']}")
            print(f"    n_fired                 : {r['n_fired']}")
            print(f"    FP rate                 : {r['fp_rate']*100:.1f}%")
            print(f"    n_per_reason            : {r['n_per_reason']}")
            print(f"    mean conf.min when fired: {r['fired_conf_min_mean']:.3f}")
            print(f"    mean conf.min when miss : {r['not_fired_conf_min_mean']:.3f}")
            if r["examples_fired"]:
                print(f"    example fires (img, reason, conf.min):")
                for ex in r["examples_fired"][:3]:
                    print(f"      {os.path.basename(ex[0])}  reason={ex[1]}  conf.min={ex[2]:.3f}")
            print()


if __name__ == "__main__":
    main()
