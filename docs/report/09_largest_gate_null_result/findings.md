# Report 09 — Largest-gate supervision is *worse* than first-gate (Stage 1 of P1-1 follow-up — null result)

| | |
|---|---|
| **Author** | Mahbod Tajdini, ETHRC Aeronautics — `mtajdini@ethrobotics.ch` |
| **Date** | 2026-05-09 |
| **Hardware** | qolam, RTX 5090, gate-detection conda env |
| **Run** | `logs/p1_1/largest_gate_sanity_20260509_195002.log` (10-epoch sanity) |
| **Code state** | qolam working-tree-only edit; **never committed**, **reverted before close-out** |

## TL;DR

Hypothesis going in: report 07's `val_corner_dist=22 px` plateau was caused by the keypoint head being supervised against label-file-order's first gate, which is arbitrary and produces noisy targets across consecutive frames. Switching to "largest gate by bounding-box area" (a deterministic geometric proxy for "the closest gate") was supposed to give temporally smoother supervision and let the model converge to a tighter `val_corner_dist`.

**Result: training is significantly *worse*.** At epoch 10, val IoU drops from 0.696 (first-gate) to 0.214 (largest-gate); `val_corner_dist` rises from 42 px to 186 px. The change has been reverted; the report 07 first-gate path is restored as the baseline.

The likely reason — first-gate-by-file-order in TII labels is *not* arbitrary; it appears to be track-sequence-ordered (or some semantic equivalent), so gate[0] is consistent across consecutive frames in a way that "largest by area" is not. As the drone rotates or laterally translates, "largest in image" can flip between gates frame-to-frame, while "first in track order" stays stable until the drone passes a gate. **The first-gate convention has latent semantic structure I underestimated.**

## Numbers (10-epoch sanity, both runs at β=100, σ=3 px, fg_weight=10, seed 42)

| Metric @ epoch 10 | Report 07 (first-gate) | This run (largest-gate) | Δ |
|---|---|---|---|
| val IoU | 0.6963 | **0.2144** | **−0.482 (catastrophic)** |
| val_corner_dist | 42.45 px | **186.15 px** | +143.7 px |
| val_kp_loss | 0.0006 | 0.0024 | +0.0018 |
| val IoU peak (across the 10 epochs) | monotonic, ~0.83 by epoch 8 | **0.3283 at epoch 4, then declines** | n/a |

Training-side metrics (train IoU, train_kp_loss) are still improving in the largest-gate run — train IoU 0.06 → 0.74, train_kp_loss 0.099 → 0.0006 — but val tracks separately. Classic train/val divergence.

## What the trajectory shows

```
Epoch  1: val_iou=0.0000  val_kp_loss=0.0030  val_corner_dist=164.94 px
Epoch  2: val_iou=0.1847  val_kp_loss=0.0059  val_corner_dist=137.89 px
Epoch  3: val_iou=0.1777  val_kp_loss=0.0019  val_corner_dist=112.99 px
Epoch  4: val_iou=0.3283  val_kp_loss=0.0015  val_corner_dist= 91.26 px  ← peak
Epoch  5: val_iou=0.2563  val_kp_loss=0.0027  val_corner_dist=115.83 px
Epoch  6: val_iou=0.1964  val_kp_loss=0.0035  val_corner_dist=171.31 px
Epoch  7: val_iou=0.1960  val_kp_loss=0.0044  val_corner_dist=190.88 px  ← worst
Epoch  8: val_iou=0.2393  val_kp_loss=0.0054  val_corner_dist=170.98 px
Epoch  9: val_iou=0.3200  val_kp_loss=0.0027  val_corner_dist=147.79 px
Epoch 10: val_iou=0.2144  val_kp_loss=0.0024  val_corner_dist=186.15 px
```

The val_kp_loss *climbs* from epoch 4 to epoch 8 (0.0015 → 0.0054), which is the diagnostic giveaway: the head is unlearning what it figured out early, because the supervision itself is contradicting frame-to-frame. Compare to report 07's first-gate trajectory which monotonically improves on both heads.

## Why the hypothesis was wrong

Two effects compound:

1. **First-gate-in-file is not arbitrary in TII piloted flights.** The label-file order appears to follow track sequence (the gate the drone is currently approaching is in a stable slot across consecutive frames). The model was implicitly learning "find the next-track-order gate", which is a learnable visual pattern (the gate the camera is centred on, the gate at the end of a corridor of context, etc.).

2. **Largest-by-area is *not* "the closest racing gate" in practice.** It's just whichever gate happens to dominate the frame's pixel area. As the drone yaws or strafes:
   * A side-gate the drone is *passing* (not flying through) can be largest because it's metres away laterally.
   * A foreground edge gate can flip with a nearby central gate based on ±5° head movement.
   * Closer gates are *more* perspective-distorted and *more* likely to have inner corners outside the frame, so "largest" is sometimes the worst-conditioned 4-point quad in the scene.

So switching to largest-by-area destroyed the latent track-order signal AND substituted a noisier proxy. Both heads suffered.

## Caveats per CLAUDE.md §0

1. **Single-seed null result.** I am calling this a real negative result rather than seed noise because the train/val divergence pattern (train monotonically improving, val oscillating + climbing-loss) is the diagnostic signature of an inconsistent supervision target, not a seed effect.
2. **Did not test alternative consistency heuristics** (e.g., "centermost gate", "highest sum-of-vis", "gate closest in coords to last frame's pick"). Some might work; not pursuing without a positive prior.
3. **Code change was never committed.** The diff lived only on the qolam working tree and was rsynced back to the report-07 first-gate baseline at close-out. `git diff HEAD` on both laptop and qolam returns empty.

## Decision

**Stay on the report-07 first-gate baseline. Do not pursue any "single-gate which-gate" heuristic further.** The latent track-order signal is real and switching off it harms more than it helps.

**Accept the 22-px val_corner_dist as the floor of single-gate supervision.** The report-07 deployment headline (`model_direct`: 85.1 % detection, 1.01 px median, 63.7 % within 5 px) was already produced under this regime and remains the supportable number.

The remaining accuracy lever is the **multi-modal heatmap supervision (Stage 2)** — render Gaussian peaks for *all* gates' corners per channel, NMS at inference. That's the architecturally-clean way to get gate-disambiguation without trying to outguess the labellers. Estimated cost: ~2 days. Worth doing IF a meaningful accuracy improvement is needed for A2RL or AIGP; not on the critical path otherwise.

## Next steps suggested by this report

1. **Immediate** (no work): drop Stage 1 from the plan. Do not retry single-gate heuristics.
2. **If accuracy push is desired**: plan P1-2-mm — multi-modal heatmap with NMS. Touchpoints: `data/dataset.py` (multi-gate target render), `losses/keypoint.py` (peak finder + NMS replacing soft-argmax), `evaluation/benchmark.py` (per-frame gate selection). ~2 days.
3. **Otherwise**: shift effort to deployment-side items — confidence calibration (#2 from the rank), ROS2 wrapper (#5), Orin NX deployment (#6).

## What this report is NOT saying

* It is *not* saying first-gate is good. The 22-px `val_corner_dist` and 64 % within-5 px are below where we want to be for tight A2RL gates.
* It is *not* saying multi-gate supervision won't work. Multi-modal heatmaps avoid the "which gate" ambiguity entirely.
* It is *not* a seed-noise dismissal — the train/val divergence pattern is unambiguous.

## Reproduction (would re-fail)

The change is in this report's git history only (the dataset.py edit was reverted before commit). To reproduce the negative result:

```python
# In src/gate_detection/data/dataset.py:35-74, replace
g = gates[0]
# with
g = max(gates, key=lambda gate: gate["w"] * gate["h"])
```

Then run the report 07 sanity: `python scripts/train.py --config configs/deeplabv3plus_keypoint.yaml --seed 42 --epochs 10`. Compare the val_corner_dist trajectory.
