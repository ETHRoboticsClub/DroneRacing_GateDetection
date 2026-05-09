# Report 08 — `prior_mode="ekf"` and why QuAdGate-refined EKF is worse than `model_direct`

| | |
|---|---|
| **Author** | Mahbod Tajdini, ETHRC Aeronautics — `mtajdini@ethrobotics.ch` |
| **Date** | 2026-05-09 |
| **Hardware** | qolam, RTX 5090, gate-detection conda env |
| **Checkpoint** | `checkpoints/best_deeplabv3plus_keypoint.pth` (the report 07 checkpoint, unchanged) |
| **Flight** | `flight-01p-ellipse` (TII v3.0.0 piloted, full 12,666 frames at stride 1 + 423 at stride 30) |
| **Run** | `logs/p1_1/ekf_20260509_184537/` (strict-priority EKF), `logs/p1_1/ekf_hybrid_20260509_185156/` (hybrid 3-tier) |
| **Companion commits** | `d35a1ee` (strict EKF), `7a70845` (hybrid EKF) |

## TL;DR

Two implementations of `prior_mode="ekf"` were benchmarked on `flight-01p-ellipse` and **neither improves on the report 07 `model_direct` baseline (85.1 % detection / 63.7 % within 5 px / 102 fps)**. Strict-priority EKF is highly accurate when QuAdGate accepts the prior (84 % within 5 px) but only does so on 4–10 % of frames; the hybrid 3-tier (EKF → model → centred default) brings detection back to ≈ 85 % but the within-5 px rate collapses to 21–28 % because QuAdGate's RANSAC, given a stale or marginally-wrong prior, "succeeds" with bad corners that displace the model's good ones.

**Decision: ship `model_direct` as the deployment mode for now and revisit EKF only as a Kalman-fusion downstream of `model_direct` corners (P2-1, not P1-1).**

## Numbers

All on `flight-01p-ellipse`, single seed 42, single checkpoint. ✓ = verified by reading the linked `bench_ekf_stride*.log` directly.

### Strict-priority EKF (`d35a1ee`)

EKF state is the only prior used; cold-start fallback to model-prior, then centred default. Once `last_detected` is set, the model and centred-default tiers are bypassed forever (single-tier in steady state).

| Stride | Det rate | Mean cerr | Median | ≤ 5 px | ≤ 10 px | Throughput |
|---|---|---|---|---|---|---|
| 1 (12,666 frames) | 513 / 12666 = **4.1 %** | 10.61 px | 2.69 px | **84.0 %** | 88.1 % | 51.2 fps |
| 30 (423 frames) | 40 / 423 = **9.5 %** | 29.24 px | 8.70 px | 42.1 % | 52.6 % | 51.3 fps |

### Hybrid 3-tier EKF (`7a70845`)

EKF first; on QuAdGate rejection, retry with model-prior (confidence-gated like `prior_mode=model`); on second rejection, retry with centred default. So every frame attempts up to three QuAdGate calls.

| Stride | Det rate | Mean cerr | Median | ≤ 5 px | ≤ 10 px | Throughput |
|---|---|---|---|---|---|---|
| 1 (12,666 frames) | 11235 / 12666 = **88.7 %** | 24.98 px | 16.90 px | 20.5 % | 30.7 % | 49.5 fps |
| 30 (423 frames) | 358 / 423 = **84.6 %** | 25.70 px | 15.47 px | 27.8 % | 38.2 % | 43.7 fps |

### Comparison vs report 07's headline modes (stride 30, `flight-01p-ellipse`)

| Mode | Det rate | Mean cerr | Median | ≤ 5 px | Throughput |
|---|---|---|---|---|---|
| `gt` (perfect prior, refinement-only) | 66.4 % | 2.41 px | 0.75 px | 94.3 % | 53.8 fps |
| `model` (head→QuAdGate) | 59.1 % | 7.23 px | 1.15 px | 72.7 % | 53.6 fps |
| `model_direct` (head only) | 85.1 % | 12.6 px | **1.01 px** | **63.7 %** | **102.0 fps** |
| **`ekf` (strict)** | 9.5 % | 29.2 px | 8.70 px | 42.1 % | 51.3 fps |
| **`ekf` (hybrid)** | 84.6 % | 25.7 px | 15.5 px | 27.8 % | 43.7 fps |

## Why both EKF variants underperform `model_direct`

1. **Strict-priority EKF burns coverage on QuAdGate's RANSAC pickiness.** When `last_detected` is bound, QuAdGate is the only path forward; if its RANSAC inlier count does not clear the threshold (e.g. because the gate moved or the LSD lines drifted), the frame is reported as missed even though the keypoint head has perfectly serviceable corners on disk. With stride=30 the previous-frame prior is ~1 second of motion stale, which QuAdGate often rejects outright.

2. **Hybrid EKF "succeeds" with worse corners than the model would have given.** When the EKF prior is *almost* right, QuAdGate accepts it and refines via affine RANSAC against the LSD lines visible in the predicted mask. If those lines are noisy or partially occluded, RANSAC converges to an affine that aligns with the available-but-wrong lines — producing corners that are *internally consistent* but spatially displaced. The model's raw corners (median 1.01 px) are demolished and replaced with QuAdGate-refined corners (median 15.5 px). The model fallback only kicks in when QuAdGate **rejects** the EKF prior, not when QuAdGate accepts it badly. Net: EKF prior is good in fresh frames (rare on this flight) and harmful in stale frames (common).

3. **`model_direct` already has near-zero per-frame error on what it detects.** Median 1.01 px on `flight-01p-ellipse` says: when the keypoint head fires, it is *more accurate* than QuAdGate's RANSAC-from-prior step. Adding QuAdGate on top can only damage. The report 07 `model` (head→QuAdGate) row already showed this: 73 % within 5 px vs `model_direct`'s 64 %, but at the cost of 26 pp of detection rate (59 % vs 85 %) — i.e. QuAdGate cleans up the easy frames but rejects the hard ones, raising precision and dropping recall in equal measure.

## Caveats called out per CLAUDE.md §0

1. **This is not a real EKF.** No motion model, no IMU integration, no covariance propagation. "EKF prior" here = "previous detected (8, 2) corners, identity motion". A real EKF would predict the next frame's corners using IMU-derived ego-motion + a learned or kinematic gate-pose model; that prior would be tighter than identity-motion and QuAdGate's rejection rate could drop. The benchmark's *negative* result on this stub-EKF should not be read as "Kalman fusion is doomed" — only "QuAdGate-refinement of stale identity-motion priors is doomed".
2. **Single seed, single flight, single checkpoint.** Same caveats as report 06 / 07.
3. **Stride=1 is closer to deployment but still flight-01p-ellipse, not real footage.** The 39 % real-footage detection rate (README) is unaddressed by this report.

## Decision

The deployment-headline mode in the README and research doc remains **`model_direct`** (per the report-07 commit `30ee246`). The implemented `prior_mode="ekf"` ships as-is in the codebase for completeness and for any future per-flight ablations, but is not on the deployment critical path.

## Next steps

1. **P2-1 — Kalman fusion downstream of `model_direct`** (≈ 1 day). Run the keypoint head on every frame to get raw corners; carry an explicit Kalman state per corner with measurement covariance from the heatmap peak's spatial variance; predict forward via IMU-derived rotation between frames. The corner *output* would be the EKF posterior, not QuAdGate-refined. Expected effect: median 1.01 px → low-single-digit on the noisy frames, smoother per-corner trajectory for the controller.
2. **Multi-modal heatmap supervision** (the report 07 P1-2 follow-up, ~1 day) is independent of EKF and remains the highest accuracy lever pending.
3. **Real-footage Tier-2 bench** still blocked on Drive data.

## Reproduction

```bash
ssh -t qolam 'tmux new -A -s ekf "cd uni/DroneRacing_GateDetection && \
  CK=checkpoints/best_deeplabv3plus_keypoint.pth && \
  CFG=configs/deeplabv3plus_keypoint.yaml && \
  FL=datasets/drone-racing-dataset/data/piloted/flight-01p-ellipse && \
  TS=\$(date +%Y%m%d_%H%M%S) && OUT=logs/p1_1/ekf_\$TS && mkdir -p \$OUT && \
  for STRIDE in 30 1; do \
    /home/mahbod/miniconda3/bin/conda run -n gate-detection --no-capture-output \
      python scripts/benchmark_flight.py \$CK \$FL --config \$CFG \
        --prior-mode ekf --stride \$STRIDE --n 0 \
        --out \$OUT/grid_ekf_stride\${STRIDE}.png \
        2>&1 | tee \$OUT/bench_ekf_stride\${STRIDE}.log; \
  done"'
```

Wall-clock: ~10 s for stride=30, ~4 min for stride=1 on RTX 5090.
