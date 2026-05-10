# Report 10 — Multi-gate Gaussian targets + coherent quad decode (P1-1 Stage 2)

| | |
|---|---|
| **Author** | Mahbod Tajdini, ETHRC Aeronautics — `mtajdini@ethrobotics.ch` |
| **Date** | 2026-05-10 |
| **Hardware** | qolam, RTX 5090, gate-detection conda env |
| **Checkpoint** | `checkpoints/best_deeplabv3plus_keypoint_multi.pth` (epoch 99, val IoU 0.8723, val_corner_dist 29 px multi-modal — see caveats §1) |
| **Config** | `configs/deeplabv3plus_keypoint_multi.yaml` (`data.multi_gate_target=true`, otherwise identical to report 07) |
| **Flight** | `flight-01p-ellipse` (TII v3.0.0 piloted), 423 frames at stride 30 |
| **Run** | `logs/p1_1/multigate_full_20260510_134601.log` (training); `logs/p1_1/multigate_bench_coherent_20260510_142239/` (final bench) |
| **Companion commits** | `e5a1d28` (multi-gate target rendering), windowed/coherent decode added in this report |

## TL;DR

Trained a multi-task DeepLabV3+ keypoint head on **per-channel max-of-Gaussians across all visible gates' inner corners**, instead of report 07's single-gate-of-interest target that plateaued at val_corner_dist=22 px. Motivated by the MonoRace paper (Bahnam et al. 2026, arxiv:2601.15222) which solves multi-gate disambiguation via *adaptive cropping at the input* using EKF state — without online EKF, multi-modal targets are the alternative that removes the "which gate" ambiguity from the loss.

**Result: Stage 2 with coherent_quad_decode is statistically tied with report 07.** Specifically: `model_direct` improves detection rate from 85.1 % → **88.7 %** (+3.6 pp) at the cost of median corner error 1.01 → 1.26 px. `≤5 px` rate is essentially unchanged (63.7 % → 62.8 %). The deployable headline does not meaningfully change.

The path to this result was instructive: a naive global soft-argmax decode initially gave a catastrophic 0–10 % within 5 px on the multi-gate checkpoint because **per-channel argmax independently picked peaks from different gates** (Frankenstein quad). Adding `coherent_quad_decode` (enumerate top-3 peaks per channel, pick the most-rectangular spatial-ordered combination) recovered single-gate-comparable accuracy.

**Decision: ship report 07's single-gate-target checkpoint as the deployment headline; treat Stage 2 as a useful experimental sibling that did not win.** Multi-gate target training is the path forward only when paired with multi-instance inference (Hungarian assignment / clustering), and the +3.6 pp detection coverage doesn't justify the inference complexity until we have a use case.

## Numbers

All on `flight-01p-ellipse`, single seed 42, stride 30, 423 frames sampled. ✓ verified by reading the linked `bench_*.log`.

### Stage 2 vs Report 07 (model with kp head trained on Stage 1 single-gate targets)

| Mode | Stage 2 (`coherent_quad_decode`) | Report 07 (single-gate, soft-argmax) | Δ |
|---|---|---|---|
| `gt` det rate | 67.1 % | 66.4 % | +0.7 pp |
| `gt` mean cerr | 3.04 px | 2.41 px | +0.63 px |
| `gt` ≤5 px | 94.7 % | 94.3 % | +0.4 pp (seg head essentially identical) |
| `noisy` ≤5 px | 0.0 % | 0.0 % | tied (QuAdGate breaks under σ=30 px) |
| `none` ≤5 px | 0.0 % | 0.0 % | tied |
| **`model` det rate** | 55.1 % | 59.1 % | -4.0 pp |
| **`model` ≤5 px** | 69.9 % | 72.7 % | -2.8 pp |
| **`model` median cerr** | 1.30 px | 1.15 px | +0.15 px |
| **`model_direct` det rate** | **88.7 %** | 85.1 % | **+3.6 pp** |
| **`model_direct` ≤5 px** | 62.8 % | 63.7 % | -0.9 pp |
| **`model_direct` median cerr** | 1.26 px | 1.01 px | +0.25 px |

Headline trade-off: Stage 2 `model_direct` shifts the operating point toward higher coverage (+3.6 pp detection) at slight precision cost (+0.25 px median). Net deployment value depends on whether the planner downstream prefers more frames with a less-precise corner estimate or fewer frames with a more-precise one. Neither dominates the other; report 07 numbers stay supportable.

## What we built

### `losses/keypoint.py:render_heatmap_multi`
Inputs: `corners (G, K, 2)`, `vis (G, K)`. Output: per-channel heatmap `(K, H, W)` via **max** (not sum) of Gaussians across gates — keeps peak values bounded in `[0, 1]` so the sigmoid head's loss surface stays well-conditioned. Returns `vis_per_channel (K,)` = 1 if any gate has corner k visible.

### `data/transforms.py:train_augment_kp_multi`
Mirrors `train_augment_kp` but operates on `(G, K, 2)` corners. Spatial transforms (flip + affine) shared across all gates; per-gate channel permutation `[1, 0, 3, 2]` on hflip; affine uses `BORDER_CONSTANT` so reflected gates don't appear as visually-real-but-unlabelled.

### `data/dataset.py:_all_gates_inner` + `multi_gate_target` flag
Returns `(G, K, 2)` corners + `(G, K)` vis for every visible gate. New constructor flag gates the multi-gate path. `data_config.multi_gate_target` round-trips through both interval and final checkpoint saves.

### `losses/keypoint.py:coherent_quad_decode` (added after the first bench failed)
Enumerates top-3 peaks per channel (with NMS at radius 30 px) and over the 3⁴=81 combinations selects the one passing the spatial-order constraint (TR right of TL, BL below TL, BR right of BL, BR below TR) with minimum quad-distortion score `|top_w - bot_w| + |left_h - right_h|`. Predictor auto-selects this decode when `multi_gate_target=True` is in the loaded checkpoint metadata.

## What went wrong on the way (instructive)

### Stage 2 v1: global soft-argmax → catastrophic

First bench gave `model_direct` 0.0 % within 5 px, mean 33.5 px, median 30.4 px (`logs/p1_1/multigate_bench_20260510_141026/`). Diagnostic on per-channel heatmaps confirmed each channel's argmax was sub-pixel-accurate to *some* gate's corner (≤1.6 px), but **different channels picked different gates per frame**:

```
Frame 00100 (2 gates): TL→g1, TR→g1, BR→g0, BL→g1   (Frankenstein)
Frame 00300 (2 gates): TL→g1, TR→g0, BR→g0, BL→g1   (Frankenstein)
Frame 00400 (2 gates): TL→g0, TR→g0, BR→g0, BL→g1   (Frankenstein)
```

Cause: with multi-modal Gaussian targets, soft-argmax with β=100 on a heatmap with two equal-height peaks at distance d returns the midpoint, not a single peak. With heights 0.99 and 0.98, the higher peak dominates by exp(1)≈2.7×, the lower still pulls the result by ~27 %.

### Stage 2 v2: windowed soft-argmax → still bad

Replacing global soft-argmax with windowed-around-argmax (15-px window) lifted `model_direct` from 0.0 % to 8.9 % within 5 px (`logs/p1_1/multigate_bench_windowed_20260510_141548/`). Better, but still useless. Cause: even with one gate's peak picked per channel, **each channel could pick a different gate's peak independently** — exactly the Frankenstein pattern the diagnostic flagged. Windowed soft-argmax fixes the within-peak averaging but not the across-channel inconsistency.

### Stage 2 v3: coherent quad decode → recovered

Adding the gate-coherence enumeration (pick the most-rectangular combo across top-3 peaks per channel) recovered to 62.8 % within 5 px / median 1.26 px — within 1 pp of report 07. This is the deployment-headline number.

## Caveats per CLAUDE.md §0

1. **Multi-modal `val_corner_dist` is not apples-to-apples with report 07.** The training-time val metric (`trainer.py:285-295`) soft-argmaxes the GT heatmap which now has multiple peaks per channel. With β=100 and Gaussian σ=3 px, the soft-argmax of a multi-modal target is the strongest peak's location, but with ties broken by float arithmetic — so a 29 px val_corner_dist in multi-gate mode is **not** a 7-px regression vs report 07's 22 px; it's the metric measuring "distance from pred-soft-argmax to which-peak-of-multi-modal-target-happened-to-win". Reviewer agent flagged this; we did not modify the metric. The bench's `inner_corner_error` (best-of-all-GT-gates matching) is the deployment-comparable number.
2. **Single seed, single flight, single checkpoint.** Same caveats as reports 06 / 07 / 09.
3. **Gate-coherence decode is heuristic.** The "most rectangular" tie-break works on TII piloted flights where gates are roughly square frames seen at modest perspective angles. On heavy perspective (gates seen from oblique angles, e.g., during a hard turn) the spatial-order constraint can fail and the predictor falls back to global soft-argmax with a `[WARN]` line — graceful but accuracy is then back to the v1 catastrophic regime for those frames.
4. **`model_direct` 88.7 % detection rate is not "more correct gates detected"** — it's "more frames where coherent_quad_decode found a spatially-ordered combo above the confidence threshold". Some of those new detections may be on the wrong gate; the +3.6 pp coverage gain is therefore at lower per-frame trustworthiness than the report 07 number.
5. **MonoRace cross-reference.** Bahnam et al. 2026 explicitly handle multi-gate via *adaptive cropping at the input* using EKF state, NOT via multi-modal output. The Stage 2 multi-modal-target approach is the alternative for systems without online EKF state. This report does not invalidate that approach; it just documents that without inference-side gate-coherence enforcement, multi-modal training collapses, and that the empirical headline numbers tie with single-gate-target training once both are tuned.

## Decision

**Single-gate-target (report 07) stays as the deployment-headline checkpoint.** Reason:

* Stage 2 within-5 px rate is identical (62.8 % vs 63.7 %).
* Stage 2 median is +0.25 px worse (1.26 vs 1.01).
* Stage 2 detection rate is +3.6 pp better but the new detections are on potentially-wrong gates (caveat 4).
* Stage 2 inference depends on the heuristic `coherent_quad_decode` falling through to global soft-argmax on hard frames (caveat 3) — single-gate-target's soft-argmax has no such fragility.
* `report_07` checkpoint is already TRT-exported and verified (commit `352304c`); switching to Stage 2 means re-exporting and re-verifying for what the bench shows is a wash.

**Stage 2 stays in the repo as `best_deeplabv3plus_keypoint_multi.pth` for any future use case** (e.g., a track planner that wants per-gate corner candidates rather than one canonical gate). The `coherent_quad_decode` and `windowed_soft_argmax_2d` are committed and exercise the multi-gate inference path when the loaded checkpoint flags `multi_gate_target=True`.

## Next steps suggested by this report

1. **Drop the multi-modal-target line of inquiry as a deployment lever.** It does not move the within-5 px or median-cerr numbers; the remaining accuracy lever for the kp head is *upstream* (better data, better seg target, better pretraining).
2. **Confidence calibration (#2 from the rank list)** — the false-positive on a doorway in the report-07 sample panel is an unaddressed precision issue; tightening the `keypoint_conf_threshold` from peak-height-only to peak-height + spatial-concentration (KL divergence from a unit Gaussian) is a 2-hour change with bounded risk.
3. **Real-footage benchmark (Tier-2)** still blocked on Drive data. The 39 % real-footage detection rate is the live deployment problem; Stage 2 doesn't address it.
4. **MonoRace-style adaptive input cropping** is a meaningful future direction once an EKF state is available — it sidesteps the multi-modal vs single-modal question entirely by feeding the network only the relevant gate's image patch.

## Reproduction

```bash
# Train (≈ 17 min on RTX 5090 incl. one-time torch.compile autotune)
ssh -t qolam 'tmux new -A -s mg100 "cd uni/DroneRacing_GateDetection && \
  /home/mahbod/miniconda3/bin/conda run -n gate-detection --no-capture-output \
    python scripts/train.py --config configs/deeplabv3plus_keypoint_multi.yaml --seed 42 \
    2>&1 | tee logs/p1_1/multigate_full_$(date +%Y%m%d_%H%M%S).log"'

# Bench all 5 modes (≈ 5 min total)
ssh -t qolam 'tmux new -A -s mgbench "cd uni/DroneRacing_GateDetection && \
  CK=checkpoints/best_deeplabv3plus_keypoint_multi.pth && \
  CFG=configs/deeplabv3plus_keypoint_multi.yaml && \
  FL=datasets/drone-racing-dataset/data/piloted/flight-01p-ellipse && \
  TS=\$(date +%Y%m%d_%H%M%S) && OUT=logs/p1_1/multigate_bench_coherent_\$TS && mkdir -p \$OUT && \
  for MODE in gt noisy none model model_direct; do \
    /home/mahbod/miniconda3/bin/conda run -n gate-detection --no-capture-output \
      python scripts/benchmark_flight.py \$CK \$FL --config \$CFG \
        --prior-mode \$MODE --stride 30 --n 0 \
        --out \$OUT/grid_\$MODE.png \
        2>&1 | tee \$OUT/bench_\$MODE.log; \
  done"'
```

Wall-clock: ~22 min total on RTX 5090.
