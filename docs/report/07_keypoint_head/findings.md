# Report 07 — Multi-task DeepLabV3+ with keypoint heatmap head (plan P1-1)

| | |
|---|---|
| **Author** | Mahbod Tajdini, ETHRC Aeronautics — `mtajdini@ethrobotics.ch` |
| **Date** | 2026-05-09 |
| **Hardware** | qolam, RTX 5090, gate-detection conda env |
| **Checkpoint** | `checkpoints/best_deeplabv3plus_keypoint.pth` (epoch 99, val IoU 0.8704, val_corner_dist 22.95 px on TII split) — 87 MB, 22.6 M params (≈580 K added by the keypoint head) |
| **Config** | `configs/deeplabv3plus_keypoint.yaml` (final α=1, β=100, fg_weight=10, σ=3 px) |
| **Flight** | `flight-01p-ellipse` (TII v3.0.0 piloted), 423 frames sampled at stride 30 |
| **Run** | `logs/p1_1/keypoint_full_20260509_170410.log` (training); `logs/p1_1/20260509_180859/` (bench) |
| **Companion commits** | `331deba` (keypoint head implementation), `8b16a31` (β=1 → β=100 fix) |

## TL;DR

The new multi-task DeepLabV3+ regresses the four inner gate corners as Gaussian heatmaps in parallel with the segmentation mask. On `flight-01p-ellipse` with **no GT prior whatsoever**, the model-derived prior pushes the within-5 px corner-accuracy rate from **0 %** (the realistic-prior result of report 06) to **72.7 %** when fed through QuAdGate, or **63.7 %** when used directly without QuAdGate refinement at **85.1 % detection rate** (vs 75.9 % for QuAdGate-with-centred-default). This is the first deployable corner-localisation result in the project — every previous configuration either required a GT prior (impossible at deploy time) or collapsed under realistic prior drift.

The val IoU regresses slightly (0.8704 vs 0.8840 in report 05, -1.4 pp) — the price for splitting capacity between two heads under β=100 multi-task weighting. The model-prior gt-equivalent (`prior_mode=gt`) numbers are essentially unchanged from report 06 (66.4 % vs 67.4 %; 94.3 % within 5 px vs 93.7 %), so the seg-head downstream usefulness is preserved.

## Numbers

All from `flight-01p-ellipse`, stride 30, seed 42, 423 frames sampled, single checkpoint. ✓ = verified by reading `logs/p1_1/20260509_180859/bench_*.log` directly.

| Prior mode | Detection rate | Mean cerr | Median cerr | Std cerr | ≤ 5 px | ≤ 10 px |
|---|---|---|---|---|---|---|
| **gt** | 281 / 423 = 66.4 % | 2.41 px | 0.75 px | 7.74 | **265 / 281 = 94.3 %** | 270 / 281 = 96.1 % |
| **noisy σ = 30 px** | 185 / 423 = 43.7 % | 38.83 px | 34.62 px | 19.46 | 0 / 185 = 0.0 % | 0 / 185 = 0.0 % |
| **none (centred)** | 314 / 423 = 74.2 % | 29.65 px | 20.65 px | 24.12 | 0 / 303 = 0.0 % | 6 / 303 = 2.0 % |
| **model** | 250 / 423 = 59.1 % | **7.23 px** | **1.15 px** | 12.93 | **181 / 249 = 72.7 %** | 189 / 249 = 75.9 % |
| **model_direct** | **360 / 423 = 85.1 %** | 12.63 px | **1.01 px** | 23.82 | 221 / 347 = 63.7 % | 237 / 347 = 68.3 % |

Mode definitions (`evaluation/benchmark.py:77-186`):
* **gt** — QuAdGate prior = GT corners from the label (refinement-only benchmark, over-states deployment).
* **noisy σ = 30 px** — QuAdGate prior = GT corners + per-coordinate Gaussian σ=30 px jitter (models EKF drift).
* **none** — QuAdGate prior = a centred default (cold-start / no-state-carryforward).
* **model** — QuAdGate prior = the keypoint head's predicted inner corners + 1.4× outer-corner synthesis. **Deployment-relevant.**
* **model_direct** — Skip QuAdGate entirely; the keypoint head's corners ARE the detected output. **Isolates the head's standalone accuracy.**

## What this changes vs report 06

* **Segmentation-only QuAdGate is no longer the bottleneck.** Report 06 showed QuAdGate refinement collapsing under realistic prior drift (0 % within 5 px in noisy mode, 0 % in none mode). The keypoint head provides QuAdGate with a meaningfully better prior than centred-default, and 72.7 % within 5 px is the result.
* **The model-direct mode is the new deployment headline.** 85.1 % detection rate at median 1.01 px corner error, with no GT, no manual init, no EKF state. This is the number a deployed system can quote on a cold-start single frame. The mean of 12.63 px reflects a long tail (see Caveats §3).
* **Seg val IoU is "good enough" at 0.8704.** A 1.4 pp drop from report 05's 0.8840 is the price for the multi-task training under β=100. The downstream seg-head behaviour (gt-prior numbers in this report's table vs report 06's) is essentially unchanged.

## Architecture & training (file:line)

* **Keypoint head** (`src/gate_detection/models/heads/keypoint.py:18-62`): 256→64 (3×3 Conv-BN-ReLU) → 64→4 (1×1 Conv) → Sigmoid → bilinear upsample to input resolution. ≈ 580 K parameters; tapped off the DeepLabV3+ ASPP decoder feature map (`models/deeplabv3plus.py:35-72`).
* **Loss** (`src/gate_detection/losses/keypoint.py:69-91`): weighted MSE with `fg_weight·target + 1` per-pixel weighting (Newell et al. 2016 hourglass convention) — upweights near-peak pixels by 11× over background. Combined as `L = L_seg + β·L_kp` at `training/trainer.py:136-160` with `β = 100` (see "loss balance" caveat below).
* **Targets** rendered AFTER augmentation (`data/dataset.py:163-185`): the augmentation pipeline at `data/transforms.py:127-217` (`train_augment_kp`) transforms image+mask+inner-corner coords+visibility flags consistently; horizontal flip permutes corner channels [TL↔TR, BL↔BR] and affine warp uses `BORDER_CONSTANT` (black) instead of `BORDER_REFLECT` so reflected gates do not appear as visually-real but unlabelled.
* **Soft-argmax decode** (`losses/keypoint.py:108-150`) at inference — global softmax with β=100 gives integer-pixel resolution; the val metric compares pred-soft-argmax to target-soft-argmax so the same systematic bias cancels.
* **Confidence gate** at benchmark (`evaluation/benchmark.py:200-216`): per-channel heatmap `min` (not mean) above 0.3 — one bad channel can poison the (8, 2) prior, so the conservative AND is correct.

## Caveats called out per CLAUDE.md §0 (must read before quoting)

1. **Loss balance was the entire fight.** The first sanity run at β=1 (the literature default) put `val_kp_loss` at 0.001 — exactly the analytical `predict-zero` baseline. The kp head was barely learning. Bumping β to 100 brought the kp gradient to seg-comparable scale and dropped `val_corner_dist` from 100+ px (random init) to 22 px (converged). The sigmoid head bound on `[0, 1]` plus weighted MSE is well-conditioned but the 384²-pixel mean dilution means the loss has to be aggressively scaled to compete with seg's per-pixel BCE+Dice.

2. **`val_corner_dist = 22 px` ≠ `bench mean cerr = 7-12 px`.** The val metric supervises against the **first gate** in `parse_gates` order (`data/dataset.py:_first_gate_inner`). The benchmark's `inner_corner_error` metric does **best-of-all-GT-gates matching** (`evaluation/metrics.py:30-34`). On 4-gate frames the model often predicts a *different* gate than the labelled-first one — an error against `_first_gate_inner` but a near-zero error against the closest GT gate. The 22 px val number is therefore a pessimistic upper bound; the bench numbers are the realistic field metric.

3. **Long tail in model_direct.** Median 1.01 px / mean 12.63 px / std 23.82 — the distribution has heavy outliers. Inspection of three sample frames at training time (commit `8b16a31` diagnostic) showed two of four channels predicting the wrong gate's corner (~85 px off) while the other two were within 1 px. The multi-gate confusion at training time (single-gate supervision in 4-gate scenes) is the proximate cause; the bench's best-match metric absorbs the in-frame confusion partially but not fully.

4. **Single seed, single flight.** Same caveats as report 06. We have not run multi-seed; numbers may shift ±2-3 pp on re-run with a different seed (especially in the within-5 px column). `flight-01p-fixed` is still on Google Drive.

5. **Confidence threshold (0.3) is heuristic.** With sigmoid output, an early-training head saturates at scattered pixels giving `peak ≈ 0.99` even at random init; a more principled gate would be peak height + spatial concentration (KL divergence from a unit Gaussian). Not a critical-path issue today since post-convergence peaks are real, but worth tightening before A2RL.

## What's now actually known

* **Direct keypoint regression is feasible** with this architecture and the existing TII training data. The head trains, the heatmaps localise, soft-argmax decode works, and the resulting prior beats every previous deployable configuration.
* **The multi-gate supervision noise is real and bounded.** With single-gate supervision, the model can converge to ~22 px val error on the supervised gate and ~7-12 px mean error vs the closest gate in the frame. This is the floor of what `_first_gate_inner` supervision can achieve; further accuracy needs either multi-modal heatmap supervision (CenterNet-style) or per-gate-instance attention (much bigger architecture change).
* **Seg accuracy is preserved.** The 0.870 val IoU and the unchanged gt-mode numbers say the seg head is still doing its job. Multi-task at β=100 was not free — the seg head paid 1.4 pp — but the cost was bounded.

## Decision points

| Action | Status |
|---|---|
| Update `README.md` § Known Issues to add the model-direct headline | **Ready** — separate `docs(readme)` commit, the `~0 % within 5 px` line is now superseded |
| Update `docs/research/dataset_strategy.tex` with the keypoint-head method + numbers | **Ready** — natural follow-up to report 06's findings |
| Implement `prior_mode="ekf"` (carry previous frame's detected corners) | **Lower priority now** — model prior beats σ=30 EKF noise model already |
| Multi-modal heatmap supervision (all gates' corners, NMS at inference) | **Highest accuracy lever** — would address caveat 3 directly; ≈ 1 day of work |
| Re-run on `flight-01p-fixed` once Drive data lands | **Blocked** |
| Multi-seed envelope | **Useful** — single-seed numbers can shift ±2-3 pp |

## Next steps suggested by this report

1. **Immediate** (15 min): write `docs/readme` commit prefacing the model-direct numbers alongside report 06's caveat. The `~0 % within 5 px under realistic prior` line is now historical — we have a 63.7 % deployable number to put in its place.
2. **High value** (1 day): plan P1-2 — multi-modal keypoint supervision. Render a heatmap channel as the SUM of Gaussian peaks across ALL visible gates' corners (not just the first). At inference, find local maxima above threshold and apply per-channel NMS to extract the K most likely corner candidates. This is the canonical CenterNet recipe and is known to handle multi-instance scenes cleanly.
3. **High value** (1 hour, blocked on data): re-run all five prior modes on `flight-01p-fixed` once that lands. Will let us tie out against the original README headline (82 % / 5.0 px) on the same flight.
4. **Investigation** (30 min): why does `noisy` mode yield 0 % within 5 px while `model` mode (with arguably noisier corners than σ=30 px GT-jitter) yields 72.7 %? Hypothesis: the model's *spatial structure* of corner errors is different from isotropic Gaussian noise — likely the corner geometry stays a valid quadrilateral, which keeps QuAdGate's RANSAC affine fit happy. Worth quantifying.

## Reproduction

```bash
# Train (≈ 20 min on RTX 5090 incl. one-time torch.compile autotune)
ssh -t qolam 'tmux new -A -s kp100 "cd uni/DroneRacing_GateDetection && \
  /home/mahbod/miniconda3/bin/conda run -n gate-detection --no-capture-output \
    python scripts/train.py --config configs/deeplabv3plus_keypoint.yaml --seed 42 \
    2>&1 | tee logs/p1_1/keypoint_full_$(date +%Y%m%d_%H%M%S).log"'

# Bench all 5 modes (≈ 5 min total)
ssh -t qolam 'tmux new -A -s kpbench "cd uni/DroneRacing_GateDetection && \
  CK=checkpoints/best_deeplabv3plus_keypoint.pth && \
  CFG=configs/deeplabv3plus_keypoint.yaml && \
  FL=datasets/drone-racing-dataset/data/piloted/flight-01p-ellipse && \
  TS=\$(date +%Y%m%d_%H%M%S) && mkdir -p logs/p1_1/\$TS && \
  for MODE in gt noisy none model model_direct; do \
    /home/mahbod/miniconda3/bin/conda run -n gate-detection --no-capture-output \
      python scripts/benchmark_flight.py \$CK \$FL --config \$CFG \
        --prior-mode \$MODE --stride 30 --n 0 \
        --out logs/p1_1/\$TS/grid_\$MODE.png \
        2>&1 | tee logs/p1_1/\$TS/bench_\$MODE.log; \
  done"'
```

Wall-clock on RTX 5090: ≈ 25 min total.
