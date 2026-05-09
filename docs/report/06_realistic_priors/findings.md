# Report 06 — Prior-mode ablation on `flight-01p-ellipse` (plan P0-2)

| | |
|---|---|
| **Author** | Mahbod Tajdini, ETHRC Aeronautics — `mtajdini@ethrobotics.ch` |
| **Date** | 2026-05-09 |
| **Hardware** | qolam, RTX 5090, gate-detection conda env |
| **Checkpoint** | `checkpoints/best_deeplabv3plus_imagenet_nonorm.pth` (epoch 90, val_iou 0.8840 — see report 05) |
| **Flight** | `flight-01p-ellipse` (TII v3.0.0 piloted), 423 frames sampled at stride 30 |
| **Run** | `logs/p0_2/20260509_155042/` |
| **Companion commits** | `9aa6308` (CLI flag), `7f500d2` (label-parse fix #1), `4f23d63` (label-parse fix #2) |

## TL;DR

QuAdGate's corner-refinement quality is **highly sensitive to the prior**. With the GT prior (the original benchmarking convenience), 93.7 % of detections land within 5 px. With a 30 px-σ Gaussian jitter on the GT prior — a generous model of EKF drift between frames — that drops to **0.0 % within 5 px**. The headline 5-px-accurate detection rate the README quotes is *not* the deployed-time number; it is *given a perfect prior*. Mean corner error rises 13× (2.78 → 37.1 px) under noisy prior, and the detection rate itself drops 21 pp (67.4 → 46.3 %).

This is exactly the failure mode plan P0-2 was meant to expose. Going forward the deployed pipeline needs either (i) a higher-fidelity prior than σ=30 px (an EKF carrying detected corners forward across frames, mode P0-2-ekf), (ii) a refinement step that does not depend on a tight prior, or (iii) a model output that bypasses QuAdGate altogether (e.g. a direct keypoint head — plan P1-1).

## Numbers

All from `flight-01p-ellipse`, stride 30, seed 42, 423 frames sampled. ✓ = verified by reading `logs/p0_2/20260509_155042/bench_*.log` directly.

| Prior mode | Detection rate | Mean cerr | Median cerr | Std cerr | ≤ 5 px | ≤ 10 px |
|---|---|---|---|---|---|---|
| **gt** | **285 / 423 = 67.4 %** | **2.78 px** | 0.73 px | 8.93 | **267 / 285 = 93.7 %** | 270 / 285 = 94.7 % |
| **noisy σ = 30 px** | 196 / 423 = 46.3 % | 37.1 px | 33.8 px | 17.0 | 0 / 196 = 0.0 % | 2 / 196 = 1.0 % |
| **none (centred)** | 321 / 423 = 75.9 % | 30.4 px | 21.3 px | 25.9 | 0 / 309 = 0.0 % | 2 / 309 = 0.6 % |

**Distance-sliced (gt mode only)** — closer gates are *harder*, not easier:

| drone→gate | N | det rate | mean cerr |
|---|---|---|---|
| < 1 m | 68 | 63.2 % | 2.34 px |
| 1–2 m | 76 | 43.4 % | 7.05 px |
| 2–3 m | 87 | 58.6 % | 2.27 px |
| 3–5 m | 175 | 80.6 % | 2.37 px |
| 5–8 m | 17 | **100 %** | **0.57 px** |

The 1–2 m bucket is the worst. Likely cause: at that distance the gate spans a large fraction of the FOV; the inner corners often touch image edges where LSD line segments get clipped by the bounds-rejection at `quadgate.py:310–311`.

## Caveats called out per CLAUDE.md §0 (must read before quoting)

1. **`gt` mode of 285 / 423 detections has 285 valid `cerr` values, but `none` mode of 321 detections only has 309 valid `cerr` values** (`bench_none.log:13`). The 12 missing entries come from frames where `parse_pixel_corners` returned no usable gate after the 4-corner-visibility threshold; `inner_corner_error` then returns `None` (`metrics.py:26–27`). For `none` mode this matters: the 75.9 % detection rate counts QuAdGate finding *something*, but for ~12 of those 321 there is no GT to compare against.

2. **`none` mode's `inner_corner_error` is "best of all GT gates in frame", not "the labelled gate"** (`metrics.py:30–34`). With 4 visible gates per frame typical, the centred default prior often locks onto whichever gate happens to overlay the image centre — not necessarily the gate the label points at. So the 75.9 % is "gate-agnostic detection", *not* deployed-mode "right gate, right corners". The mean cerr (30.4 px) is the distance from those snap-on detections to the *closest* labelled gate; some snap-ons may be on the gate the labels named, others on a neighbouring gate that happened to be central.

3. **Single seed.** `noisy` mode used `np.random.RandomState(42)`. Different seeds will move the noisy numbers; the gt/none cells are deterministic (no per-frame randomness).

4. **Single flight.** All numbers are from `flight-01p-ellipse`. The README's 82 % / 5.0 px headline was on `flight-01p-fixed`, which is on Google Drive and not yet on qolam. We can re-run all three modes on `-fixed` once that lands.

5. **The original 82 % number was almost certainly inflated by a missing-script bug.** `parse_pixel_corners` historically returned `[]` for any 17-col label (because outer-synthesis lived in the missing `scripts/prepare_labels.py`). Whoever measured the original 82 % had to be running on 29-col data, which means `prepare_labels.py` was actually present in their checkout and absent from ours. So *our* 67.4 % `gt` number on `-ellipse` is comparable across configs *within this report*, but **not directly comparable to the 82 %** until we have the same 29-col labels they had.

## What this changes about the previous reports

- **Report 02 / 05's val IoU 0.8840** still stands — segmentation IoU is computed pixel-wise from the SAM masks, independent of `parse_pixel_corners`.
- **Report 03's TRT latency** (0.45 ms FP16 on RTX 5090) still stands — that measurement is a pure architecture+weights forward pass.
- **The README's 82 % / 5.0 px is suspect** until reproduced with the 29-col labels and the same checkpoint. The plan's P0-2 entry should be updated to read "the headline numbers reflect a perfect prior; the realistic-prior result is 0 % accurate-to-5 px".

## What's now actually known

- **Segmentation is solid** at 88 % val IoU on TII split, ~50–100 fps projected on Orin NX (report 03).
- **Corner refinement is fragile** under realistic prior drift. QuAdGate is a refinement step, not a from-scratch detector — its performance is bounded above by the prior quality.
- **The `gt` mode of `benchmark_flight` was a refinement-quality benchmark all along**, never a deployed-detection benchmark. Now that `--prior-mode {gt, noisy, none}` exists, that distinction is *visible* rather than hidden.

## Decision points

The plan's P0-2 done-criterion was *"`README.md` § Known Issues lists detection rate and reprojection for {gt-prior, noisy-σ-30-prior, ekf-prior, no-prior}"*. We now have three of those four (no `ekf` yet). The honest read of the three:

| Action | Status |
|---|---|
| Update `README.md` § Known Issues with the prior-mode caveat | **Ready** — pending separate `docs(readme)` commit |
| Update `docs/research/dataset_strategy.tex` with the same caveat | **Ready** — same |
| Implement `prior_mode="ekf"` (carry previous frame's detected corners forward) | **Worth doing** — gives the actual deployment-relevant number |
| Reconsider P1-1 priority (multi-task keypoint head) | **Higher** now that we know QuAdGate is the bottleneck |

## Next steps suggested by this report

1. **Immediate** (10 min): write a separate `docs(readme)` commit that prefaces the 82 % / 5.0 px line with "*on `flight-01p-fixed` with GT priors; under realistic prior drift the within-5 px rate drops to ~0 %, see report 06*". Stops the misleading-headline bleed.
2. **High value** (1 hour): implement `prior_mode="ekf"`. Keep last frame's detected corners; if last detection failed, re-init from a centred default. Re-run on `flight-01p-ellipse`. The number this produces is the *deployment-relevant* corner-accuracy metric.
3. **Highest value** (multi-day): plan P1-1 — add a keypoint head, train multi-task. A direct keypoint regression head sidesteps QuAdGate entirely; if it gives ≥ 80 % within-5 px without the EKF crutch, it ends the prior-sensitivity discussion.
4. **Blocked on Drive data**: re-run all four prior modes on `flight-01p-fixed` once it lands on qolam. Will let us tie out the original README headline.

## Reproduction

```bash
ssh -t qolam 'tmux new -A -s prior "cd uni/DroneRacing_GateDetection && \
  CK=checkpoints/best_deeplabv3plus_imagenet_nonorm.pth && \
  CFG=configs/deeplabv3plus_imagenet_nonorm.yaml && \
  FL=datasets/drone-racing-dataset/data/piloted/flight-01p-ellipse && \
  TS=\$(date +%Y%m%d_%H%M%S) && mkdir -p logs/p0_2/\$TS && \
  for MODE in gt noisy none; do \
    /home/mahbod/miniconda3/bin/conda run -n gate-detection --no-capture-output \
      python scripts/benchmark_flight.py \$CK \$FL --config \$CFG \
        --prior-mode \$MODE --stride 30 --n 0 \
        --out logs/p0_2/\$TS/grid_\$MODE.png \
        2>&1 | tee logs/p0_2/\$TS/bench_\$MODE.log; \
  done"'
```

Wall-clock: ~3 × 10 s on RTX 5090, plus initial `torch.compile(max-autotune)` warm-up (~3 min on first cell). Total ~5 min.
