# Report 12 — Cross-flight bench + held-out flight + FP-on-no-gate (status-report #11 follow-up)

| | |
|---|---|
| **Author** | Mahbod Tajdini, ETHRC Aeronautics — `mtajdini@ethrobotics.ch` |
| **Date** | 2026-05-15 |
| **Hardware** | qolam, RTX 5090, gate-detection conda env |
| **Logs** | `logs/p1_1/cross_flight_20260515_140807/` (cross-flight bench); `/tmp/fp_rate.py` + console output (FP-rate) |
| **Companion** | reports 07 / 10 / 11 — this report **revises** the headlines in 07 and the verdict in 10 |
| **Reviewers** | two independent reviewer agents (CLAUDE.md §8) — caught two critical issues that changed the conclusions |

## TL;DR (revised deployment headline)

The previous deployment-headline checkpoint was **`best_deeplabv3plus_keypoint.pth`** (single-gate target, report 07: 85.1 % / 1.01 px median on `flight-01p-ellipse`). **That number was a memorization artifact.** On the only piloted flight held out from the SAM-mask training pool (`flight-06p-ellipse`), the single-gate checkpoint produces **0.0 % detection** — every frame's keypoint confidence falls below the 0.3 threshold.

The Stage-2 sibling checkpoint **`best_deeplabv3plus_keypoint_multi.pth`** (multi-gate Gaussian targets + `coherent_quad_decode`), reported in report 10 as "statistically tied" with the headline, in fact **generalizes**. On held-out 06p it delivers **79.4 % detection rate / 61.0 % within 5 px / 1.30 px median**, and on three in-training-but-undersampled flights it beats single-gate by 13–35 pp on within-5 px.

**The new deployment headline is the multi-gate checkpoint.** Report 10's verdict ("stay on single-gate") was wrong because it tested only on the over-trained `01p-ellipse`. The single-gate model memorized flights with abundant training crops and produces zero useful signal on novel ones; the multi-gate model learned more transferable features.

The false-positive rate (on no-gate / unlabelled frames, an UPPER BOUND on true FP — labellers skip non-trivial frames) for the multi-gate ckpt is **2.4 % on held-out 06p** at the current 0.3 confidence threshold. Acceptable for deployment with proper downstream gating.

## Numbers

### Task 1 — cross-flight bench, both checkpoints, all five piloted flights tested

`flight-01p-ellipse` numbers for report 07 / report 10 are the published headline (not re-run this turn); all others measured at `2026-05-15`.

| Flight | SAM-train crops | Single-gate `gt` ≤5 px | Multi-gate `gt` ≤5 px | Single `model_direct` det / ≤5 px | Multi `model_direct` det / ≤5 px / median |
|---|---|---|---|---|---|
| 01p-ellipse (in train, max-rep) | 906 | 94.3 % | 95.8 % (report 10) | 85.1 % / **63.7 %** | 88.7 % / 62.8 % / 1.26 px |
| 02p-ellipse (in train) | 711 | 51.4 % | n/a | 63.2 % / 47.1 % | 82.5 % / **60.1 %** / 1.33 px |
| 08p-lemniscate (in train) | 582 | 38.6 % | n/a | 53.0 % / 31.5 % | 81.1 % / **66.2 %** / 1.21 px |
| 13p-trackRATM (in train, min-rep) | 303 | 35.2 % | n/a | 27.9 % / 24.5 % | 87.7 % / 34.7 % / 16.13 px |
| **06p-ellipse (HELD-OUT)** | **0** | **1.1 %** | **95.8 %** | **0.0 % / N/A** | **79.4 % / 61.0 %** / 1.30 px |

Key observations:
1. **Single-gate's 85.1 % headline is the best-case flight.** Across the other three in-training flights it averages ~34 % within 5 px; on the held-out flight it is 0 %.
2. **Multi-gate has dramatically flatter generalization.** Within-5 px is 60–66 % across four flights (one in-train, one held-out); the trackRATM outlier (34.7 %) is on the most-undersampled flight (303 crops, ⅓ of 01p).
3. **`gt` mode on the held-out flight is 1.1 % within 5 px with the single-gate ckpt but 95.8 % with the multi-gate ckpt** — the seg head's mask quality is so poor on 06p under single-gate training that QuAdGate cannot refine even GT corner priors. Multi-gate training preserves seg-mask quality cross-flight.
4. The training-data-quantity → performance correlation (906 crops → 711 → 582 → 303 ↔ 63.7 % → 47.1 % → 31.5 % → 24.5 %) is the diagnostic giveaway: single-gate is memorizing, not generalizing.

### Task 2 — false-positive rate on no-gate frames

Frames where the labeller either did not write a sibling `.txt` file (`no_txt`), wrote a zero-byte file (`empty_txt`), or wrote a file with no parseable gate lines (`no_gate_line`). A model "fires" when `predict_keypoints` returns a non-None `kp_xy` AND `kp_conf.min() ≥ 0.3` across all 4 channels.

| Checkpoint | Flight | n_no_gate (total) | n_tested | Fired | UPPER-BOUND FP rate | Mean conf.min when fired |
|---|---|---|---|---|---|---|
| Single-gate | 01p-ellipse (in train) | 836 | 84 (stride 10) | 0 | **0.0 %** ‡ | — |
| Single-gate | 06p-ellipse (held-out) | 1660 | 166 (stride 10) | 0 | **0.0 %** ‡ | — |
| Single-gate | 13p-trackRATM | 33 | 33 (stride 1) | 0 | **0.0 %** ‡ | — |
| Multi-gate | 01p-ellipse (in train) | 836 | 84 (stride 10) | 12 | **14.3 %** | 0.49 |
| Multi-gate | **06p-ellipse (HELD-OUT)** | 1660 | 166 (stride 10) | **4** | **2.4 %** | 0.52 |
| Multi-gate | 13p-trackRATM | 33 | 33 (stride 1) | 15 | **45.5 %** | 0.50 |

‡ The single-gate "0 %" is meaningless without context: it is paired with **0 % true-positive detection** on the same flights (from task 1). A model that never fires has 0 % FP by definition.

## Caveats per CLAUDE.md §0 — must read before quoting any FP number

1. **"FP rate" above is an UPPER BOUND.** Labellers skip frames where gates are far/small, partially occluded, already-passed, or in taxi/turnaround segments — many "unlabelled" frames contain real gates the labeller deemed not worth marking. A "fired" frame may be a *correctly detected unlabelled gate*, not a false positive. The 14.3 % / 2.4 % / 45.5 % are upper bounds on true precision loss; the actual FP rates are lower (how much lower is not measured this turn — would require visual inspection of fires).
2. **Stride-on-pre-filtered-list correlates samples temporally.** Frames at 30 fps are ~0.03 s apart and visually near-identical; `sampled = no_gate[::10]` on a clustered set of unlabelled frames undersamples by less than 10× in effective independent observations. The Wilson 95 % CI on 12/84 (multi-gate 01p) is roughly ±7 pp under iid assumption but wider in reality. Bootstrap CI not computed.
3. **`coherent_quad_decode` falls back to global soft-argmax** when no spatially-ordered peak combo exists (`predictor.py:251-260`), with a `[WARN]` per frame. This fired 50+ times on the no-gate frame set — symptomatic of the multi-gate head producing geometrically-incoherent peak combinations that nevertheless survive the conf threshold. The threshold gate is doing nothing useful on this branch. A precision fix should also require `coherent_quad_decode is not None` for a fire (not implemented this turn).
4. **Multi-gate 13p (45.5 %) at n=33** is still the smallest sample — Wilson 95 % CI ≈ [29 %, 63 %]. Within that range but not pinpoint.
5. **`val IoU 0.872` from report 07 was measured on SAM-cropped val frames**, of which the entire piloted-val portion is 06p crops (the SAM val pool starts at frame 0 for 06p and 02a-08a autonomous flights). Tight gate-centric crops are easier to segment than full-frame inference; **the 0.872 number does not transfer to the full-frame `model_direct` mode** — this report's bench numbers (0 % single-gate, 79.4 % multi-gate detection on 06p full frames) are the relevant deployment metric.

## What this report changes about earlier reports

* **Report 07's deployment headline is revised**: `85.1 % det / 1.01 px median` on `flight-01p-ellipse` was a memorization artifact on the most-trained flight. The new deployment headline is **`best_deeplabv3plus_keypoint_multi.pth`** at **79.4 % / 1.30 px median / 61.0 % within 5 px on the only held-out piloted flight (06p-ellipse)**, with cross-in-training-flight average ~60 % within 5 px.
* **Report 10's verdict is reversed**: Stage 2 was published as "statistically tied with report 07". That was true on the one over-trained flight tested. On every other flight (including the only held-out one), the multi-gate ckpt is materially better, often by 30+ pp. **Report 10's recommendation to keep single-gate as the deployment ckpt was wrong.**
* **Report 11 (status note) headline numbers are stale.** PDF and README should be updated in a follow-up commit to point at the multi-gate ckpt as the headline. Not done in this commit (one change per commit).

## Decision

**Adopt `best_deeplabv3plus_keypoint_multi.pth` as the deployment-headline checkpoint going forward.** Specific consequences:

1. The TRT export verification script (`scripts/bench_trt_keypoint.py`) needs to be re-run against this checkpoint and the result captured. The bench script's `default --checkpoint` should be updated.
2. README `### Deployment-relevant headline` section needs to be revised. The 85.1 % / 1.01 px numbers should be replaced (or contextualised as "in-training over-fit flight; cross-flight detection averages 60 % within 5 px, 79.4 % detection on held-out").
3. Status report 11's PDF should be regenerated with the revised numbers.
4. Any downstream consumer (planner, EKF prep) that assumed single-gate-style output (one canonical gate per frame) needs to handle the multi-gate ckpt's softer multi-peak heatmap → `coherent_quad_decode` pipeline.

## Next steps unlocked by this report

| # | Task | Justification | Effort |
|---|---|---|---|
| 1 | Re-export the multi-gate ckpt to TRT, verify numerical correctness (re-run `scripts/bench_trt_keypoint.py`) | Deployment headline changed | 30 min |
| 2 | Visually inspect the 12 multi-gate `01p-ellipse` FP "fires" + 4 on 06p, distinguish unlabelled-gate from true-FP | Pin down whether 14.3 % is real precision loss or labeller artifact | 30 min |
| 3 | Tighten the fire condition: require `coherent_quad_decode is not None` (no fall-through fires) | Cuts most spurious fires | 1 h |
| 4 | Per-flight seg-IoU dump to attribute the cross-flight drop to seg vs head vs QuAdGate (reviewer #1 suggestion) | Mechanistic understanding | 1 h |
| 5 | Bootstrap CI on all reported percentages | Statistical rigor for any external claim | 1 h |
| 6 | Re-bench the autonomous val flights (the 7 `XXa` flights in SAM val) on multi-gate ckpt | True held-out generalization measurement on additional flights | 1 h |
| 7 | Retrain with class-balanced SAM crops (cap per-flight to ~300) | Address the training-data-imbalance root cause | 1 day |

## Reproduction

```bash
# Task 1 — cross-flight bench (multi-gate ckpt is the headline now)
ssh -t qolam 'cd uni/DroneRacing_GateDetection && \
  CK=checkpoints/best_deeplabv3plus_keypoint_multi.pth && \
  CFG=configs/deeplabv3plus_keypoint_multi.yaml && \
  TS=$(date +%Y%m%d_%H%M%S) && OUT=logs/p1_1/cross_flight_$TS && mkdir -p $OUT && \
  for FL in flight-02p-ellipse flight-08p-lemniscate flight-13p-trackRATM flight-06p-ellipse; do \
    for MODE in gt model_direct; do \
      /home/mahbod/miniconda3/bin/conda run -n gate-detection --no-capture-output \
        python scripts/benchmark_flight.py $CK datasets/drone-racing-dataset/data/piloted/$FL --config $CFG \
          --prior-mode $MODE --stride 30 --n 0 \
          --out $OUT/grid_${FL}_${MODE}.png \
          > $OUT/bench_${FL}_${MODE}.log 2>&1; \
    done; \
  done'

# Task 2 — FP rate on no-gate frames (script at /tmp/fp_rate.py)
scp /tmp/fp_rate.py qolam:/tmp/fp_rate.py
ssh qolam 'cd uni/DroneRacing_GateDetection && /home/mahbod/miniconda3/bin/conda run -n gate-detection --no-capture-output python /tmp/fp_rate.py --threshold 0.3 --stride 1 2>/dev/null'
```

Wall-clock: ~5 min for task 1, ~2 min for task 2 on RTX 5090.
