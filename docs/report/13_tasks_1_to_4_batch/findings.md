# Report 13 — Tasks 1–4 batch from report 12 next-steps (null result on 3 of 4)

| | |
|---|---|
| **Author** | Mahbod Tajdini, ETHRC Aeronautics — `mtajdini@ethrobotics.ch` |
| **Date** | 2026-05-15 |
| **Hardware** | qolam, RTX 5090, gate-detection conda env |
| **Companion** | report 12 (the four open items being attacked here); deployment headline UNCHANGED at end of this report — `best_deeplabv3plus_keypoint_multi.pth` remains |
| **Reviewers** | 4 independent reviewer agents across the batch (CLAUDE.md §8) — caught the BUGs documented inline |

## TL;DR

Attempted tasks 1, 2, 3, 4 from report 12's next-steps. **The deployment headline does not change** at the end of this batch — all four tasks either yielded null/negative results, surfaced a known limitation, or provided informational measurements that don't trigger an architecture change. The headline stays at `best_deeplabv3plus_keypoint_multi.pth` from report 12: 78.9 % detection / 1.30 px median / 61.0 % within 5 px on held-out flight-06p-ellipse, 0.44 ms / 2274 fps TRT FP16.

| Task | Outcome | Action |
|---|---|---|
| 1 — Temporal coherence at decode | **Null / negative** — hurt -2.8 pp within-5 px at stride=30; on consecutive multi-gate frames, locked onto Frankenstein quads | Reverted before commit |
| 2 — Class-balanced retrain (cap=150/flight) | **Mixed / negative on average** — +5 pp on piloted held-out 06p, but **-3.8 pp average** across all 8 held-out flights when including autonomous XXa | Balanced ckpt kept as experimental sibling; deployment headline unchanged |
| 3 — Autonomous held-out bench (7 XXa flights, original multi-gate ckpt) | **Informational** — auto-trackRATM 67-73 % det / 65-69 % ≤5 px (deployment-relevant); auto-ellipse 20-24 % det (blind spot); 08a-lemniscate outlier (35 % / 40 %) | No code action; flagged auto-ellipse as known gap |
| 4 — Visually inspect 4 FP fires on held-out 06p | **Refined the 2.4 % upper bound** — 3 true FPs + 1 ambiguous; all 4 cluster at conf.min ∈ [0.30, 0.41] (just above threshold) | Threshold-sweep work deferred (task 5 from report 12 next-steps) |

The most important meta-finding: the reviewer agents caught **load-bearing errors** that would have otherwise shipped — specifically reviewer #2 on task 2 demanded the auto-XXa bench *before* claiming the +5 pp 06p win as deployment-ready. That cross-check showed the +5 pp was a single noisy point in a -3.8 pp average — exactly the §0 "no toy baselines" failure mode the review process exists to catch.

## Task 1 — Temporal coherence at decode (null result, reverted)

### Approach

Extended `coherent_quad_decode(heatmap, prev_center=None, proximity_weight=0.05)` to optionally bias the score toward combinations whose centroid is closer to the previous frame's predicted center. Plumbed `prev_quad_center` through `predictor.predict_keypoints` (`prev_quad_center: tuple | None = None` kwarg) and `evaluation/benchmark.py` (carries state between frames in the per-flight loop). With `prev_center=None` the decoder behaves identically to the report-12 stateless version — backwards-compatible.

### Bench (held-out 06p, stride=30, multi-gate ckpt)

| Metric | Pre-task-1 | Post-task-1 | Δ |
|---|---|---|---|
| `model_direct` detection rate | 78.9 % (295/374) | 78.9 % (295/374) | unchanged |
| ≤5 px | 61.0 % (172/282) | 58.2 % (164/282) | **-2.8 pp** |
| Mean cerr | 12.21 px | 13.14 px | +0.93 |

### Diagnostic on the 4-frame window A (consecutive frames 500-510, multi-gate scene)

Per-channel reassignment after temporal coherence:

| Frame | Pre-task-1 | Post-task-1 |
|---|---|---|
| 500 | Frankenstein (g0, g1) | Frankenstein (g0, g1) |
| 502 | coherent on g1 | **→ Frankenstein** (g0, g1) — degraded |
| 505 | coherent on g1 | **→ Frankenstein** (g0, g1) — degraded |
| 510 | Frankenstein (g1, g2) | coherent on g2 — improved |

Net: 1 win, 2 losses, 1 unchanged. The temporal-coherence term locked onto the Frankenstein-quad that frame 500 picked first, propagating that decision forward to frames 502 and 505 (which had cleanly chosen g1 in the stateless decode).

### Why it failed

When the first frame in a sequence has a Frankenstein quad, the temporal-coherence term biases subsequent frames toward the same Frankenstein cluster centroid, propagating the error. The naive proximity term cannot distinguish "previous frame picked the right gate" from "previous frame picked an incoherent quad". A non-trivial fix would need either:
- a quality gate on prev_center (only carry forward if the previous quad's score was below some threshold), OR
- a proper tracker (Kalman / state-space) that fuses the heatmap evidence with motion priors.

### Decision

**Reverted.** All three modified files (`losses/keypoint.py`, `inference/predictor.py`, `evaluation/benchmark.py`) restored to their HEAD state. `git diff` empty post-revert. Per CLAUDE.md §0 — "no plausible-looking code without verification". Code change is in this report's git history only.

## Task 2 — Class-balanced retrain (mixed-to-negative; sibling-only)

### Approach

Added `per_flight_cap: int | None = None` and `cap_seed: int = 42` to `GateDataset`. When set, samples are grouped by flight prefix (regex `r"(flight-[0-9]+[apAP]?-[a-zA-Z0-9]+)"`) and deterministically truncated using `random.Random(cap_seed)`. Plumbed through trainer + script + new config (`configs/deeplabv3plus_keypoint_multi_balanced.yaml`, `cap=150, seed=42`). Persisted in both interval-save and final-best-save `data_config` blocks. Val pool intentionally uncapped (val IoU score should reflect the natural val distribution).

### Smoke test (qolam)

- Unbalanced: 4000 samples across 28 flights, max 302 (01p-ellipse), min 39 (autonomous lemniscate).
- Balanced (cap=150, seed=42): **2903 samples** (−27 %), max 150 (11 flights), min 39 unchanged.
- Determinism: re-constructing with the same seed yields identical sample list (verified).

### Training (100 epochs, RTX 5090, seed 42)

- val IoU best: **0.8701** (vs report-10's 0.8704 — within noise).
- val_corner_dist plateau: ~28 px (matches report 10).
- val_kp_loss: 0.0002 (same).
- Checkpoint: `checkpoints/best_deeplabv3plus_keypoint_multi_balanced.pth`.

### Bench — piloted (cross-flight)

| Flight | Held-out? | Unbal ≤5 px | Balanced ≤5 px | Δ |
|---|---|---|---|---|
| 06p-ellipse | YES (0 train crops) | 61.0 % | **66.0 %** | **+5.0** |
| 01p-ellipse | no (most-trained, 302) | 62.8 % | 66.1 % | +3.3 |
| 02p-ellipse | no | 60.1 % | 64.8 % | +4.7 |
| 08p-lemniscate | no | 66.2 % | 68.1 % | +1.9 |
| 13p-trackRATM | no | 34.7 % | 32.7 % | -2.0 |

On piloted, this looked like a clean win — **+5.0 pp on the only held-out piloted flight**. I drafted a "SWITCH the deployment headline" recommendation at this point.

### Bench — autonomous held-out (the load-bearing cross-check)

Reviewer #2 (CLAUDE.md §8) explicitly demanded the auto-XXa cross-check before claiming the result. Numbers (balanced ckpt vs unbalanced report-10 ckpt, same auto-XXa flights as task 3):

| Flight | Trajectory | Unbal ≤5 px | Balanced ≤5 px | Δ |
|---|---|---|---|---|
| 02a-ellipse | auto-ellipse | 73.9 % | 66.7 % | **-7.2** |
| 04a-ellipse | auto-ellipse | 77.3 % | 59.1 % | **-18.2** |
| 05a-ellipse | auto-ellipse | 78.9 % | 57.1 % | **-21.8** |
| 08a-lemniscate | auto-lemniscate | 40.0 % | 40.0 % | 0 |
| 15a-trackRATM | auto-trackRATM | 69.0 % | 69.0 % | 0 |
| 16a-trackRATM | auto-trackRATM | 64.5 % | 70.6 % | +6.1 |
| 18a-trackRATM | auto-trackRATM | 65.6 % | 71.6 % | +6.0 |

Combined with the piloted 06p: **average across 8 held-out flights = −3.8 pp ≤5 px**.

### What the −3.8 pp shows

Balanced training **redistributes generalization, doesn't improve it**. It helps on piloted-ellipse and auto-trackRATM but dramatically hurts auto-ellipse. The +5 pp on 06p was a single noisy point that, in isolation, looked deployment-ready. With the auto-XXa cross-check, the verdict reverses.

### Decision

**Deployment headline UNCHANGED**: keep `best_deeplabv3plus_keypoint_multi.pth` (report 10's original unbalanced ckpt). Balanced ckpt is retained in the repo as `best_deeplabv3plus_keypoint_multi_balanced.pth` for future investigation but is NOT the deployment headline. The reviewer's GATED recommendation is what report 13 ships:

1. Land code change (per_flight_cap is a useful flag for any future retrain) — done.
2. Keep balanced ckpt as experimental sibling — done.
3. Keep report-10 unbalanced as deployment headline until (a) multi-seed (≥ 3) sweep confirms the +5 pp 06p win is outside CI AND (b) auto-XXa numbers don't regress. **Neither precondition met.** Followups deferred.

### What we learned

- The reviewer-agent process is doing exactly what CLAUDE.md §8 designed it to do. Without the reviewer's explicit demand for the auto-XXa cross-check, I would have shipped the balanced ckpt as the deployment headline based on the piloted-only numbers. That would have ground-truth-degraded autonomous-ellipse performance by 7-22 pp in deployment.
- The "balanced training improves generalization" hypothesis is **false at this dataset scale**. The autonomous flights are too under-represented (max 119 crops, average 70) for the cap-based balancing to genuinely equalize the distribution — cap=150 just removes the over-representation tail without addressing the under-representation tail.

## Task 3 — Autonomous held-out bench (informational)

### Approach

Bench the (unbalanced) multi-gate ckpt on the 7 autonomous flights in SAM val. Verified preconditions: all 7 have 0 SAM train crops; all dir paths exist at `data/raw/autonomous/flight-XXa-*`. Honest framing per reviewer #3: "held-out from training weight updates; SAM val crops contributed to checkpoint selection only" — not bare "fully held-out".

### Results (stride 30, seed 42, model_direct mode)

| Flight | Trajectory | Det rate | Median cerr | ≤5 px |
|---|---|---|---|---|
| **06p-ellipse** (piloted held-out, report 12) | ellipse | **78.9 %** | 1.30 | 61.0 % |
| 02a-ellipse | ellipse | 24.2 % | 1.20 | 73.9 % |
| 04a-ellipse | ellipse | 20.5 % | 1.12 | 77.3 % |
| 05a-ellipse | ellipse | 21.5 % | 1.18 | 78.9 % |
| 08a-lemniscate | lemniscate | 35.0 % | 15.58 | 40.0 % |
| 15a-trackRATM | trackRATM | **70.9 %** | 2.31 | 69.0 % |
| 16a-trackRATM | trackRATM | 67.4 % | 2.20 | 64.5 % |
| 18a-trackRATM | trackRATM | 72.7 % | 1.91 | 65.6 % |

### Interpretation

- **Auto-trackRATM is the most deployment-relevant generalisation figure to date** (67-73 % det, 65-69 % ≤5 px). This is the closest trajectory family to the AIGP / A2RL competition layouts.
- **Auto-ellipse is a blind spot** (20-24 % det). When auto-ellipse fires it's *conservatively accurate* (73-79 % ≤5 px on the small detected subset), consistent with the report-12 task-3 tightening that prefers abstention over Frankenstein-quad fallback.
- **08a-lemniscate outlier** (35 % / 40 %, 15.58 px median) is a known generalisation gap worth visual inspection in a follow-up — likely firing on wrong structure.

Per reviewer #3: the trajectory-feature inference is an **observation, not a causal mechanism** — would need controlled ablation (train without trackRATM, test on auto-trackRATM) to assert mechanism. Reported as observation.

### Denominator caveat (reviewer #3)

The ≤5 px denominators are "detections with parseable GT", not "all detections". Per `evaluation/benchmark.py:259-261` and `evaluation/metrics.py:19-34`, a frame can count as detected but have `inner_corner_error == None` (GT corners not parseable). Example: 02a-ellipse 24/99 det → 17/23 ≤5 px (denominator 23, not 24). Footnote this in any quoted percentage.

## Task 4 — Visual inspection of held-out 06p FP fires

### Approach

Re-ran the report-12 FP-rate measurement (4 fires / 166 no-gate frames @ threshold 0.3) and visualised each fire alongside a labelled ±10-frame neighbour to see what real gates look like in the same scene. Replication is correct (reviewer #4 verified).

### Per-fire classification

| Fire | Filename | conf.min | seg max (>0.5 frac) | Classification |
|---|---|---|---|---|
| 1 | 01651_…jpg | 0.30 | 0.86 (0.1 %) | True FP — quad on structural edge, seg barely fires |
| 2 | 02104_…jpg | 0.41 | 0.49 (0.4 %) | True FP — similar |
| 3 | 08184_…jpg | 0.41 | 1.07 (5.5 %) | True FP — quad on upper-left structural feature |
| 4 | 08071_…jpg | 0.30 | 0.93 (**4 %**) | **Ambiguous** — strong seg signal, possibly labeller-skipped real gate |

### Statistical framing per reviewer #4

Per Wilson 95 % CI: 4/166 = 2.4 % point estimate, CI = **[0.7 %, 6.0 %]**. Re-classifying a single sample (3 true FPs + 1 ambiguous) shifts the point estimate but **not the CI meaningfully**. The honest reporting is:

> Upper-bound FP rate on held-out 06p is **2.4 % (4/166, 95 % Wilson CI [0.7 %, 6.0 %])**. Of the 4 fires inspected, 3 appear to be true FPs (weak seg activation, no obvious gate, conf right at threshold); 1 (frame 08071) has strong seg activation and is ambiguous between true FP and labeller-missed gate. All 4 fires have `conf.min ∈ [0.30, 0.41]`, clustered at the lower threshold boundary.

### Threshold-calibration follow-up

Bumping `keypoint_conf_threshold` from 0.3 to 0.5 would eliminate every measured FP (all 4 have conf.min ≤ 0.41). The TP cost is **not yet measured** — that's report-12 next-step #5 (confidence calibration), deferred. Don't claim "0.5 is the right threshold" until the TP-cost sweep is run. Per CLAUDE.md §0, half-finished claims do not ship.

### BUG fixed per reviewer #4

`fp_inspect.py:140` — `cv2.imread` was being called without a None-check, which would crash with an obscure error on a corrupt jpg. Added `[WARN]` per CLAUDE.md §6 (no silent fallbacks). Committed.

## Cross-cutting findings

### CLAUDE.md §0 — "every experimental claim needs a baseline, a metric, and a fixed seed"

- Task 1: baseline ✓, metric ✓, seed ✓. Negative result. Reverted.
- Task 2: baseline ✓, metric ✓, seed ✓ **single seed**. Reviewer #2 flagged this as a §0 violation for a near-noise-floor result (Wilson CI overlap ~6 pp on the +5 pp held-out improvement). Multi-seed sweep deferred; balanced ckpt not promoted.
- Task 3: baseline ✓, metric ✓, seed ✓. Wider held-out picture documented.
- Task 4: baseline ✓, metric ✓, seed ✓. Statistical framing with Wilson CI added per reviewer #4.

### CLAUDE.md §8 — reviewer agents

- Task 1: stateless code change, one reviewer expected; numerical evidence was self-evident negative, didn't need reviewer for the revert decision.
- Task 2 (load-bearing per §8): **two reviewer agents**, both delivered actionable findings. Reviewer #1 caught a stray-typo BUG in `generate_sam_masks.py` (working-tree artifact) and config-comment staleness; reviewer #2 caught the load-bearing methodology gap (auto-XXa cross-check missing) that reversed the task 2 decision.
- Task 3: one reviewer (informational task).
- Task 4: one reviewer (small task); caught the `cv2.imread` silent-fallback BUG.

Without reviewer #2's intervention on task 2, the deployment headline would have been switched on a -3.8 pp average regression. **The §8 review process was load-bearing here in the most literal sense.**

### Data hygiene

All evaluation in this batch used held-out flights:
- Piloted: `flight-06p-ellipse` only (0 SAM train crops, verified on qolam via `ls data/sam_masked/train/ | grep -c "flight-06p"` returning 0).
- Autonomous: 7 XXa flights, all with 0 SAM train crops, verified per-flight.

No training-distribution flight was used to derive any headline number in this report.

## Decisions

| Decision | Status |
|---|---|
| Deployment-headline checkpoint | **UNCHANGED**: `best_deeplabv3plus_keypoint_multi.pth` (report 10's unbalanced) |
| `coherent_quad_decode` `prev_center` extension (task 1) | reverted; not in HEAD |
| `per_flight_cap` flag on `GateDataset` (task 2 code) | shipped; **default `None` preserves existing behaviour** |
| `best_deeplabv3plus_keypoint_multi_balanced.pth` (task 2 ckpt) | kept as experimental sibling on qolam; not the deployment headline |
| `configs/deeplabv3plus_keypoint_multi_balanced.yaml` | shipped as a sibling config |
| Status note `docs/report/11_status/gate_detection_status.tex` | **no change required** — headline numbers still apply to the unbalanced ckpt which remains the deployment ckpt |
| README headline | **no change required** — same reasoning |

## Next steps (revised after this batch)

1. **Multi-seed sweep on the BALANCED ckpt** (3 seeds, 100 epochs each, ~1.5 hours of qolam) to either confirm the +5 pp 06p win is outside CI OR confirm it's a single-seed wobble. **Required before any future "balanced training generalises" claim**. ~1.5 h.
2. **Investigate why autonomous-ellipse generalises so poorly** (24 %, 20 %, 21 % det rates across 3 auto-ellipse held-outs vs 78.9 % on piloted ellipse). Hypothesis: piloted vs autonomous camera trajectory differences not captured in our augmentation. Controlled augmentation ablation. ~1 day.
3. **Threshold-sweep on conf**: bump from 0.3 to 0.4 / 0.5 / 0.6 and measure (a) TP cost on 06p (b) FP rate. The 4 fires cluster at 0.30-0.41; raising to 0.45 might be a sweet spot. ~2 h.
4. **Frankenstein-quad tracker** (revisit task 1 with a proper quality gate): only carry forward `prev_center` when the previous frame's quad-score was below a threshold (i.e., the previous decode was geometrically clean). Avoids the error-propagation that killed task 1 v1. ~3 h.
5. Items 5-10 from report 12's next-steps remain (calibration with KL, per-flight seg-IoU dump, MonoRace input cropping, EKF, ROS2, Orin NX) — none unblocked by this batch.

## Reproduction

All four tasks reproducible from the commit history:
- Task 1 code (reverted) — present in this report's git history only, not in HEAD.
- Task 2 code — present in HEAD; retrain via:
  ```bash
  ssh -t qolam 'tmux new -A -s bal100 "cd uni/DroneRacing_GateDetection && \
    /home/mahbod/miniconda3/bin/conda run -n gate-detection --no-capture-output \
      python scripts/train.py --config configs/deeplabv3plus_keypoint_multi_balanced.yaml --seed 42 \
      2>&1 | tee logs/p1_1/balanced_full_$(date +%Y%m%d_%H%M%S).log"'
  ```
- Task 3 bench (auto-XXa on unbalanced ckpt):
  ```bash
  ssh qolam 'cd uni/DroneRacing_GateDetection && \
    CK=checkpoints/best_deeplabv3plus_keypoint_multi.pth && \
    CFG=configs/deeplabv3plus_keypoint_multi.yaml && \
    for FL in flight-02a-ellipse flight-04a-ellipse flight-05a-ellipse \
              flight-08a-lemniscate flight-15a-trackRATM flight-16a-trackRATM flight-18a-trackRATM; do \
      /home/mahbod/miniconda3/bin/conda run -n gate-detection --no-capture-output \
        python scripts/benchmark_flight.py $CK data/raw/autonomous/$FL --config $CFG \
          --prior-mode model_direct --stride 30 --n 0 \
          --out /dev/null > /tmp/bench_$FL.log 2>&1; \
    done'
  ```
- Task 4: `python /tmp/fp_inspect.py` (now committed at `docs/report/13_tasks_1_to_4_batch/fp_inspect.py`).
