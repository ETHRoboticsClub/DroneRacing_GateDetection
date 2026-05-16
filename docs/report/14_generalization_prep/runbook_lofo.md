# Runbook — Leave-One-Flight-Out cross-validation

Report 14 · Tier 2.3 · prerequisite for any claim that the v2 retrain
*generalises*. Replaces report-13's single held-out 06p-ellipse with an
N-fold sweep so we know the per-flight detection-rate variance band
**before** competition data arrives. Without this, any +/− 3 pp delta
on the held-out is uninterpretable.

## Why this matters

The SAM train pool spans 28 flights after `per_flight_cap=150`. Today's
"held-out" is exactly one flight (06p-ellipse, 0 train crops). That
gives us a single noisy point estimate of OOD performance. LOFO turns
it into 28 point estimates → per-flight mean + variance → a real signal
for which flights are easy / hard / impossible. The hardest flights are
where competition footage will most likely fail.

## Inputs

- v2 config: `configs/deeplabv3plus_keypoint_multi_balanced_v2.yaml`
  - already has `imagenet_norm: true`, `augment_strong: true`, `per_flight_cap: 150`
- Need a **`hold_out_flight: str | null`** plumb so train.py can drop a
  specific flight from the train pool before per_flight_cap is applied.
  → small addition to `data/dataset.py` (filter `all_samples` by
  filename pattern before grouping). Cost: ~15 LoC + 1 reviewer.
- Benchmark harness — `scripts/benchmark_flight.py` already operates
  on one flight at a time, so the per-fold eval is one call per fold.

## Sweep design

- 28 folds, one per flight in the SAM train pool.
- Single seed per fold (multi-seed sweep is a separate runbook — do
  not couple them; that's 28 × 3 = 84 trainings).
- Each fold: train 100 epochs on the other 27 flights, bench on the
  held-out flight.
- Bench metrics per fold: det-rate-within-5px, mean reprojection err,
  Wilson 95% CI.
- Aggregate: mean det-rate across folds + per-flight bar chart.

## Compute budget

- 1 fold ≈ 3.0 h on qolam (RTX 5090, bf16, batch 8, 100 epochs).
- 28 folds × 3.0 h ≈ **84 GPU-hours sequentially**.
- Parallel via tmux + sequential `for` loop (single 5090 cannot
  parallelise within-fold). Or pause between folds for any other
  job that needs the box.
- Budget caveat: this is a one-time investment. After it lands, every
  subsequent claim of "the model generalises to flight X" can quote a
  real bound from the LOFO table.

## Step-by-step

1. **Add `hold_out_flight` plumb** — small surgical edit in
   `data/dataset.py` (filter all_samples by `not flight_re.search(...)
   .group(1) == hold_out_flight`), `trainer.py` (pass through), and
   `scripts/train.py` (read `cfg["data"].get("hold_out_flight")` and
   accept `--hold-out-flight` CLI override). Smoke-test on a 1-epoch
   dry run before committing.

2. **Build a sweep driver** — `scripts/sweep_lofo.sh`:
   ```bash
   #!/usr/bin/env bash
   set -euo pipefail
   FLIGHTS=$(ls datasets/sam_masked/train | grep -oE 'flight-[0-9]+[apAP]?-[a-zA-Z0-9]+' | sort -u)
   for f in $FLIGHTS; do
       python scripts/train.py \
           --config configs/deeplabv3plus_keypoint_multi_balanced_v2.yaml \
           --hold-out-flight "$f" \
           --seed 42 \
           > "logs/lofo_${f}.log" 2>&1
       python scripts/benchmark_flight.py \
           --checkpoint "checkpoints/best_deeplabv3plus_keypoint_multi_balanced_v2_lofo_${f}.pth" \
           --flight "$f" \
           > "logs/lofo_bench_${f}.txt" 2>&1
   done
   ```
   (Checkpoint name needs a `--checkpoint-name-override` to avoid the
   28 folds overwriting each other; default config sets a single name.)

3. **Aggregate** — `scripts/sweep_lofo_aggregate.py`: parse the 28
   bench logs, build a CSV (flight, det_rate, ci_lo, ci_hi, reproj_err,
   n_frames), render a per-flight bar chart with error bars + a
   summary table.

4. **Report findings** — `docs/report/14_generalization_prep/lofo_findings.md`
   with the table, the bar chart, and a paragraph naming the 2-3
   worst-performing flights as the priority for the next aug round.

## Data hygiene

Per CLAUDE.md §0: the held-out flight's SAM val crops are ALREADY in
`<SAM_ROOT>/val` (they were never seen at training time across all
folds — val crops contribute to checkpoint selection only). For LOFO
we additionally exclude them from the val pool of THAT fold so the
checkpoint is selected on flights *other than* the one we're testing.
Otherwise we'd be selecting the model on a flight that's also the
test target — small but real circular signal.

→ Translate to code: in `build_dataloaders`, also pass
`hold_out_flight` to the val GateDataset and filter by it.

## Verification (must pass before reporting numbers)

- 28 ckpts exist, all distinct (different val_iou values; if any two are
  identical, the data-filter plumb didn't actually exclude the fold).
- Each fold's training log shows `Train: N samples` where N ≈ 2750
  (2903 minus ~150 for the held-out flight after cap).
- Per-fold detection rates are NOT all 0.0 or all 1.0 — one of those
  is a bug.
- Sweep total wall-clock ≈ 84 h within ±10 % — outside that, the
  trainer is broken or thermally throttling.

## When to skip / defer

Skip this if competition data lands before the 84 h budget is
available — at that point, real competition footage is the better
held-out and a synthetic LOFO is wasted GPU. Do it only while the
GPU would otherwise be idle.
