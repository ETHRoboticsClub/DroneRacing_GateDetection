# Runbook — Multi-seed sweep on the v2 ckpt

Report 14 · Tier 2.5 · prerequisite for claiming any +/− N pp result on
the v2 ckpt is real vs seed noise. Today every "improvement" we report
is single-seed; a 5 pp delta could be inside the seed-noise band and
we wouldn't know.

## Why this matters

Report 13 already taught us this the hard way — the balanced ckpt
beat the unbalanced ckpt by +5 pp on the held-out 06p-ellipse, **but**
the cross-flight bench (auto-XXa) showed −3.8 pp on the same models.
A single seed cannot distinguish "the balanced recipe works" from "the
seed-42 sample happened to land in a good basin". Multi-seed gives a
confidence interval; without it the deployment-headline decision is
guesswork.

## Sweep design

- Three seeds: 42, 7, 314 (no special meaning, just three distinct
  draws).
- Config: `configs/deeplabv3plus_keypoint_multi_balanced_v2.yaml`,
  unchanged. Override via `--seed`.
- One bench-set: the same held-out 06p + autonomous XXa flights used
  in report 13, so seed-noise on v2 is directly comparable to v1.
- Bench metrics per seed: det-rate-within-5px, mean reproj err, Wilson
  95% CI. Aggregate: per-flight (mean ± std) across three seeds.

## Compute budget

- 1 seed ≈ 3.0 h on qolam (RTX 5090).
- 3 seeds × 3.0 h ≈ **9 GPU-hours sequentially**.
- Easy to run overnight while the LOFO sweep is paused or before it
  starts.

## Step-by-step

1. **No code change needed** — `scripts/train.py` already takes
   `--seed`, and the checkpoint name is config-driven. Just override
   `checkpoint_name` per seed:

   ```bash
   for s in 42 7 314; do
       python scripts/train.py \
           --config configs/deeplabv3plus_keypoint_multi_balanced_v2.yaml \
           --seed $s \
           > "logs/v2_seed${s}.log" 2>&1
       # ABOVE OVERWRITES the same checkpoint! Either edit the config to
       # set checkpoint_name to include the seed, or copy after each run:
       mv checkpoints/best_deeplabv3plus_keypoint_multi_balanced_v2.pth \
          checkpoints/best_deeplabv3plus_keypoint_multi_balanced_v2_seed${s}.pth
   done
   ```

   (Better: thread a `--checkpoint-name` CLI override into scripts/train.py.
    ~5 LoC. Worth doing before running the sweep.)

2. **Bench each seed** — `scripts/benchmark_flight.py` for each of the
   9 held-out flights × 3 seeds = 27 bench runs. ~5 min each → ~2.5 h.

3. **Aggregate** — `scripts/sweep_seed_aggregate.py`: per-flight mean
   det-rate across 3 seeds + std. The std is the "seed noise band".
   Any v2-vs-v1 comparison ≤ 2× this std is **not** a signal.

4. **Decision gate** — report the std bands explicitly. If v2 beats
   v1 by < 2σ on the held-out 06p, the v2 recipe is NOT validated as
   an improvement; do not promote it to deployment headline.

## Data hygiene

Same val/train discipline as report 13: held-out 06p-ellipse has 0
SAM-train crops (verified); auto-XXa flights have 0 SAM-train crops
(verified). All 3 seeds use the same `per_flight_cap=150` + `cap_seed=42`
to keep the *data* fixed across seeds — the only thing varying is the
PyTorch / numpy / random seed for init + shuffle + aug.

## Verification (must pass before reporting numbers)

- 3 checkpoints exist, ALL DIFFERENT (different val_iou values).
- Per-seed training logs show identical `Train: 2903 samples` lines —
  if a seed landed at a different sample count, `cap_seed` is being
  shadowed and the experiment is invalid.
- If all three seeds agree to 4 decimal places on val_iou, the seed
  isn't actually being honoured anywhere in the data path — investigate.

## When to skip / defer

Skip if v2 evidently fails the smoke test (sanity check) — no point
spending 9 GPU-hours sweeping a recipe we know is broken. Always run
the smoke test (1-epoch dry run with `--epochs 1`) on seed=42 first
and confirm `train_loss` decreases monotonically before kicking off
the full sweep.
