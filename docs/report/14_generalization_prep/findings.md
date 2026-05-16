# Report 14 — Generalization preparation (pre-competition-data)

> The competition data hasn't arrived yet. This report covers the work
> done to maximise the chance that whatever ships from TII transfers to
> unseen footage. None of the claims here are validated against
> competition footage — by design, that's what comes next. CLAUDE.md §0
> framing applies: lives depend on this.

## TL;DR

| Tier   | Item                                                                                       | Status   |
|--------|--------------------------------------------------------------------------------------------|----------|
| 1.1    | `imagenet_norm: true` in v2 config (was silent mismatch with ImageNet-pretrained encoder)  | merged   |
| 1.2    | `augment_strong` flag → perspective + brightness/contrast/gamma + cutout + JPEG + HSV±25   | merged   |
| 2.3    | Leave-one-flight-out cross-validation                                                      | runbook  |
| 2.4    | OOD inspection script for unlabelled folders (`scripts/ood_inspect.py`)                    | merged   |
| 2.5    | Multi-seed sweep (3 seeds on v2)                                                           | runbook  |
| 3.6    | Finetune cookbook: `init_from` + `freeze_encoder` + template config                        | merged   |
| 3.7    | `sample_dir_weights` for upweighting a small competition source against the big TII pool   | merged   |

Plus: full 100-epoch v2 retrain **running on qolam** (tmux session
`train_v2`, log at `logs/train_v2_20260516_123622.log`). Started 12:36
local. ETA ≈ 15:30. Smoke (2-epoch dry run) completed cleanly first
and produced a valid checkpoint with all 5 new `data_config` fields
persisted correctly.

## Why these tiers, in this order

The user's stated context: **competition data arrives later**, so any
"is this good?" measurement against TII footage is at best a proxy.
Two questions matter now:

1. **Will the recipe transfer?** Tier 1 (norm + aug) — these are bets
   that close known generalization-failure modes (lighting variation,
   viewpoint variation, occlusion, compression). They are cheap to add
   (one retrain) and easy to revert (one config flag).
2. **Can we react quickly when data lands?** Tier 3 (finetune cookbook +
   per-source weights) — the moment competition labelled frames exist,
   we want a same-day fine-tuned model.

Tier 2 is verification — without it, any number reported on v1 vs v2 is
single-seed and uninterpretable. Multi-seed + LOFO are documented as
runbooks rather than executed because the full sweep would take
~84 GPU-h and is dominated by the LOFO fold count.

## Two reviewer rounds — what they caught

Four agents reviewed Round 1 (config, aug primary, aug independent,
dataset+trainer plumbing). Two agents reviewed Round 2 (finetune
template, ood_inspect.py). Findings:

### Round 1

| Reviewer        | Severity   | Finding                                                                 | Action                                                              |
|-----------------|------------|-------------------------------------------------------------------------|---------------------------------------------------------------------|
| Config v2       | none       | Clean. Two intended deltas present, no drift.                           | —                                                                   |
| Aug (primary)   | none       | Approve. Perspective math, cutout semantics, JPEG round-trip all OK.    | —                                                                   |
| Aug (indep.)    | minor      | Citation conflates Cutout with random erasing (Zhong 2017)              | Comment cleaned up — only Cutout cited                              |
| Aug (indep.)    | note       | JPEG round-trip is heavy CPU; could dominate dataloader at 16 workers   | Acknowledged. Smoke test confirmed throughput is fine                |
| Plumbing        | **CRITICAL**| Interval checkpoint save was missing all 5 new `data_config` fields    | Fixed — both save blocks now include `augment_strong`, `sample_dir_weights`, `extra_train_dirs`, `init_from`, `freeze_encoder` |

### Round 2

| Reviewer         | Severity   | Finding                                                                       | Action                                                  |
|------------------|------------|-------------------------------------------------------------------------------|---------------------------------------------------------|
| Finetune config  | minor      | `TODO_PATH_TO_...` placeholder would silently train without competition data  | Added pre-flight check in `scripts/train.py`: reject `"TODO"` substring + reject non-existent paths |
| Finetune config  | minor      | No rationale for "freeze for ALL 20 epochs" vs unfreeze-after-warmup           | One-line comment added citing Kornblith 2019 §3         |
| OOD inspect      | minor      | Threshold docstring conflates keypoint-conf with seg-mask threshold           | Argparse help text clarified                            |

Both rounds together also revealed that the existing dataset code had a
**latent silent-fallback**: `GateDataset.__init__` does
`if not os.path.isdir(d): continue`, so a stale `extra_train_dirs` path
gets silently skipped. The new pre-flight check in `scripts/train.py`
covers it without touching the dataset.

## Code changes — file:line trail

- New: [configs/deeplabv3plus_keypoint_multi_balanced_v2.yaml](configs/deeplabv3plus_keypoint_multi_balanced_v2.yaml) — `imagenet_norm: true`, `augment_strong: true`, new ckpt name.
- New: [configs/finetune_competition_template.yaml](configs/finetune_competition_template.yaml) — template with TODO placeholders for `extra_train_dirs`, `sample_dir_weights: [1, 30]`, `init_from`, `freeze_encoder: true`, lr 1e-4, epochs 20.
- New: [scripts/ood_inspect.py](scripts/ood_inspect.py) — unlabelled-folder visual sanity tool with grid PNG + CSV output.
- New: [docs/report/14_generalization_prep/runbook_lofo.md](docs/report/14_generalization_prep/runbook_lofo.md), [docs/report/14_generalization_prep/runbook_multiseed.md](docs/report/14_generalization_prep/runbook_multiseed.md) — runbooks for the sweep work that doesn't fit in this session.
- Modified: [src/gate_detection/data/transforms.py](src/gate_detection/data/transforms.py) — added 6 augmenters (`random_perspective`, `random_perspective_kp`, `random_perspective_kp_multi`, `brightness_contrast_gamma`, `random_cutout`, `jpeg_compression`) and threaded `strong: bool` through 3 chains.
- Modified: [src/gate_detection/data/dataset.py](src/gate_detection/data/dataset.py) — new params `augment_strong`, `sample_dir_weights`. Internal 3-tuple `(img_path, secondary_path, src_idx)` threaded through the per-flight-cap path, stripped to 2-tuple at the end.
- Modified: [src/gate_detection/training/trainer.py](src/gate_detection/training/trainer.py) — `build_dataloaders` and `train()` gained `augment_strong`, `sample_dir_weights`, `extra_train_dirs`, `init_from`, `freeze_encoder`. Both checkpoint save blocks (interval + final) persist all 5 new fields.
- Modified: [scripts/train.py](scripts/train.py) — reads 5 new config fields; pre-flight rejects `TODO` placeholders and missing dirs.

## Smoke + remote launch trail

```
12:35   2-epoch dry run on qolam (RTX 5090, gate-detection env)
        result: ckpt @ epoch 2, val_iou=0.0, val_loss=1.117
                data_config: imagenet_norm=True ✓ augment_strong=True ✓
                sample_dir_weights=None ✓ init_from=None ✓ freeze_encoder=False ✓
        → all 5 new fields persisted; aug pipeline runs without crash

12:36   Full 100-epoch retrain launched in tmux session `train_v2`
        log: ~/uni/DroneRacing_GateDetection/logs/train_v2_20260516_123622.log
        early signal: loss 56.1 → 47.5 in ~12 s, IoU 0.057 → 0.074
        GPU at 89% utilisation, 7.5 GB
```

## What is NOT done (and why)

- **LOFO sweep** (~84 GPU-h) — runbook only. Wall-clock dwarfs the
  available session window. Run it manually when the box is otherwise
  idle.
- **Multi-seed sweep** (~9 GPU-h) — runbook only. Should be run AFTER
  the current v2 retrain completes so we have the seed-42 baseline
  to compare against.
- **OOD eyeball on real OOD footage** — script is ready; needs a
  folder of real non-TII drone-racing images. No such dataset is
  pulled locally yet. Suggested sources: RPMRC, AlphaPilot 2019,
  YouTube clips, FalconGym renderings.
- **Competition-data finetune** — template is ready, gated on the
  competition data actually existing. Pre-flight check will fail
  loudly if anyone tries to run the template without filling in the
  TODO path.

## Open risks (CLAUDE.md §0 — surface, don't hide)

1. **v2 may underperform v1** — strong aug can hurt if it overshoots
   the appearance variation actually seen at test time. The runbook for
   multi-seed bench *must* be executed before claiming v2 is an
   improvement. Until then, the deployment headline stays at the
   report-13 unbalanced ckpt.
2. **JPEG round-trip CPU cost** — Reviewer 3 flagged this. Smoke
   throughput looked fine (35-43 it/s in early-epoch warmup), but
   late-epoch fully-compiled throughput on the v1 unbalanced run was
   ~6 it/s sustained. If the strong-aug run drops well below v1's
   sustained throughput, JPEG is the suspect.
3. **Perspective at 8% displacement** — Reviewer 1 noted edge gates
   could have corners pushed OOF; the vis-update logic handles this
   correctly per inspection, but multi-gate scenes with corners exactly
   at the frame boundary are an untested edge case.
4. **`extra_train_dirs` silent-skip** still exists at the dataset
   layer. Pre-flight check at `scripts/train.py` covers the common
   case (running through the CLI). Direct calls to `train()` from
   other entry points would not benefit. Acceptable surface today.

## Priority for the next session

In order — assuming competition data has not yet arrived:

1. **Wait for v2 retrain to complete**, bench it on the report-13
   held-out set (06p + auto-XXa), and compare.
2. **Run the multi-seed sweep** (runbook ready). Get 3 seeds before
   making any deployment decision.
3. **Run OOD inspect** on at least one external OOD folder (download
   one if needed). Confirms the model isn't catastrophically broken
   on non-TII appearance.
4. **Threshold sweep on `keypoint_conf` threshold** (carried over from
   report 13). Decoupled from this report's scope but cheap and
   complementary.
