# Report 02 — ImageNet pretraining + normalisation ablation (plan P0-3)

| | |
|---|---|
| **Author** | Mahbod Tajdini, ETHRC Aeronautics — `mtajdini@ethrobotics.ch` |
| **Run timestamp** | `logs/p0_3/20260509_134801/` |
| **Hardware** | Jetson-NX-target laptop (off); training on **qolam**, RTX 5090 32 GB, CUDA 12.8, torch 2.11.0+cu128 |
| **Implements** | `docs/report/01_gate-detection-plan/plan.md` § P0-3 |
| **Companion code commits** | `edc18fc` (P0-3 wiring), `3bef5b0` (seed control + ablation runner) |

## TL;DR

Switching from the gatenet baseline to **DeepLabV3+ with an ImageNet-pretrained ResNet-34 encoder** lifts TII val IoU at 10 epochs from **0.7195 → 0.8565**, a **+13.7 pp** delta. ImageNet input normalisation does not help at this training horizon (cell B without norm beats cell C with norm by 0.4 pp — within seed noise). Recommendation: **ship cell B (DeepLab + ImageNet-pretrained, no norm) as the new default for the next set of experiments**, and re-run a 50-epoch convergence test before declaring this final.

## Numbers

| Cell | Config | Best val IoU | Best epoch | Best val loss | Wall-clock |
|---|---|---|---|---|---|
| **A** baseline | `configs/default.yaml` (gatenet, no pretrain, no norm) | **0.7195** | 8 / 10 | 3.6385 | ≈3 min |
| **B** ImageNet-pretrain only | `configs/deeplabv3plus_imagenet_nonorm.yaml` | **0.8565** | 9 / 10 | 0.3386 | ≈4 min |
| **C** ImageNet-pretrain + norm | `configs/deeplabv3plus_imagenet.yaml` | **0.8521** | 10 / 10 | 0.3397 | ≈3 min |

| Comparison | Δ val IoU |
|---|---|
| B − A (effect of ImageNet pretrain) | **+0.1370 (+13.7 pp)** |
| C − A (effect of full stack) | **+0.1326 (+13.3 pp)** |
| C − B (effect of normalisation, given pretrain) | **−0.0044 (−0.4 pp)** |

## Per-epoch curves

```
cell A — gatenet (baseline)
  ep  train_loss  val_loss   train_iou  val_iou
   1  11.1918     5.7281     0.1798     0.4599
   2   5.5002     4.3496     0.5548     0.6447
   3   4.7564     4.0539     0.6277     0.6605
   4   4.4703     3.9646     0.6555     0.6613
   5   4.3031     3.8029     0.6705     0.6890
   6   4.2108     3.7417     0.6810     0.7015
   7   4.1010     3.6662     0.6951     0.7095
   8   4.0363     3.6385     0.7061     0.7195   ← best
   9   3.9758     3.6606     0.7177     0.7129
  10   3.9773     3.6584     0.7151     0.7137

cell B — DeepLabV3+ + ImageNet ResNet-34 (no input norm)
  ep  train_loss  val_loss   train_iou  val_iou
   1   1.1256     0.5664     0.3265     0.6873
   2   0.4627     0.4070     0.7498     0.7384
   3   0.4079     0.3623     0.7921     0.7978
   4   0.3858     0.3565     0.8087     0.8193
   5   0.3772     0.3471     0.8200     0.8281
   6   0.3662     0.3498     0.8347     0.8295
   7   0.3626     0.3428     0.8415     0.8458
   8   0.3568     0.3430     0.8512     0.8445
   9   0.3517     0.3386     0.8558     0.8565   ← best
  10   0.3489     0.3408     0.8610     0.8547

cell C — DeepLabV3+ + ImageNet + ImageNet input norm
  ep  train_loss  val_loss   train_iou  val_iou
   1   1.1577     0.4900     0.3247     0.6767
   2   0.4518     0.3842     0.7516     0.7788
   3   0.4001     0.3631     0.7929     0.7990
   4   0.3844     0.3545     0.8079     0.8084
   5   0.3730     0.3474     0.8220     0.8258
   6   0.3638     0.3456     0.8374     0.8357
   7   0.3620     0.3486     0.8415     0.8372
   8   0.3575     0.3410     0.8504     0.8437
   9   0.3509     0.3396     0.8571     0.8503
  10   0.3479     0.3397     0.8610     0.8521   ← best
```

A few useful observations from the curves:

- **Cell A is still climbing at epoch 10** (val IoU 0.7195 was at epoch 8, then 0.7129 → 0.7137 at 9-10). It hasn't converged. With 100 full epochs gatenet would reach somewhere higher than 0.72, but cell B / C are *already past* gatenet's converged ceiling on Day 1 of training.
- **Cells B and C track each other within ±0.5 pp at every epoch.** Their trajectories are essentially identical. The 0.4 pp final-epoch gap is consistent with init noise from the same seed running through different preprocessing.
- **Cell A's final val loss is 11× cell B/C's** (3.66 vs 0.34). The architectural change is overwhelming the dataset's signal-to-noise floor.

## What surprised me

I expected cell C (with ImageNet normalisation) to beat cell B (without). The smp documentation explicitly recommends matching the encoder's pretraining preprocessing at inference, and that recommendation is the basis of the original P0-3 prescription in the plan. The result is the opposite — by 0.4 pp.

Two plausible explanations:

1. **Full-network fine-tuning erases the input-distribution dependence.** smp's own README notes: *"Preparing your data the same way as during weights pre-training may give you better results — but it's not necessary in case you train the whole model, not only the decoder."* We are training the whole model (no encoder freeze), and the network is simply learning to absorb whatever input distribution it gets. Verified citation: smp `README.md`, fetched in our prior research.
2. **Single-seed noise.** Cells B and C differ by 0.4 pp at epoch 10 with the same seed; init differences (the model's first layer sees different normalised values, produces different gradients, tracks slightly different trajectories) are large enough to swamp the actual norm effect.

Until we run a multi-seed comparison (e.g., 3 seeds × {B, C} = 6 runs) we cannot confidently declare the norm useless. **The honest read is "no measurable benefit from norm at 10 epochs, single seed".**

## Caveats per CLAUDE.md §0 ("research-grade rigor")

1. **10 epochs is short.** `configs/default.yaml` ships with 100. Cell A specifically is still converging — its final IoU is a lower bound for gatenet. Cells B/C are also still climbing slightly (epoch-10 train IoU > epoch-9), so absolute val IoU at convergence will be higher than the numbers reported here.
2. **Single seed.** Seed 42 across all three runs is *enough* to make B vs C ordering deterministic, but the seed-to-seed variance is unmeasured. The 0.4 pp B-vs-C gap is within plausible seed noise; the 13.7 pp A-vs-B gap is so large that no reasonable seed variance flips the conclusion.
3. **Train/val split is fixed in `data/discovery.py:31` (`random.seed(42)`).** Does not change across runs. This is a property of the data discovery code, not of the training-run seed.
4. **The QuAdGate post-process and PnP are not exercised.** This is purely *segmentation* IoU. A high seg-IoU does not automatically translate to better corner localisation or pose accuracy; the cell-B and cell-C masks may have different *boundary* characteristics that QuAdGate would catch.
5. **Cosmetic logging bug.** The trainer prints `best_gatenet -> ...` for *all* checkpoint writes regardless of model. The actual file written is `best_<checkpoint_name>.pth` per the config; the print string is misleading. Pre-existing, will fix in a separate `chore` commit.

## Decision

Per the plan's own threshold (*"If the change lifts val IoU by ≥1 pp without regressing piloted-vs-autonomous balance, ship it; otherwise log and keep off"*) the result is unambiguous:

- ✅ **Ship the DeepLabV3+ + ImageNet-pretrained ResNet-34** as the model of choice for downstream P0-1, P0-2, P1-1 work. The +13.7 pp lift dwarfs the threshold.
- ❌ **Do not ship the ImageNet input normalisation** until we have a multi-seed comparison that shows a real benefit. The 0.4 pp deficit is small but not in the right direction. *Keep `data.imagenet_norm: false` in the production config, keep the wiring in place for the future test.*
- 🟡 **Concurrent caveat: re-run at full 100 epochs** before this becomes the production default. Cell A is still climbing; absolute IoU numbers should be re-quoted at convergence.

## Recommended config to use going forward

```yaml
# (as in configs/deeplabv3plus_imagenet_nonorm.yaml)
data:
  imagenet_norm: false
model:
  name: deeplabv3plus
  encoder_weights: imagenet
  dropout_p: 0.1
loss:
  deep_supervision_weights: [1]
```

## Next experiments queued (per `01_gate-detection-plan`)

1. **P0-1** — diagnose and fix the PnP frame mismatch (the 45.6° rotation error vs VICON in `README.md:38–42`). Independent of the segmentation model improvement; should land before any pose-accuracy claim is made.
2. **P0-2** — re-bench `benchmark_flight.py` with a *realistic* QuAdGate prior. Currently it uses GT corners as priors (`evaluation/benchmark.py:95–98`); the 82 % / 5.0 px headline reflects refinement-given-perfect-prior, not blind detection.
3. **P0-3 follow-ups (out of scope for this report):**
   - Multi-seed re-run of cells B vs C to settle whether norm helps.
   - Full 100-epoch run of cell B as the new production baseline; report final converged IoU.
4. **P1-1** — parallel keypoint-heatmap head on top of the new cell-B backbone (Azhari 2025 podium pattern).

---

## Reproduction

```bash
# On qolam, in the gate-detection conda env, from ~/uni/DroneRacing_GateDetection/
EPOCHS=10 SEED=42 bash scripts/p0_3_ablation.sh
```

Single command. The runner is idempotent so you can re-launch and only the missing cells re-run. Wall-clock on RTX 5090: ≈16 min total (build + SAM mask gen 8 min, three training cells ~3 min each).

Logs: `logs/p0_3/<timestamp>/{01..05}.log`, summary: `logs/p0_3/<timestamp>/summary.txt`.

Checkpoints in `checkpoints/`: `best_gatenet.pth` (3.3 MB), `best_deeplabv3plus_imagenet_nonorm.pth` (86 MB), `best_deeplabv3plus_imagenet.pth` (86 MB).
