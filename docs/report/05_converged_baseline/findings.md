# Report 05 — Converged baseline for DeepLabV3+ + ImageNet ResNet-34

| | |
|---|---|
| **Author** | Mahbod Tajdini, ETHRC Aeronautics — `mtajdini@ethrobotics.ch` |
| **Date** | 2026-05-09 |
| **Run** | `logs/converge_20260509_150216.log` on qolam (RTX 5090, ~15 min wall-clock) |
| **Config** | `configs/deeplabv3plus_imagenet_nonorm.yaml` |
| **Seed** | 42 (`torch.manual_seed` + `numpy` + `random`, per `c19b4a8` predecessor commits) |
| **Companion** | `docs/report/02_imagenet-ablation/findings.md` (the 10-epoch ablation that motivated this run) |

## TL;DR

DeepLabV3+ with ImageNet-pretrained ResNet-34 trained to **100 epochs** at fixed seed 42 reaches **val IoU 0.8840** (val loss 0.4622) at epoch 90. That is +2.75 pp over the 10-epoch number reported in report 02 (0.8565), confirming the lift is robust at convergence. Total lift over the gatenet baseline (val IoU 0.7195) is **+16.45 pp**.

This is the new production checkpoint for downstream P0-1 / P0-2 / P1-1 work: `checkpoints/best_deeplabv3plus_imagenet_nonorm.pth` on qolam (86 MB).

## Numbers

| Cell | Epochs | val IoU | val loss | best epoch |
|---|---|---|---|---|
| Gatenet baseline (`configs/default.yaml`) | 10 | 0.7195 | 3.6385 | 8 |
| DeepLab + ImageNet, 10-epoch (report 02) | 10 | 0.8565 | 0.3386 | 9 |
| **DeepLab + ImageNet, 100-epoch (this run)** | **100** | **0.8840** | 0.4622 | **90** |

The val IoU is converged; the val loss starts climbing past epoch ~30 while IoU continues to improve — the classical decoupling between proxy loss and downstream metric in segmentation. The model is overfitting on Dice $+ 2{\cdot}$BCE while the at-threshold IoU keeps sharpening. Trainer's "best by val IoU" selection at `trainer.py:383–390` handles this correctly; the saved checkpoint is the IoU-best one.

## Caveats called out per CLAUDE.md §0

1. **Single seed.** No multi-seed envelope. The +2.75 pp 10-→100 epoch lift could be partly seed-noise; the +16.45 pp lift over gatenet is dominant enough that no plausible seed variance flips it.
2. **TII val split only.** This is the deterministic 28/8 train/val split from `data/discovery.py:31`. We have no published number on the *real-recorded* footage (the 39 % set the README mentions) for this checkpoint — that's blocked on getting the Google Drive data on qolam (P1-2).
3. **Mask quality only.** This number is segmentation IoU. Whether the QuAdGate corner extractor + PnP downstream actually benefit is a separate question (P0-1 / P0-2). A mask whose boundary jiggles by a pixel can fail QuAdGate's 4-bit bit-pattern matching even if IoU is high.
4. **Data-prior caveat from report 02 still applies.** TII detection-rate metrics in `benchmark_flight.py` use GT corners as the QuAdGate prior. Until P0-2 lands, we are reading "refinement-given-perfect-prior", not blind detection. The val IoU here is *not* affected by the prior — it's a per-pixel score — but the downstream pose metrics will be.

## Decision

This becomes the **production training checkpoint going forward**. Specifically:

- All TRT export / latency benchmarks (report 03 series) now run on this checkpoint instead of the 10-epoch one. Latency does not depend on weights, so the report-03 numbers remain valid (tested on the same architecture).
- The `[WARN]` for legacy checkpoints in `predictor.py:158–168` will fire when this new checkpoint is loaded, because runs that pre-dated commit `edc18fc` lack the `data_config` field. That is by design — a single audible nudge per inference session, and the predictor defaults to `imagenet_norm=False` which matches this checkpoint.

## What this report does *not* unlock

- The 39 % real-footage detection rate from `README.md:48` is still the live problem. This report's 88 % is on the same TII training distribution; nothing here directly improves real-footage generalisation. **The next two plan items (P0-1 PnP frame fix, P0-2 realistic-prior re-bench, then P1-2 real-footage Tier 2 benchmark) are what tells us whether the 88 % converged seg-IoU actually moves the needle on the deployment metric that matters.**

## Reproduction

```bash
ssh -t qolam 'tmux new -A -s converge "cd uni/DroneRacing_GateDetection && \
  /home/mahbod/miniconda3/bin/conda run -n gate-detection --no-capture-output \
    python scripts/train.py --config configs/deeplabv3plus_imagenet_nonorm.yaml --seed 42 \
    2>&1 | tee logs/converge_$(date +%Y%m%d_%H%M%S).log"'
```

Wall-clock on RTX 5090: ~15 min for 100 epochs at batch 8, bf16 AMP, `torch.compile(max-autotune)`. The 5090 averages ~65 train iters/sec at 384×384 (~10 s/epoch including val + checkpoint save).
