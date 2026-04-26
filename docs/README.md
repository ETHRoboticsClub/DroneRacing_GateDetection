# Documentation

Two-stage gate detector for autonomous drone racing:

1. **Segmentation backbone** (GateNet or DeepLabV3+) produces a binary
   gate-region mask from a 384×384 RGB frame.
2. **QuAdGate** post-processes that mask into 8 sub-pixel gate corners
   (4 outer + 4 inner) used downstream by PnP for pose estimation.

Reference: [MonoRace](https://arxiv.org/abs/2601.15222).

## Map

| Document                         | Read this when you want to … |
| -------------------------------- | --- |
| [architecture.md](architecture.md) | Understand the pipeline, the model registry, the loss, the training stack, and the QuAdGate algorithm. |
| [data.md](data.md)                 | Work with the label format, prepare data, or design augmentations. |
| [usage.md](usage.md)               | Run any of the eight CLI scripts, or use the package programmatically. |
| [reference.md](reference.md)       | Look up what a specific module / function does. |

## 30-second tour

```
RGB frame ─► preprocess ─► segmentation backbone ─► binary mask ─► QuAdGate ─► 8 sub-pixel corners
                                  │                                    ▲
                                  └─ deep supervision (GateNet only)   │
                                                                       └─ 8 prior corners (EKF / GT)
```

* Two backbones are interchangeable, selected by config: a custom
  deep-supervised U-Net (`gatenet`) and `deeplabv3plus` (via
  `segmentation_models_pytorch`).
* Both expose the same interface (`SegmentationModelWrapper`) so the trainer,
  evaluator, and predictor are backbone-agnostic.
* The training stack uses AMP (bf16), `channels_last`, `torch.compile`,
  polynomial LR decay with warmup, and `torchmetrics` for online IoU.
* QuAdGate needs prior corner positions — these come either from a previous
  EKF estimate (deployment) or from the ground-truth label (during
  training-time evaluation).

## Quick install

```bash
pip install -e .                 # editable install
python scripts/inference.py checkpoints/best_gatenet.pth --n 4
```

Override paths via env vars:

```bash
export DATA_ROOT=/mnt/datasets/drone_racing
export CHECKPOINTS_ROOT=/mnt/checkpoints
```
