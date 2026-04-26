# Architecture

## Pipeline

```
            ┌─────────────────────────────────────────────────────┐
            │  preprocessing                                       │
RGB frame ─►│  BGR→RGB · resize 384×384 · /255 · NCHW float32      │
            └─────────────────────────────────────────────────────┘
                                  │
                                  ▼
            ┌─────────────────────────────────────────────────────┐
            │  segmentation backbone                               │
            │  → SegmentationModelWrapper.forward                  │
            │  → tuple(probability maps), coarse → fine            │
            │  GateNet      : 5 outputs (deep supervision)         │
            │  DeepLabV3+   : 1 output                             │
            └─────────────────────────────────────────────────────┘
                                  │  last entry = full-res sigmoid map
                                  ▼
            ┌─────────────────────────────────────────────────────┐
            │  binarise at threshold (default 0.5)                 │
            └─────────────────────────────────────────────────────┘
                                  │
                                  ▼   ⟵──── 8 prior corners (EKF / GT)
            ┌─────────────────────────────────────────────────────┐
            │  QuAdGate  (derotate · LSD · extend · intersect ·    │
            │             descriptor match · RANSAC affine)        │
            └─────────────────────────────────────────────────────┘
                                  │
                                  ▼
                    8 sub-pixel corners  (outer×4 + inner×4)
```

## Package layout

```
src/gate_detection/
├── paths.py           Repo-relative paths (DATA_ROOT, SYN_ROOT, BG_ROOT, …)
├── config.py          YAML loader (load_config) · device picker (get_device)
├── models/
│   ├── interface.py       ModelConfig (frozen dataclass) + SegmentationModelWrapper
│   ├── factory.py         build_model(cfg) — backbone registry
│   ├── components.py      DoubleConv · Down · Up · OutConv (with optional Dropout2d)
│   ├── gatenet.py         GateNet + build_gatenet
│   └── deeplabv3plus.py   build_deeplabv3plus (segmentation_models_pytorch wrapper)
├── losses/
│   └── segmentation.py    dice_loss (kornia) · gate_loss (multi-scale Dice + BCE)
├── data/
│   ├── labels.py      Format I/O: parse_gates, corners_to_mask, parse_pixel_corners,
│   │                  make_synthetic_label_lines, gates_to_lines, all4_visible,
│   │                  synthesize_outer, assign_corner_types, CORNER_TYPES
│   ├── dataset.py     GateDataset (image + mask pairs from flat directories)
│   ├── transforms.py  Online augmentations: flip · affine · HSV jitter · motion
│   │                  blur · Gaussian noise
│   ├── synthesis.py   Offline augmentation: warp gate · composite over background
│   └── discovery.py   Filesystem walks: flight_dirs, collect_candidates, find_flight_paths
├── postprocessing/
│   └── quadgate.py    detect_corners (+ detect_corners_debug for visualisation)
├── training/
│   └── trainer.py     build_dataloaders · train_epoch · validate · train
├── inference/
│   └── predictor.py   GatePredictor — preprocessing + forward pass wrapper
├── evaluation/
│   ├── metrics.py     compute_iou (torchmetrics) · inner_corner_error
│   ├── iou.py         evaluate_iou (segmentation IoU on the test split)
│   └── benchmark.py   benchmark_flight (end-to-end pipeline on one flight)
└── visualization/
    ├── colors.py      CORNER_COLORS palette
    └── plotting.py    plot_inference_grid · plot_iou_grid · plot_flight_grid
```

## Model registry

Backbones are constructed through a single factory:

```python
from gate_detection.models import ModelConfig, build_model

cfg   = ModelConfig(name="deeplabv3plus", encoder_name="resnet34",
                    encoder_weights="imagenet", dropout_p=0.1)
model = build_model(cfg)         # → SegmentationModelWrapper
```

The wrapper normalises every backbone to a single contract:

```
forward(x: NCHW float)  →  tuple(probability maps),  ordered coarse → fine
                           model.ds_weights : tuple(float)  loss weights, same length
```

Trainer / evaluator / predictor only ever see this tuple, so adding a new
backbone (e.g. UNet++, SegFormer) is a one-file change in `models/`.

`ModelConfig` is a frozen dataclass that captures every architectural field
that affects the state-dict layout — including `dropout_p`, which gates an
extra `Dropout2d` wrapper on the DeepLabV3+ segmentation head. The trainer
saves it inside the checkpoint, and `predictor.load_checkpoint` restores it
so any saved model rebuilds exactly. CLI args act as fallbacks for legacy
checkpoints that lack a saved config.

## GateNet

A standard encoder–decoder U-Net, narrowed by a channel divisor `f` (default
`f=4`, ~¼ the channels of the original U-Net). All convolutions use
`BN → ReLU`; an optional `Dropout2d(p)` is appended inside each `DoubleConv`
when `dropout_p > 0`. **Skip connections are fused by addition**, not
concatenation — this keeps the decoder narrow.

| Stage    | Block        | Output ch (f=4) | Spatial   |
| -------- | ------------ | --------------- | --------- |
| inc      | DoubleConv   | 16              | 384×384   |
| down1    | MaxPool→DC   | 32              | 192×192   |
| down2    | MaxPool→DC   | 64              |  96×96    |
| down3    | MaxPool→DC   | 128             |  48×48    |
| down4    | MaxPool→DC   | 128             |  24×24    |
| up1      | TConv→Add→DC | 64              |  48×48    |
| up2      | TConv→Add→DC | 32              |  96×96    |
| up3      | TConv→Add→DC | 16              | 192×192   |
| up4      | TConv→Add→DC | 16              | 384×384   |

Five deep-supervision heads (`outc0…outc4`) sit on the bottleneck and on
each decoder stage. Each is `1×1 conv → sigmoid` producing a single-channel
probability map at that scale.

* **At deployment:** only `y4` (full-resolution) is used.
* **At training time:** all five outputs feed the loss.

Weights are Xavier-initialised. `build_gatenet` wraps the model in
`SegmentationModelWrapper(ds_weights=(4, 2, 1, 1, 1))`.

## DeepLabV3+

Built via [`segmentation_models_pytorch.DeepLabV3Plus`](https://github.com/qubvel-org/segmentation_models_pytorch).
Uses an Atrous Spatial Pyramid Pooling (ASPP) decoder over a configurable
encoder (default `resnet34`, optional ImageNet pretraining). When
`dropout_p > 0`, the segmentation head is wrapped in
`Sequential(Dropout2d, original_head)` — this changes the state-dict layout
(see `model_config.dropout_p` round-tripping above).

Single-output model, so `ds_weights = (1.0,)`. The loss falls back to
single-scale Dice + BCE.

## Deep supervision

For GateNet, the loss is computed on the predicted mask **at every output
scale**. The per-scale ground-truth mask is obtained by
`F.interpolate(mask, size=pred.shape[2:], mode="nearest")` — nearest-neighbour
to keep labels strictly binary at every resolution.

Why: it provides direct gradient signal to early layers (faster convergence)
and forces the encoder bottleneck to already encode gate location.

For single-output backbones (`deeplabv3plus`), the mechanism collapses to
a one-term loss automatically — `train` overrides the deep-supervision
weights with `(1.0,)` based on `model.ds_weights`.

## Loss

For each output scale `i`:

```
L_i  =  Dice(pred_i, gt_i)   +   bce_weight · BCE(pred_i, gt_i)
```

The total loss is a weighted sum:

```
L_total = Σ_i  w_i · L_i
            w  = (4, 2, 1, 1, 1)  for GateNet     (default)
            w  = (1,)             for DeepLabV3+
            bce_weight = 2.0
```

The high weight on `w_0` (coarsest scale) emphasises supervision where the
gate occupies a large fraction of the feature map; equal weights on
`w_2..w_4` keep gradient signal to the upper decoder layers without
dominating.

Dice is computed via `kornia.losses.dice_loss` with the binary foreground
expanded to a 2-channel `(bg, fg)` distribution to match Kornia's
multi-class API. BCE is wrapped in
`torch.autocast(enabled=False)` and forced to `float32` because
`F.binary_cross_entropy` is not safe under bf16/fp16 autocast (only the
`*_with_logits` variant is).

Implementation: [`gate_detection.losses.segmentation`](../src/gate_detection/losses/segmentation.py).

## Training stack

The trainer uses the conventional ML toolchain:

| Concern                  | Mechanism |
| ------------------------ | --- |
| Mixed precision          | `torch.autocast` with bf16 (default) or fp16; `GradScaler` enabled only for fp16 on CUDA. |
| Memory layout            | `channels_last` end-to-end (model + each batch). |
| Graph compilation        | `torch.compile(mode="max-autotune")`. The compile wrapper adds an `_orig_mod.` prefix to state-dict keys; the predictor strips it on load. |
| Optimizer                | AdamW or SGD (momentum + weight decay). Selected by `train.optimizer`. |
| LR schedule              | Polynomial decay with linear warmup, via HuggingFace `get_polynomial_decay_schedule_with_warmup`. Per-step (not per-epoch). |
| Online metric            | `torchmetrics.classification.BinaryJaccardIndex` updated each batch and shown in the tqdm bar. |
| Checkpoint selection     | **Highest val IoU** (tie-break: lower val loss). The best CPU-resident state-dict is held in memory and written once at the end of training to `checkpoint_name`. |
| Periodic snapshots       | Every `checkpoint_interval` epochs, an additional `<stem>_epochNNN.pth` is written. |
| Reproducibility knobs    | `torch.set_float32_matmul_precision("high")`, `torch.backends.cudnn.benchmark = True`. |

Top-level entry point: `gate_detection.training.trainer.train(...)`.
The CLI [`scripts/train.py`](../scripts/train.py) is a thin wrapper that
loads `configs/default.yaml`, applies CLI overrides, and forwards every
parameter as a keyword argument.

### Checkpoint format

```python
torch.save({
    "epoch":        int,
    "model":        state_dict,                # raw backbone keys (no _orig_mod.)
    "model_config": {                          # round-trips ModelConfig
        "name", "width_factor", "encoder_name",
        "encoder_weights", "dropout_p",
    },
    "val_loss":     float,
    "val_iou":      float,
    # interval snapshots also include train_loss, train_iou, optimizer
}, ckpt_path)
```

`predictor.load_checkpoint` is tolerant of:
* `_orig_mod.` prefixes from `torch.compile`,
* legacy checkpoints saved as `model.<...>` (wrapper-prefixed) or as the
  raw backbone state-dict (no `model.` prefix), and
* missing `model_config` (uses CLI fallbacks, which is how older
  GateNet checkpoints still load drop-in).

## QuAdGate (corner extraction)

Classical, interpretable, no extra neural network. Operates on the binary
mask plus 8 prior corner positions (in `CORNER_TYPES` order).

```
1. Derotate the mask by the drone roll → world-up = image-up.
   (After this step "TL", "TR", … directions are well-defined image directions.)

2. LSD line detection on the binary mask → set of segments.

3. Extend each segment ×5/3 from its centre, then clip to the image bounds.
   (Reason: nearby-but-not-touching gate edges become intersecting once extended.)

4. Pairwise intersection of all extended lines → corner candidates
   (intersections outside the image are dropped).

5. For each candidate, sample a 4-bit local descriptor:
       [vTL, vTR, vBR, vBL] = mask values at ±h_dir·step ±v_dir·step
   where (h_dir, v_dir) are the two contributing line directions, canonicalised
   so h_dir → +x and v_dir → +y.

6. Match each prior to the nearest candidate whose descriptor matches the
   corner's canonical pattern (PRIOR_DESCRIPTORS):

       outer_TL = [0,0,1,0]   inner_TL = [1,1,0,1]
       outer_TR = [0,0,0,1]   inner_TR = [1,1,1,0]
       outer_BR = [1,0,0,0]   inner_BR = [0,1,1,1]
       outer_BL = [0,1,0,0]   inner_BL = [1,0,1,1]

   (Outer corners are convex out of the ring → one mask-bit set.
    Inner corners are convex in           → three mask-bits set.)
   Reject the match if dist(prior, candidate) > max_dist.

7. Fit a 4-DOF (rotation + uniform scale + translation) RANSAC affine
   priors → matched candidates. Reject if |tx| or |ty| exceeds max_translation.

8. Apply the affine to all 8 priors → refined corners (this fills in any
   priors whose descriptor match failed in step 6).

9. Un-derotate back into the original image frame.
   Reject the whole result if any refined corner falls outside the image.
```

Returns `None` on any failure mode above. `detect_corners_debug` returns a
dict with every intermediate (mask_rot, lines, lines_ext, cand_pts,
cand_descs, matches, M_affine, result) used by
`scripts/inference.py --qag-debug`.

Full implementation: [`gate_detection.postprocessing.quadgate`](../src/gate_detection/postprocessing/quadgate.py).

### Corner ordering

Throughout the codebase the canonical order is

```python
CORNER_TYPES = ["outer_TL", "outer_TR", "outer_BR", "outer_BL",
                "inner_TL", "inner_TR", "inner_BR", "inner_BL"]
```

* Outer corners come first, then inner.
* Within each ring: TL, TR, BR, BL (clockwise from top-left).
* `assign_corner_types(outer, inner)` sorts unordered (4,2)+(4,2) inputs
  into this order using the rule

  ```
  TL = argmin(x + y)   TR = argmax(x - y)
  BR = argmax(x + y)   BL = argmin(x - y)
  ```

## Configuration

Hyperparameters live in [`configs/default.yaml`](../configs/default.yaml) and
are loaded by `gate_detection.config.load_config()`. Every CLI script accepts
`--config path/to/other.yaml`; individual flags (e.g. `--epochs`) override
single fields on top of the YAML.

Paths are resolved by `gate_detection.paths` relative to the repo root and
can be overridden at runtime via env vars (`DATA_ROOT`, `CHECKPOINTS_ROOT`,
`CONFIGS_ROOT`).
