# Usage

Every CLI script lives in `scripts/` and is a thin wrapper around the
package — they take an optional `--config FILE.yaml` (defaulting to
`configs/default.yaml`) and forward CLI flags as overrides. All
programmatic functionality is also reachable via the package modules.

## Configuration

`configs/default.yaml` — the single source of truth for every
hyperparameter:

```yaml
data:        img_size, syn_subdir, use_aug
model:       name, width_factor, encoder_name, encoder_weights, dropout_p
loss:        deep_supervision_weights, bce_weight
train:       epochs, batch_size, lr, optimizer, sgd_momentum, weight_decay,
             num_workers_train, num_workers_val, checkpoint_interval,
             scheduler:    {type, warmup_ratio, power, lr_end}
             performance:  {compile, compile_mode, amp, amp_dtype,
                             channels_last, cudnn_benchmark,
                             matmul_precision, prefetch_factor}
             checkpoint_name
eval:        threshold
postprocess: max_dist, ransac_threshold, max_translation
augment:     outer_corner_scale, multiplier
```

Override paths with env vars: `DATA_ROOT`, `CHECKPOINTS_ROOT`,
`CONFIGS_ROOT`.

---

## Data preparation

### `scripts/prepare_labels.py`

Walks `data/raw/` + `data/piloted/`, picks frames whose label has all 4
inner corners visible, synthesises outer corners via radial scaling, and
copies image + label into `data/synthetic/<split>/`.

Re-running is **idempotent**: existing files are kept and only the missing
quota is filled. Test-set generation sources only from val-split flights.

```bash
# default quotas: train (2000) + val (500)
python scripts/prepare_labels.py

# generate the test set (300, sourced from val-split flights)
python scripts/prepare_labels.py --split test

# specific split with custom target
python scripts/prepare_labels.py --split train --n 800 --scale 1.4
```

Flags: `--split {train,val,test}+`, `--n <int>`, `--scale <float>`.

### `scripts/viewer.py`

Interactive OpenCV viewer for manual quality control of the curated splits.

```bash
python scripts/viewer.py                       # only new (unreviewed) samples
python scripts/viewer.py --shuffle             # randomised order
python scripts/viewer.py --all --split train val
```

Controls: `SPACE / K` keep · `D` discard (deletes both `.jpg` and
`.txt`) · `B` back · `Q / ESC` quit. The set of reviewed stems is logged
to `data/synthetic/.reviewed`. Discarded images can be re-filled later
with `prepare_labels.py`.

### `scripts/augment_data.py`

Offline augmentation pass. For each curated 29-col sample, generate
`--mult` augmented copies into `<split>_aug/`. Each copy composites the
warped gate over a random indoor background, with HSV jitter and noise.

Idempotent: existing `<stem>_augNN.jpg` files are kept.

```bash
python scripts/augment_data.py                       # 5× over train + val
python scripts/augment_data.py --mult 10 --split train
```

### `scripts/download_backgrounds.py`

Streams the `Andron00e/Places365-custom` HuggingFace dataset and saves
resized 640×480 JPEGs.

```bash
python scripts/download_backgrounds.py               # 300 → data/raw/backgrounds/
python scripts/download_backgrounds.py --n 500 --out backgrounds/
```

Requires `pip install datasets pillow` (already in `requirements.txt`).

---

## Training

### `scripts/train.py`

```bash
python scripts/train.py                              # uses configs/default.yaml
python scripts/train.py --epochs 50 --batch-size 8 --no-aug
python scripts/train.py --config configs/my_experiment.yaml
```

Flags: `--config`, `--epochs`, `--batch-size`, `--lr`, `--no-aug`.

What it does:
1. Builds the backbone via `build_model(ModelConfig(...))` from
   `cfg.model.*`.
2. Optionally moves it to `channels_last` and wraps in `torch.compile`.
3. Builds train / val dataloaders with `persistent_workers` and
   `prefetch_factor`.
4. Builds optimizer (AdamW or SGD) and a polynomial-decay-with-warmup LR
   schedule (HuggingFace transformers).
5. Trains for `epochs` epochs:
   * Each batch runs under `torch.autocast(amp_dtype)`.
   * `GradScaler` is enabled only for fp16 on CUDA.
   * The LR scheduler is **per-step**, not per-epoch.
   * Train + val IoU are tracked online via
     `torchmetrics.BinaryJaccardIndex`.
   * Best model selection: highest val IoU; tie-break = lower val loss.
   * Periodic snapshots every `checkpoint_interval` epochs.
6. Writes the best checkpoint to `checkpoints/<checkpoint_name>` at the
   end of training.

Programmatic equivalent:

```python
from gate_detection.training.trainer import train
train(epochs=50, batch_size=8, model_name="deeplabv3plus",
      encoder_name="resnet34", encoder_weights="imagenet",
      use_amp=True, channels_last=True, compile_model=True)
```

---

## Evaluation

### `scripts/evaluate.py`

Computes per-image segmentation IoU on `data/synthetic/test/` and saves a
visual grid (image · GT mask · prediction prob · prediction binary).

```bash
python scripts/evaluate.py checkpoints/best_gatenet.pth
python scripts/evaluate.py checkpoints/best_gatenet.pth --threshold 0.3 --n 20
python scripts/evaluate.py checkpoints/best_gatenet.pth --out results/iou.png
```

Flags: `--config`, `--n` (rows in grid), `--out`, `--threshold`.

Output (printed): `Mean IoU`, `Median IoU`, `Std`, `Min/Max`, fraction
above {0.50, 0.75, 0.90}.

Programmatic equivalent:

```python
from gate_detection.inference.predictor import GatePredictor
from gate_detection.evaluation.iou import evaluate_iou

pred    = GatePredictor("checkpoints/best_gatenet.pth")
results = evaluate_iou(pred, threshold=0.5, n_visuals=16)
print(results.mean, results.fraction_above(0.75))
```

### `scripts/benchmark_flight.py`

End-to-end benchmark on a single flight directory: GateNet ⇒ binary
mask ⇒ QuAdGate corners (using GT corners as priors), measuring detection
rate and inner-corner pixel error.

```bash
python scripts/benchmark_flight.py checkpoints/best_gatenet.pth \
    data/piloted/flight-01p-ellipse
python scripts/benchmark_flight.py checkpoints/best_gatenet.pth <flight> \
    --stride 10 --out results/bench.png
```

Flags: `--config`, `--stride` (sample every Nth frame), `--n` (rows),
`--out`.

Output (printed): detection rate, mean / median / std corner error in px,
fraction ≤ {5, 10} px, and — when `<flight>_cam_ts_sync.csv` exists — a
distance-binned table (drone-to-nearest-gate distance vs detection rate
and corner error per bin).

Programmatic equivalent:

```python
from gate_detection.evaluation.benchmark import benchmark_flight
b = benchmark_flight(pred, "data/piloted/flight-01p-ellipse", stride=30)
print(b.fps, b.detected_mask.mean(), b.distances.shape)
```

---

## Inference / visualisation

### `scripts/inference.py`

Run the model on a few images and save a matplotlib grid. Two layouts:

* **default** (5 cols): image · GT mask · prediction (overlay) ·
  prediction (B&W) · QuAdGate corners on top of the predicted mask.
* **`--qag-debug`** (7 cols): image · GT mask · prediction · QuAdGate
  step 1 (LSD lines) · step 2 (intersection candidates) · step 3
  (descriptor matches) · step 4 (refined corners on the original image).

```bash
# 8 random samples drawn from data/**/camera_*/
python scripts/inference.py checkpoints/best_gatenet.pth

# more samples, custom output
python scripts/inference.py checkpoints/best_gatenet.pth --n 12 --out infer.png

# every QuAdGate intermediate step
python scripts/inference.py checkpoints/best_gatenet.pth --qag-debug --n 4

# specific images
python scripts/inference.py checkpoints/best_gatenet.pth \
    --images data/piloted/flight-01p-ellipse/camera_flight-01p-ellipse/00081_*.jpg
```

Flags: `--config`, `--images`, `--n`, `--out`, `--qag-debug`. Default
output filename is `inference.png` (or `inference_debug.png` with
`--qag-debug`).

---

## Programmatic API

### Single-image prediction

```python
from gate_detection.inference.predictor import GatePredictor

pred = GatePredictor("checkpoints/best_gatenet.pth")   # restores arch from ckpt
img_rgb, prob = pred.predict_image("frame.jpg")         # (H,W,3) uint8, (H,W) float
```

`GatePredictor` constructor falls back to its `model_name`,
`encoder_name`, `encoder_weights`, `dropout_p`, `width_factor` arguments
only if the checkpoint lacks a saved `model_config` — otherwise the
checkpoint's config is authoritative.

### End-to-end pipeline (segmentation + corners)

```python
from gate_detection.inference.predictor    import GatePredictor
from gate_detection.postprocessing.quadgate import detect_corners
from gate_detection.data.labels             import parse_pixel_corners

pred              = GatePredictor("checkpoints/best_gatenet.pth")
img_rgb, prob_map = pred.predict_image("frame.jpg")
priors            = parse_pixel_corners("frame.txt", img_size=384)[0]
corners           = detect_corners(prob_map, priors)    # (8, 2) or None
```

### QuAdGate with per-step intermediates

```python
from gate_detection.postprocessing.quadgate import detect_corners_debug

debug = detect_corners_debug(prob_map, priors,
                             attitude_angle=0.0,        # use drone roll if available
                             max_dist=100.0,
                             ransac_threshold=5.0,
                             max_translation=150.0)
debug["mask_rot"]   # derotated binary mask
debug["lines_ext"]  # extended LSD segments
debug["cand_pts"]   # all intersection candidates
debug["matches"]    # {corner_type: matched_xy}
debug["M_affine"]   # 4-DOF RANSAC affine
debug["result"]     # (8, 2) refined corners or None
```

### Building a model from scratch

```python
from gate_detection.models import ModelConfig, build_model
cfg   = ModelConfig(name="deeplabv3plus", encoder_name="resnet34",
                    encoder_weights="imagenet", dropout_p=0.1)
model = build_model(cfg)            # → SegmentationModelWrapper
out   = model(x)                    # → tuple(probability maps), coarse → fine
```

### Custom training loop

If you don't want the full `train()` driver, the building blocks are
independently exported:

```python
from gate_detection.training.trainer import (
    build_dataloaders, train_epoch, validate,
)

train_loader, val_loader, _, _ = build_dataloaders(batch_size=16)
loss, iou = train_epoch(model, train_loader, optimizer, device,
                        scheduler=lr_sched, scaler=grad_scaler,
                        use_amp=True, amp_dtype=torch.bfloat16)
val_loss, val_iou = validate(model, val_loader, device,
                             use_amp=True, amp_dtype=torch.bfloat16)
```
