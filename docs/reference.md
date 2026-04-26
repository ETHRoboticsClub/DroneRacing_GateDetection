# Reference

Module-by-module summary of the public API. Source links use repo-relative
paths.

## Top-level

### `gate_detection.paths` — [paths.py](../src/gate_detection/paths.py)

Project paths, all overridable via env vars of the same name.

| Symbol             | Default                                | Purpose |
| ------------------ | -------------------------------------- | --- |
| `REPO_ROOT`        | parent of `src/`                       | repo root |
| `DATA_ROOT`        | `REPO_ROOT/data`                       | datasets root |
| `CHECKPOINTS_ROOT` | `REPO_ROOT/checkpoints`                | model checkpoints |
| `CONFIGS_ROOT`     | `REPO_ROOT/configs`                    | YAML configs |
| `SYN_ROOT`         | `DATA_ROOT/synthetic`                  | curated split dirs |
| `BG_ROOT`          | `DATA_ROOT/raw/backgrounds`            | augmentation backgrounds |
| `DEFAULT_CONFIG`   | `CONFIGS_ROOT/default.yaml`            | default config |

### `gate_detection.config` — [config.py](../src/gate_detection/config.py)

| Function          | Description |
| ----------------- | --- |
| `load_config(p)`  | Load a YAML config; falls back to `DEFAULT_CONFIG`. |
| `get_device()`    | Pick best torch device: `cuda` > `mps` > `cpu`. |

---

## `gate_detection.models` — [models/](../src/gate_detection/models)

### `interface.py`

```python
@dataclass(frozen=True)
class ModelConfig:
    name: str            = "gatenet"        # "gatenet" | "deeplabv3plus"
    width_factor: int    = 4                # GateNet only
    encoder_name: str    = "resnet34"       # SMP backbones only
    encoder_weights: str | None = None      # e.g. "imagenet"
    dropout_p: float     = 0.0              # affects state-dict layout
```

```python
class SegmentationModelWrapper(nn.Module):
    """Backbone-agnostic wrapper.

    forward(x) → tuple(probability maps), coarse → fine (length = 1 for
    single-output models). model.ds_weights is the matching loss-weight
    tuple used by gate_loss.
    """
```

### `factory.py`

```python
def build_model(cfg: ModelConfig) -> SegmentationModelWrapper: ...
```

Resolves `cfg.name` to `build_gatenet` or `build_deeplabv3plus`.
`ValueError` for unknown names.

### `components.py`

| Class        | Notes |
| ------------ | --- |
| `DoubleConv` | `Conv → BN → ReLU → [Dropout2d] → Conv → BN → ReLU → [Dropout2d]`. |
| `Down`       | `MaxPool(2) → DoubleConv`. |
| `Up`         | `ConvTranspose2d(2x) → BN → add(skip) → DoubleConv`. Adds skip via **addition**, not concatenation. |
| `OutConv`    | `1×1 Conv → Sigmoid` (single-channel probability head). |

### `gatenet.py`

```python
class GateNet(nn.Module):
    def __init__(self, f: int = 4, dropout_p: float = 0.0): ...
    def forward(self, x) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: ...
```

```python
def build_gatenet(*, width_factor=4, dropout_p=0.0) -> SegmentationModelWrapper: ...
```

5-output deep-supervision network. Wrapper sets
`ds_weights = (4.0, 2.0, 1.0, 1.0, 1.0)`.

### `deeplabv3plus.py`

```python
def build_deeplabv3plus(
    *, encoder_name="resnet34",
    encoder_weights=None,            # e.g. "imagenet"
    dropout_p=0.0,
) -> SegmentationModelWrapper: ...
```

Builds `smp.DeepLabV3Plus(in_channels=3, classes=1, activation="sigmoid")`.
When `dropout_p > 0`, wraps `segmentation_head` in
`Sequential(Dropout2d, original_head)`. Single output → `ds_weights = (1.0,)`.

---

## `gate_detection.losses` — [losses/](../src/gate_detection/losses)

### `segmentation.py`

```python
def dice_loss(pred, target, eps=1e-6) -> Tensor:
    """Soft Dice via kornia, adapted to GateNet's single-channel sigmoid."""
```

```python
def gate_loss(
    outputs: Sequence[Tensor],   # backbone forward output
    mask: Tensor,                # ground-truth (N, 1, H, W)
    deep_supervision_weights = (4, 2, 1, 1, 1),
    bce_weight = 2.0,
) -> Tensor:
    """Σᵢ wᵢ · (Dice(predᵢ, gtᵢ) + bce_weight · BCE(predᵢ, gtᵢ))."""
```

GT is downsampled per-scale via `F.interpolate(..., mode="nearest")`. BCE is
forced to fp32 inside an `autocast(enabled=False)` block.

---

## `gate_detection.data` — [data/](../src/gate_detection/data)

### `labels.py`

```python
CORNER_TYPES = ["outer_TL","outer_TR","outer_BR","outer_BL",
                "inner_TL","inner_TR","inner_BR","inner_BL"]

parse_gates(lbl_path)                         # → list[dict]
all4_visible(lbl_path)                        # → bool
synthesize_outer(inner_norm, scale=1.4)        # → (4,2) ndarray
make_synthetic_label_lines(lbl_path, scale=1.4)  # → list[str]  (29-col)
gates_to_lines(gates)                          # → list[str]
corners_to_mask(lbl_path, out_size)            # → (H,W) uint8 binary
assign_corner_types(outer, inner)              # → (8,2) sorted CORNER_TYPES order
parse_pixel_corners(lbl_path, img_size, outer_scale=1.4)  # → list[(8,2)]
```

`parse_gates` returns dicts with keys
`{cls, cx, cy, w, h, inner, inner_vis, outer, outer_vis}` where
`outer`/`outer_vis` are `None` for 17-col labels. `corners_to_mask`
produces a ring (29-col) or filled inner (17-col).

### `dataset.py`

```python
class GateDataset(Dataset):
    def __init__(self, sample_dirs: Iterable[str],
                 img_size: int = 384,
                 augment: bool = True): ...
    def __getitem__(idx) -> tuple[Tensor, Tensor]:
        # img_t: (3, H, W) float in [0,1]
        # mask_t: (1, H, W) float in {0,1}
```

Aggregates `<stem>.jpg` + `<stem>.txt` pairs across one or more flat
directories. Applies `transforms.train_augment` when `augment=True`.

### `transforms.py`

Online augmenters used by `GateDataset`. All operate on
`(img: HWC RGB uint8, mask: HW uint8)` pairs and use OpenCV.

| Function            | Effect |
| ------------------- | --- |
| `random_flip`       | Horizontal flip with probability `p`. |
| `random_affine`     | Rotation, translation, scale; `BORDER_REFLECT`. |
| `hsv_jitter`        | Hue shift, sat / val multiplicative scaling. |
| `motion_blur`       | Random directional kernel. |
| `gaussian_noise`    | Additive noise, σ ~ U[0, sigma_max]. |
| `train_augment`     | Default chain: flip → affine → HSV → blur → noise. |

### `synthesis.py`

Offline augmentation pipeline. Composes a warped curated frame onto a
random indoor background.

| Function                       | Effect |
| ------------------------------ | --- |
| `transform_points(pts, M)`     | Apply 3×3 perspective matrix to (N, 2) points. |
| `random_perspective_matrix`    | Random 3×3 perspective for image corners. |
| `random_affine_matrix`         | Random 3×3 affine (rot + scale + translate). |
| `random_background(W, H)`      | One random BGR image from the lazy bg pool. |
| `hsv_jitter_bgr`               | BGR-input variant of `hsv_jitter`. |
| `augment_sample(in_img, in_lbl, out_img, out_lbl)` | Write one augmented sample; returns `True` on success. |

### `discovery.py`

```python
flight_dirs(split, data_root=DATA_ROOT)        # → train/val flight dirs (seeded)
collect_candidates(dirs)                        # → [(img_path, lbl_path), ...]
find_flight_paths(flight_root)                  # → (camera_dir, label_dir, sync_csv|None)
```

Used by `prepare_labels.py` and `benchmark.py`. The 80 / 20 train/val
split is frozen by `random.seed(42)` — never change without re-curating
the synthetic splits.

---

## `gate_detection.postprocessing` — [postprocessing/](../src/gate_detection/postprocessing)

### `quadgate.py`

```python
PRIOR_DESCRIPTORS  : dict[str, np.ndarray]    # canonical 4-bit per CORNER_TYPES

detect_corners(
    mask, priors,
    attitude_angle=0.0,
    max_dist=100.0, ransac_threshold=5.0, max_translation=150.0,
) -> np.ndarray | None                          # (8, 2) or None on failure

detect_corners_debug(...)                       # → dict with every intermediate
```

Args:
* `mask` — H×W float (sigmoid output) or uint8 binary; binarised at 0.5.
* `priors` — (8, 2) prior corner positions in `CORNER_TYPES` order in the
  *original* (un-derotated) frame.
* `attitude_angle` — deg to rotate the mask so world-up = image-up.
* `max_dist` — px gating between prior and matched candidate.
* `ransac_threshold` — RANSAC reprojection threshold (px).
* `max_translation` — reject the affine if `|tx|` or `|ty|` exceeds this.

The debug dict has keys: `mask_rot`, `M_rot`, `priors_rot`, `lines`,
`lines_ext`, `cand_pts`, `cand_descs`, `matches`, `M_affine`, `result`.

---

## `gate_detection.training` — [training/trainer.py](../src/gate_detection/training/trainer.py)

```python
build_dataloaders(syn_root=SYN_ROOT, *, img_size=384, batch_size=16,
                  use_aug=True, num_workers_train=12, num_workers_val=12,
                  prefetch_factor=4)
    -> (train_loader, val_loader, train_ds, val_ds)

train_epoch(model, loader, optimizer, device, *,
            scheduler=None, scaler=None,
            deep_supervision_weights=(4,2,1,1,1), bce_weight=2.0,
            iou_threshold=0.5,
            use_amp=False, amp_dtype=torch.float16,
            channels_last=False, desc="train")
    -> (mean_loss: float, mean_iou: float)

validate(model, loader, device, *, ...)         # same kwargs minus optimizer/scheduler
    -> (mean_loss: float, mean_iou: float)

train(*,
    # core
    epochs=100, batch_size=16, base_lr=1e-3, use_aug=True, img_size=384,

    # model
    model_name="gatenet", width_factor=4,
    encoder_name="resnet34", encoder_weights=None,
    dropout_p=0.0,

    # optimisation
    optimizer_name="adamw", sgd_momentum=0.9, weight_decay=1e-4,
    deep_supervision_weights=(4,2,1,1,1), bce_weight=2.0, iou_threshold=0.5,
    warmup_ratio=0.1, polynomial_power=1.0, polynomial_lr_end=1e-6,

    # performance
    compile_model=True, compile_mode="max-autotune",
    use_amp=True, amp_dtype="bf16", channels_last=True,
    cudnn_benchmark=True, matmul_precision="high",
    num_workers_train=4, num_workers_val=2, prefetch_factor=4,

    # paths
    syn_root=SYN_ROOT, checkpoint_dir=CHECKPOINTS_ROOT,
    checkpoint_name="best_gatenet.pth", checkpoint_interval=10,
    device=None,
) -> nn.Module
```

Selection metric: highest val IoU (tie-break: lower val loss). The best
state-dict is held CPU-side and written once at the end. Periodic
snapshots are written every `checkpoint_interval` epochs as
`<stem>_epochNNN.pth`.

---

## `gate_detection.inference` — [inference/predictor.py](../src/gate_detection/inference/predictor.py)

```python
load_checkpoint(path, *,
    model_name="gatenet", width_factor=4,
    encoder_name="resnet34", encoder_weights=None, dropout_p=0.0,
    device=None,
) -> (SegmentationModelWrapper, ckpt: dict)

class GatePredictor:
    def __init__(self, checkpoint, *, img_size=384,
                 model_name="gatenet", width_factor=4,
                 encoder_name="resnet34", encoder_weights=None,
                 dropout_p=0.0, device=None): ...
    def predict_image(self, img_path) -> (img_rgb: ndarray, prob: ndarray): ...
    # Internals: _preprocess_bgr, _predict_tensor (last entry of the tuple).
```

The CLI args are **fallbacks** — when the checkpoint contains a
`model_config` dict, those fields take precedence. `_normalize_state_dict_keys`
strips the `_orig_mod.` prefix added by `torch.compile`, and
`_load_state_dict_flexible` handles both wrapped (`model.*`) and legacy
raw-backbone state-dicts.

---

## `gate_detection.evaluation` — [evaluation/](../src/gate_detection/evaluation)

### `metrics.py`

```python
compute_iou(pred_bin, gt_bin) -> float      # via torchmetrics binary_jaccard_index
inner_corner_error(detected, gt_gates) -> float | None
    # mean L2 px error of inner corners (CORNER_TYPES indices 4..7)
    # vs the closest matching GT gate; None on detection failure
```

### `iou.py`

```python
@dataclass class IoUSample : img_rgb, gt_mask, pred_prob, pred_bin, iou, fname
@dataclass class IoUResults: ious, samples, visuals
                              .mean .median .std .fraction_above(t)

def evaluate_iou(predictor, *, test_dir=None, threshold=0.5, n_visuals=16)
    -> IoUResults
```

Iterates every test image, builds the GT mask via `corners_to_mask`,
runs the predictor, binarises at `threshold`, and stores up to
`n_visuals` per-sample visuals for the plotting helpers.

### `benchmark.py`

```python
@dataclass class FlightSample   : img_rgb, pred_prob, detected, gt_gates, cerr, fname
@dataclass class FlightBenchmark: flight_name, detected_mask, corner_errors,
                                   distances, visuals, fps, n_total

def benchmark_flight(predictor, flight_root, *, stride=30, n_visuals=16)
    -> FlightBenchmark
```

Walks the flight's `camera_*/` dir at `stride`, runs both stages
(GateNet + QuAdGate), and — if `<flight>_cam_ts_sync.csv` exists — reads
drone↔gate distances for the distance-binned report.

---

## `gate_detection.visualization` — [visualization/](../src/gate_detection/visualization)

### `colors.py`

```python
CORNER_COLORS: dict[str, str]    # warm for outer, cool for inner; keyed by CORNER_TYPES
```

### `plotting.py`

All four functions are matplotlib-only and write a PNG to `out_path`
(creating parent dirs as needed).

```python
plot_inference_grid(rows, ckpt_meta, out_path)
    # 5 cols: image · GT mask · pred (overlay) · pred (B&W) · QuAdGate result
plot_inference_grid_with_qag_steps(rows, ckpt_meta, out_path)
    # 7 cols: + per-step QuAdGate breakdown (lines / candidates / matches / corners)
plot_iou_grid(visuals, ckpt_meta, mean_iou, median_iou, threshold, out_path)
    # 4 cols: image · GT mask · pred prob · pred binary
plot_flight_grid(visuals, flight_name, fps, n_total, n_detected, out_path)
    # 3 cols: image+corner overlay · pred mask · text status
```

`rows` are dicts with `img_rgb, gt_mask, pred_prob, priors, fname`;
`visuals` are the `IoUSample` / `FlightSample` dataclasses produced by
the evaluators. Uses the `Agg` backend so the module is import-safe in
headless contexts (CI, SSH).
