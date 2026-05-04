# DroneRacing_GateDetection

Two-stage gate detection for autonomous drone racing:

1. **Segmentation backbone** — produces a binary gate-region mask from a 384×384 RGB frame, trained with deep-supervised Dice + BCE loss. Two interchangeable backbones: `gatenet` (custom narrow U-Net) and `deeplabv3plus` (via `segmentation_models_pytorch`). **Default: `deeplabv3plus`.**
2. **QuAdGate** — classical post-processor that extracts 8 sub-pixel gate corners (4 outer + 4 inner) from the mask via LSD line detection, descriptor-based prior matching, and a 4-DOF RANSAC affine fit.

Reference: [MonoRace](https://arxiv.org/abs/2601.15222). Full docs in [`docs/`](docs/) (architecture, data, usage, reference).

Training data: [tii-racing drone racing dataset](https://github.com/tii-racing/drone-racing-dataset).

---

## Status

### Done

- **Segmentation pipeline**: DeepLabV3+ (resnet34) and GateNet (custom U-Net) backbones, deep-supervised Dice + BCE loss, trained on tii-racing dataset
- **Training infrastructure**: AMP (bf16), `torch.compile` (max-autotune), channels-last, polynomial LR with warmup, val-IoU best-model selection, ETH HPC SLURM script
- **SAM-based mask generation**: `generate_sam_masks.py` / `build_sam_dataset.py` produce higher-quality segmentation targets; model re-trained on SAM-enhanced masks
- **QuAdGate post-processing**: LSD → descriptor matching → 4-DOF RANSAC → 8 sub-pixel corners (4 outer + 4 inner)
- **PnP pose estimation**: `benchmark_pnp.py` runs full PnP pipeline and logs reprojection error, translation/rotation error vs ground-truth corners and vs VICON ground truth
- **Evaluation scripts**: segmentation IoU (`evaluate.py`), flight-level corner error + detection rate (`benchmark_flight.py`), PnP benchmark (`benchmark_pnp.py`)
- **Inference & visualisation**: `inference.py` with per-step QuAdGate debug mode
- **Data preparation**: label synthesis (29-col with outer corners), interactive QA viewer, background download, offline augmentation (5× multiplier)

### Known Issues

#### PnP — reprojection error too high

Current results on `flight-01p-fixed` (360 frames, 82% gate detection rate):

| Metric | Value |
|--------|-------|
| Mean reprojection error (detected corners) | **5.0 px** |
| Max reprojection error | 41.2 px |
| Translation error vs GT label corners | 52.8 cm |
| Rotation error vs GT label corners | 7.4° |
| Translation error vs VICON | 4.4 m |
| Rotation error vs VICON | 45.6° |

The large VICON errors strongly suggest a coordinate frame mismatch between the PnP output frame and the VICON world frame — not just a corner accuracy problem. The ~5 px reprojection error itself also indicates the corner estimates or gate geometry model need refinement.

On real recorded data (137 frames): mean reprojection error **30.7 px**, max 96.1 px.

#### Segmentation — false positives on real data

The model was trained entirely on tii-racing data (piloted + synthetic flights). On real recorded footage the domain gap causes significant false positives — the model segments non-gate regions as gates. This degrades both the detection rate (39% on real vs 82% on tii-racing data) and the downstream PnP accuracy.

### Not Done

- **ROS2 integration**: no ROS2 node, publisher, or launch files yet — the pipeline runs offline only
- **Real-data fine-tuning**: no fine-tuning or domain adaptation on the real recorded dataset (real data is on Google Drive, not in the repo)
- **PnP coordinate frame alignment**: VICON ↔ camera world frame transform needs to be resolved
- **Online / real-time inference**: no latency benchmarking or inference optimisation for embedded hardware

---

## Layout

```
DroneRacing_GateDetection/
├── pyproject.toml             # editable install metadata
├── requirements.txt
├── configs/
│   └── default.yaml           # training / eval / postprocess hyperparameters
├── data/                      # gitignored — provide your own (see Setup)
├── checkpoints/               # gitignored — written here by training
├── results/                   # benchmark outputs (CSVs, visualisations)
│   ├── pnp_benchmark_01p_fixed.csv
│   ├── pnp_real.csv
│   └── visuals_01p_fixed/
├── src/gate_detection/
│   ├── data/                  # GateDataset, label I/O, transforms, synthesis
│   ├── models/                # gatenet, deeplabv3plus, SAM, factory, components
│   ├── losses/                # dice_loss, gate_loss (deep supervision)
│   ├── postprocessing/        # QuAdGate corner extractor
│   ├── training/              # trainer (build_dataloaders, train_epoch, validate, train)
│   ├── inference/             # GatePredictor wrapper
│   ├── evaluation/            # metrics, IoU evaluator, flight benchmark, PnP
│   ├── visualization/         # matplotlib grid plotters
│   ├── config.py              # YAML loader, device picker
│   └── paths.py               # repo-root-relative paths (env-overridable)
├── scripts/                   # thin CLI entry points
│   ├── train.py               # → training.train
│   ├── evaluate.py            # → evaluation.iou
│   ├── benchmark_flight.py    # → evaluation.benchmark
│   ├── benchmark_pnp.py       # → evaluation.pnp
│   ├── inference.py           # GatePredictor + visualization
│   ├── viewer.py              # interactive label keep / discard
│   ├── generate_sam_masks.py  # SAM mask generation
│   ├── build_sam_dataset.py   # assemble SAM training split
│   ├── prepare_labels.py      # synthesise outer corners (train/val/test)
│   ├── augment_data.py        # offline augmented copies
│   └── download_backgrounds.py
├── docs/                      # README, architecture, data, usage, reference
└── tests/
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                # editable install, pulls in requirements.txt
```

### Data and checkpoints

`data/` and `checkpoints/` are **gitignored** — you provide them. Two options:

```bash
# Option A: symlink to an existing dataset / checkpoint dir
ln -s /path/to/tii-racing-dataset data
ln -s /path/to/checkpoints       checkpoints

# Option B: point env vars at custom locations (no symlinks needed)
export DATA_ROOT=/mnt/datasets/drone_racing
export CHECKPOINTS_ROOT=/mnt/checkpoints
```

The expected `data/` layout (raw flight dirs + synthetic splits + backgrounds) is described in [`docs/data.md`](docs/data.md).

---

## Data preparation

Source data: [tii-racing drone racing dataset](https://github.com/tii-racing/drone-racing-dataset).

```bash
# Synthesise 29-col labels with outer corners — idempotent top-up.
python scripts/prepare_labels.py                    # train (2000) + val (500)
python scripts/prepare_labels.py --split test       # test (300)

# Optional: review and discard low-quality auto-labels.
python scripts/viewer.py --shuffle                  # SPACE/K=keep, D=discard

# Optional: download indoor backgrounds for offline augmentation.
python scripts/download_backgrounds.py --n 300

# Offline augmented copies (composited over backgrounds).
python scripts/augment_data.py --mult 5

# Generate SAM masks for higher-quality segmentation targets.
python scripts/generate_sam_masks.py
python scripts/build_sam_dataset.py
```

---

## Training

```bash
python scripts/train.py                             # uses configs/default.yaml
python scripts/train.py --epochs 50 --batch-size 8 --no-aug
python scripts/train.py --config configs/my_experiment.yaml
```

The default config trains DeepLabV3+ (resnet34 encoder); switch to GateNet by setting `model.name: gatenet` in your config. Best checkpoint (highest val IoU) is written to `checkpoints/best_gatenet.pth`; periodic snapshots land in the same directory.

For ETH HPC: `run_train.sh` is a ready-made SLURM submission script (4090 GPU, conda env `ethrc`).

---

## Evaluation

```bash
# Segmentation IoU on data/synthetic/test/
python scripts/evaluate.py checkpoints/best_gatenet.pth

# End-to-end pipeline benchmark on a single flight (corner-error + det-rate)
python scripts/benchmark_flight.py checkpoints/best_gatenet.pth \
    data/piloted/flight-01p-ellipse --stride 30

# PnP pose estimation benchmark
python scripts/benchmark_pnp.py checkpoints/best_gatenet.pth \
    data/piloted/flight-01p-ellipse
```

---

## Inference / visualisation

```bash
# Random samples from data/, default 5-column layout
python scripts/inference.py checkpoints/best_gatenet.pth --n 8

# Show every QuAdGate intermediate step
python scripts/inference.py checkpoints/best_gatenet.pth --qag-debug --n 4

# Specific images
python scripts/inference.py checkpoints/best_gatenet.pth \
    --images data/piloted/flight-01p-ellipse/camera_flight-01p-ellipse/00081_*.jpg
```

---

## Programmatic use

```python
from gate_detection.inference.predictor import GatePredictor
from gate_detection.postprocessing.quadgate import detect_corners
from gate_detection.data.labels import parse_pixel_corners

pred = GatePredictor("checkpoints/best_gatenet.pth")  # arch restored from ckpt
img_rgb, prob = pred.predict_image("frame.jpg")

priors  = parse_pixel_corners("frame.txt", img_size=384)[0]
corners = detect_corners(prob, priors)                # (8, 2) or None
```
