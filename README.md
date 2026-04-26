# DroneRacing_GateDetection

Two-stage gate detection for autonomous drone racing:

1. **Segmentation backbone** — produces a binary gate-region mask from a 384×384
   RGB frame, trained with deep-supervised Dice + BCE loss. Two interchangeable
   backbones: `gatenet` (custom narrow U-Net) and `deeplabv3plus` (via
   `segmentation_models_pytorch`). **Default: `deeplabv3plus`.**
2. **QuAdGate** — classical post-processor that extracts 8 sub-pixel gate
   corners (4 outer + 4 inner) from the mask via LSD line detection,
   descriptor-based prior matching, and a 4-DOF RANSAC affine fit.

Reference: [MonoRace](https://arxiv.org/abs/2601.15222). Full docs in
[`docs/`](docs/) (architecture, data, usage, reference).

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
├── src/gate_detection/
│   ├── data/                  # GateDataset, label I/O, transforms, synthesis
│   ├── models/                # gatenet, deeplabv3plus, factory, components
│   ├── losses/                # dice_loss, gate_loss (deep supervision)
│   ├── postprocessing/        # QuAdGate corner extractor
│   ├── training/              # trainer (build_dataloaders, train_epoch, validate, train)
│   ├── inference/             # GatePredictor wrapper
│   ├── evaluation/            # metrics, IoU evaluator, flight benchmark
│   ├── visualization/         # matplotlib grid plotters
│   ├── config.py              # YAML loader, device picker
│   └── paths.py               # repo-root-relative paths (env-overridable)
├── scripts/                   # thin CLI entry points
│   ├── train.py               # → training.train
│   ├── evaluate.py            # → evaluation.iou
│   ├── benchmark_flight.py    # → evaluation.benchmark
│   ├── inference.py           # GatePredictor + visualization
│   ├── viewer.py              # interactive label keep / discard
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

The expected `data/` layout (raw flight dirs + synthetic splits + backgrounds)
is described in [`docs/data.md`](docs/data.md).

---

## Data preparation

Source data is the [tii-racing dataset](https://github.com/tii-racing/drone-racing-dataset).

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
```

---

## Training

```bash
python scripts/train.py                             # uses configs/default.yaml
python scripts/train.py --epochs 50 --batch-size 8 --no-aug
python scripts/train.py --config configs/my_experiment.yaml
```

The default config trains DeepLabV3+ (resnet34 encoder); switch to GateNet by
setting `model.name: gatenet` in your config. Best checkpoint (highest val IoU)
is written to `checkpoints/best_gatenet.pth`; periodic snapshots land in the
same directory.

---

## Evaluation

```bash
# Segmentation IoU on data/synthetic/test/
python scripts/evaluate.py checkpoints/best_gatenet.pth

# End-to-end pipeline benchmark on a single flight (corner-error + det-rate)
python scripts/benchmark_flight.py checkpoints/best_gatenet.pth \
    data/piloted/flight-01p-ellipse --stride 30
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
