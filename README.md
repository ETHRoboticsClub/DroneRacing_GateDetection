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

### Deployment-relevant headline (current — revised report 12)

> **Headline checkpoint revised** as of [`docs/report/12_cross_flight_held_out/findings.md`](docs/report/12_cross_flight_held_out/findings.md): the previous report-07 ckpt (`best_deeplabv3plus_keypoint.pth`, single-gate target) memorized `flight-01p-ellipse` and **fails to detect anything on the only held-out piloted flight** (`flight-06p-ellipse`, 0 SAM training crops). The Stage 2 sibling (`best_deeplabv3plus_keypoint_multi.pth`, multi-gate Gaussian targets, see [`docs/report/10_multi_gate_target/findings.md`](docs/report/10_multi_gate_target/findings.md)) generalizes and is the new deployment checkpoint.

On `flight-06p-ellipse` (TII v3.0.0 piloted, **held-out — 0 crops in SAM training**, 374 frames sampled stride 30, no GT prior, no EKF state):

| Mode | Det rate | Mean cerr | Median cerr | ≤ 5 px | Throughput | Notes |
|---|---|---|---|---|---|---|
| `model_direct` (head only, **single forward pass**, `coherent_quad_decode`) | **78.9 %** | 12.2 px | **1.30 px** | **61.0 %** | **0.44 ms / 2274 fps** TRT FP16 on RTX 5090 | Deployment headline. |
| `gt` (perfect prior — refinement benchmark only) | 57.0 % | 2.17 px | 0.74 px | **95.8 %** | — | Confirms the seg head generalizes — perfect-prior gives near-perfect refinement on the held-out flight. |

False-positive rate on no-gate frames of the same held-out flight (upper bound — labellers skip non-trivial frames so some "fires" may be detecting unlabelled gates): **2.4 % (4/166 at threshold 0.3)**.

Cross-flight detection rate on the four in-training flights tested ranges 80.5–88.7 %, within-5 px ranges 34.7–66.2 % — see [report 12](docs/report/12_cross_flight_held_out/findings.md) for the per-flight table.

The keypoint head is in `configs/deeplabv3plus_keypoint_multi.yaml`; the checkpoint round-trips its own metadata so the predictor auto-detects the multi-gate decode path. The single-gate `*_keypoint.pth` checkpoint is preserved as an experimental sibling — see report 07 — but is not the deployment headline.

### Known issues — historical numbers

> **Caveat — read before quoting any historical number below.** The 82 % / 5.0 px / 30.7 px / "0 % under realistic prior" headlines all measured QuAdGate-only with various artificial priors (GT, GT + jitter, centred default). They are not deployment numbers. See [`docs/report/06_realistic_priors/findings.md`](docs/report/06_realistic_priors/findings.md) for the full `gt` / `noisy` / `none` prior-mode ablation that exposed the prior-sensitivity, and [report 12](docs/report/12_cross_flight_held_out/findings.md) (revising report 07's headline upward to the multi-gate ckpt) for the deployable 61 % within-5 px result on the held-out flight.
>
> The original 82 % was measured on `flight-01p-fixed` with 29-col labels produced by `scripts/prepare_labels.py` — a script absent from the current checkout (tracked as plan P0-4) — and the flight itself is on Google Drive, not yet on qolam. On `flight-01p-ellipse` with the 17-col labels TII ships and `parse_pixel_corners(..., synth_outer=True)`, the equivalent `gt`-prior numbers are **67.4 % detection** with **2.78 px mean corner error** at **93.7 %** within 5 px. The 39 % real-footage detection rate further down is a separate domain-gap issue and is unaffected by the prior caveat — that one stands as a deliverable until we get the team-recorded flights on qolam (P1-2).

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
