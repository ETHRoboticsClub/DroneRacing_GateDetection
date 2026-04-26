# Data

## Source dataset

[tii-racing/drone-racing-dataset](https://github.com/tii-racing/drone-racing-dataset)
— ~20 piloted and autonomous flights, each providing onboard camera frames
plus YOLO-keypoint labels for the gate **inner** corners only.

We synthesise outer corners ourselves (radial scaling from the inner
centroid) so that the segmentation model can be trained to predict the
**ring** between the inner and outer rectangles — a much more informative
target for QuAdGate's descriptor matching than just the inner quadrilateral.

## On-disk layout

```
data/
├── raw/
│   ├── autonomous/
│   │   └── flight-XX-name/
│   │       ├── camera_flight-XX-name/  *.jpg
│   │       └── label_flight-XX-name/   *.txt          (17-col, inner only)
│   └── backgrounds/                    *.jpg          (downloaded indoor scenes)
├── piloted/
│   └── flight-XX-name/                 (same structure as autonomous)
└── synthetic/
    ├── train/         *.jpg + *.txt    (curated 29-col labels)
    ├── val/           *.jpg + *.txt
    ├── test/          *.jpg + *.txt
    ├── train_aug/     *.jpg + *.txt    (offline-augmented copies)
    ├── val_aug/       *.jpg + *.txt
    └── .reviewed                       (log written by scripts/viewer.py)
```

In this repo the top-level `data/` is a symlink into the legacy ETHRC
gate-detection tree; override via `export DATA_ROOT=…` if you keep the
dataset elsewhere.

## Label format

Each `.txt` is a YOLO-keypoint label, one line per gate.

### 17-col (original tii-racing format)

```
cls  cx cy w h   ix1 iy1 vis1   ix2 iy2 vis2   ix3 iy3 vis3   ix4 iy4 vis4
```

* `cls` — class index (always 0 for gates).
* `cx cy w h` — bbox centre + size, normalised to `[0, 1]`.
* `ixK iyK visK` — k-th **inner** corner: normalised xy + visibility flag
  (`0` not visible, `2` visible).

### 29-col (after outer-corner synthesis)

```
cls cx cy w h
    ix1 iy1 vis1   ix2 iy2 vis2   ix3 iy3 vis3   ix4 iy4 vis4     ← inner
    ox1 oy1 vos1   ox2 oy2 vos2   ox3 oy3 vos3   ox4 oy4 vos4     ← outer
```

Outer corners are produced by
`synthesize_outer(inner, scale=1.4) = centroid + scale·(inner − centroid)`.
An outer corner inherits visibility `2` only when (a) its matching inner
corner is visible **and** (b) the synthesised xy lies inside `[0, 1]`;
otherwise the outer entry is `0.0 0.0 0`.

## Coordinate conventions

* All label coordinates are **normalised** to `[0, 1]`.
* Image coordinates everywhere else are pixels, with `+x` right and `+y`
  down.
* `parse_pixel_corners(lbl_path, img_size)` is the bridge: it returns
  `(8, 2)` arrays in **pixel** coordinates, in `CORNER_TYPES` order.

## Splits

The split between train and val flights is **deterministic**, frozen by
`random.seed(42)` in `data.discovery.flight_dirs`:

* All `flight-*` directories under `raw/autonomous/` and `piloted/` are
  sorted alphabetically, then shuffled with seed 42.
* The first 80 % go to **train**, the remaining 20 % go to **val**.
* The **test** set is sourced exclusively from the val-split flights —
  no train flight ever contributes to test, and no test stem may overlap
  with the curated `val/` directory.

Default per-split sample quotas (`scripts/prepare_labels.py`):

| Split | Default `--n` |
| ----- | ------------- |
| train | 2000          |
| val   | 500           |
| test  | 300           |

This is enforced by `scripts/prepare_labels.py`. Do **not** change the seed
or sort order without re-curating the synthetic splits.

## Data preparation pipeline

```
raw flight dirs
       │
       │   prepare_labels.py
       │   (filter all4_visible · synthesise outer · copy + write)
       ▼
data/synthetic/{train,val,test}/                                (auto-curated)
       │
       │   viewer.py                            (manual review: keep / discard)
       ▼
data/synthetic/{train,val,test}/                                (curated)
       │
       │   augment_data.py                      (offline 5× multiplication)
       ▼
data/synthetic/{train,val}_aug/                                 (composited)
       ▲
       │   download_backgrounds.py              (HuggingFace indoor scenes)
       │
data/raw/backgrounds/
```

End state used by training: `GateDataset` reads from
`{train, train_aug}` (or just `{train}` with `--no-aug`), and similarly
for val.

## Augmentations

The codebase distinguishes **online** and **offline** augmentation:

### Online (`data/transforms.py`)

Applied per-batch inside `GateDataset.__getitem__` when `augment=True`.
Cheap, synchronous, runs every epoch:

| Transform        | Default range                            |
| ---------------- | ---------------------------------------- |
| Horizontal flip  | p = 0.5                                  |
| Affine           | rot ±15°, trans ±10 %, scale 0.9–1.1     |
| HSV jitter       | hue ±10, sat ×0.8–1.2, val ×0.8–1.2      |
| Motion blur      | kernel 5–15 px, random angle             |
| Additive noise   | σ ~ U[0, 15]                             |

Default chain: `flip → affine → HSV → motion blur → Gaussian noise`. Each
function is independently importable for ablations.

### Offline (`data/synthesis.py`)

Run once via `scripts/augment_data.py`; produces fixed augmented copies on
disk in `<split>_aug/`. Each copy:

1. Loads the curated frame and a random indoor background.
2. Warps the frame by a random rigid + perspective transform
   (`random_affine_matrix · random_perspective_matrix`).
3. Composites the warped gate region onto the background — the gate mask
   is the union of every gate's outer polygon, dilated 7 px to hide
   compositing fringes.
4. Independently HSV-jitters gate and background.
5. Adds Gaussian noise (σ ~ U[0, 12]) to the composite.
6. Updates every label corner through the same warp; drops gates whose
   inner corners all leave the image.

Offline augmentation is the **dominant** source of variety — it places
gates on backgrounds the network would never see in raw flight data, which
is what prevents over-fitting to the lab environment.

## Background pool

`scripts/download_backgrounds.py` streams the
[Andron00e/Places365-custom](https://huggingface.co/datasets/Andron00e/Places365-custom)
HuggingFace dataset and saves resized 640×480 JPEGs to
`data/raw/backgrounds/`. The pool is loaded once per process (lazy global
in `synthesis._get_bg_pool`).

## Programmatic label I/O

```python
from gate_detection.data.labels import (
    parse_gates,                 # → list of dicts {cls, cx,cy,w,h,
                                 #                  inner, inner_vis,
                                 #                  outer, outer_vis}
    corners_to_mask,             # → (H,W) uint8 binary mask
                                 #   ring (29-col) or filled inner (17-col)
    parse_pixel_corners,         # → list of (8,2) arrays in CORNER_TYPES order
    synthesize_outer,            # outer = centroid + scale·(inner − centroid)
    make_synthetic_label_lines,  # 17-col path → 29-col line strings
    gates_to_lines,              # parsed gates → label-file line strings
    assign_corner_types,         # unordered (4,2)+(4,2) → (8,2) sorted
    all4_visible,                # all 4 inner corners visible in every gate
    CORNER_TYPES,
)

from gate_detection.data.discovery import (
    flight_dirs,                 # → train / val flight directories (seeded)
    collect_candidates,          # → (img, lbl) pairs with all4_visible
    find_flight_paths,           # → (camera_dir, label_dir, sync_csv)
)
```
