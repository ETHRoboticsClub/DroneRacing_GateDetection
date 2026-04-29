"""
Generate SAM2 ring segmentation masks for gate samples.

For each .jpg/.txt pair under <data-root>/{train,val,test}/:
  1. Parse the label -> list of gates with absolute-pixel bboxes + inner corners.
  2. SAM2 with each bbox as a prompt -> filled gate region.
  3. Ring = (SAM filled) AND NOT (inner-corner polygon from labels).
     Inner corners are hand-labeled and accurate, so we use the GT polygon
     for the hole instead of asking SAM to produce it.
  4. Save the binary ring as <stem>_mask.png, copying the image + label too.

Re-running is idempotent (existing _mask.png files are kept unless --force).

Default <data-root> is data/sam/ (the dataset built by build_sam_dataset.py).
Default <out-root>  is data/sam_masked/ (consumed by GateDataset when
use_sam_masks=true).

Usage:
    # smoke test on a single image (writes debug viz to data/sam_smoke/)
    python scripts/generate_sam_masks.py --smoke

    # full run on all splits
    python scripts/generate_sam_masks.py

    # specific splits / model / data root
    python scripts/generate_sam_masks.py --splits train val
    python scripts/generate_sam_masks.py --model facebook/sam2.1-hiera-base-plus
    python scripts/generate_sam_masks.py --data-root data/synthetic
"""

import argparse
import glob
import os
import shutil
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Sam2Model, Sam2Processor

from gate_detection.paths import DATA_ROOT

DEFAULT_DATA_ROOT = str(DATA_ROOT / "sam")
DEFAULT_OUT_ROOT  = str(DATA_ROOT / "sam_masked")
SMOKE_DIR         = str(DATA_ROOT / "sam_smoke")

DEFAULT_MODEL = "facebook/sam2.1-hiera-large"
MASK_SUFFIX   = "_mask.png"

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")


# ---------------------------------------------------------------------------
# Label parsing — bbox + inner corners only
# ---------------------------------------------------------------------------
def _parse_gates(lbl_path: str, W: int, H: int) -> list[dict]:
    gates = []
    with open(lbl_path) as f:
        for line in f:
            vals = list(map(float, line.split()))
            if len(vals) < 17:
                continue
            cx, cy, w, h = vals[1:5]
            xmin = max(0.0,      (cx - w / 2) * W)
            ymin = max(0.0,      (cy - h / 2) * H)
            xmax = min(float(W), (cx + w / 2) * W)
            ymax = min(float(H), (cy + h / 2) * H)

            inner = []
            for k in range(4):
                b = 5 + k * 3
                if vals[b + 2] > 0:
                    inner.append([vals[b] * W, vals[b + 1] * H])

            gates.append({
                "bbox":     [xmin, ymin, xmax, ymax],
                "inner_px": np.array(inner, dtype=np.float32) if len(inner) >= 3 else None,
            })
    return gates


# ---------------------------------------------------------------------------
# SAM2 inference
# ---------------------------------------------------------------------------
def run_sam2(model, processor, image_bgr: np.ndarray,
             boxes: list[list[float]]) -> np.ndarray:
    """Return a (H, W) uint8 mask (0/255) = union of best-IoU SAM masks per box."""
    H, W = image_bgr.shape[:2]
    if not boxes:
        return np.zeros((H, W), dtype=np.uint8)

    pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    input_boxes = [boxes]   # batch=1
    inputs = processor(images=pil, input_boxes=input_boxes,
                       return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    masks_per_image = processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"]
    )
    masks  = masks_per_image[0]            # (num_boxes, num_preds, H, W) bool
    scores = outputs.iou_scores[0].cpu()   # (num_boxes, num_preds)
    best   = scores.argmax(dim=-1).tolist()

    union = np.zeros((H, W), dtype=bool)
    for i, bi in enumerate(best):
        union |= masks[i, bi].numpy().astype(bool)
    return union.astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# Ring construction: subtract inner polygon from SAM filled mask
# ---------------------------------------------------------------------------
def _build_inner_mask(gates: list[dict], H: int, W: int) -> np.ndarray:
    inner = np.zeros((H, W), dtype=np.uint8)
    for g in gates:
        if g["inner_px"] is not None and len(g["inner_px"]) >= 3:
            cv2.fillPoly(inner, [g["inner_px"].astype(np.int32)], 255)
    return inner


def _build_ring(filled: np.ndarray, gates: list[dict]) -> np.ndarray:
    inner = _build_inner_mask(gates, *filled.shape)
    return cv2.bitwise_and(filled, cv2.bitwise_not(inner))


# ---------------------------------------------------------------------------
# Smoke test — single image, 6-panel debug viz
# ---------------------------------------------------------------------------
def _label_panel(title: str, img: np.ndarray) -> np.ndarray:
    label_h = 26
    out = np.zeros((img.shape[0] + label_h, img.shape[1], 3), dtype=np.uint8)
    out[label_h:] = img
    cv2.putText(out, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def smoke_test(model, processor, img_path: str, lbl_path: str):
    os.makedirs(SMOKE_DIR, exist_ok=True)
    img = cv2.imread(img_path)
    if img is None:
        sys.exit(f"smoke: could not read {img_path}")
    H, W = img.shape[:2]
    gates = _parse_gates(lbl_path, W, H)
    print(f"[smoke] {os.path.basename(img_path)}  ({W}x{H}, {len(gates)} gate(s))")

    boxes  = [g["bbox"] for g in gates]
    filled = run_sam2(model, processor, img, boxes)
    inner  = _build_inner_mask(gates, H, W)
    ring   = cv2.bitwise_and(filled, cv2.bitwise_not(inner))

    bbox_img = img.copy()
    for x0, y0, x1, y1 in boxes:
        cv2.rectangle(bbox_img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 2)

    overlay = img.copy()
    sel = ring > 0
    overlay[sel] = (0.45 * overlay[sel] + 0.55 * np.array([0, 255, 0])).astype(np.uint8)

    panels = [
        ("image",           img),
        ("bbox prompts",    bbox_img),
        ("SAM filled",      cv2.cvtColor(filled, cv2.COLOR_GRAY2BGR)),
        ("inner subtract",  cv2.cvtColor(inner,  cv2.COLOR_GRAY2BGR)),
        ("ring (final GT)", cv2.cvtColor(ring,   cv2.COLOR_GRAY2BGR)),
        ("overlay",         overlay),
    ]
    panels = [_label_panel(t, p) for t, p in panels]
    grid = np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])])

    debug_path = os.path.join(SMOKE_DIR, "smoke_debug.jpg")
    mask_path  = os.path.join(SMOKE_DIR, "smoke_mask.png")
    cv2.imwrite(debug_path, grid)
    cv2.imwrite(mask_path,  ring)

    af, ai, ar = int((filled > 0).sum()), int((inner > 0).sum()), int((ring > 0).sum())
    print(f"[smoke] filled={af}px  inner-hole={ai}px  ring={ar}px")
    print(f"[smoke] -> {debug_path}")
    print(f"[smoke] -> {mask_path}")


# ---------------------------------------------------------------------------
# Full pipeline over a split
# ---------------------------------------------------------------------------
def process_split(model, processor, data_root: str, out_root: str, split: str, force: bool):
    in_dir  = os.path.join(data_root, split)
    out_dir = os.path.join(out_root, split)
    if not os.path.isdir(in_dir):
        print(f"[{split}] skipped (no dir at {in_dir})")
        return
    os.makedirs(out_dir, exist_ok=True)

    stems = sorted(os.path.splitext(f)[0]
                   for f in os.listdir(in_dir) if f.endswith(".txt"))

    skipped = generated = failed = 0
    for stem in tqdm(stems, desc=f"[{split}]"):
        img_path  = os.path.join(in_dir,  stem + ".jpg")
        lbl_path  = os.path.join(in_dir,  stem + ".txt")
        mask_path = os.path.join(out_dir, stem + MASK_SUFFIX)

        if not os.path.exists(img_path):
            failed += 1
            continue
        if os.path.exists(mask_path) and not force:
            skipped += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            failed += 1
            continue
        H, W = img.shape[:2]
        gates = _parse_gates(lbl_path, W, H)
        if not gates:
            failed += 1
            continue

        try:
            filled = run_sam2(model, processor, img, [g["bbox"] for g in gates])
        except Exception as e:
            tqdm.write(f"  {stem}: SAM2 error {type(e).__name__}: {e}")
            failed += 1
            continue

        ring = _build_ring(filled, gates)
        cv2.imwrite(mask_path, ring)
        shutil.copy2(img_path, os.path.join(out_dir, stem + ".jpg"))
        shutil.copy2(lbl_path, os.path.join(out_dir, stem + ".txt"))
        generated += 1

    print(f"[{split}] generated={generated}  skipped(existing)={skipped}  "
          f"failed={failed}  total={len(stems)}")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_id: str):
    print(f"Loading {model_id} on {DEVICE} ...")
    processor = Sam2Processor.from_pretrained(model_id)
    model     = Sam2Model.from_pretrained(model_id).to(DEVICE)
    model.eval()
    return model, processor


# ---------------------------------------------------------------------------
# Smoke-image resolution helper
# ---------------------------------------------------------------------------
def resolve_smoke_image(data_root: str, arg: str | None) -> tuple[str, str]:
    if arg and os.path.exists(arg):
        img = arg
        lbl = os.path.splitext(arg)[0] + ".txt"
        return img, lbl

    if arg:
        for split in ("train", "val", "test"):
            cand = os.path.join(data_root, split, arg + ".jpg")
            if os.path.exists(cand):
                return cand, os.path.splitext(cand)[0] + ".txt"
        sys.exit(f"smoke image '{arg}' not found in any split")

    for split in ("train", "val", "test"):
        cands = sorted(glob.glob(os.path.join(data_root, split, "*.jpg")))
        if cands:
            return cands[0], os.path.splitext(cands[0])[0] + ".txt"
    sys.exit(f"no images found in any split under {data_root}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Generate SAM2 ring masks for gate samples.")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                   help=f"root with train/val/test subdirs (default: {DEFAULT_DATA_ROOT})")
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT,
                   help=f"where to write masked outputs (default: {DEFAULT_OUT_ROOT})")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--model",  default=DEFAULT_MODEL)
    p.add_argument("--force",  action="store_true", help="overwrite existing masks")
    p.add_argument("--smoke",  action="store_true",
                   help="run smoke test on a single image and exit")
    p.add_argument("--smoke-image", default=None,
                   help="stem or full path for --smoke")
    args = p.parse_args()

    model, processor = load_model(args.model)

    if args.smoke:
        img_path, lbl_path = resolve_smoke_image(args.data_root, args.smoke_image)
        smoke_test(model, processor, img_path, lbl_path)
        return

    for split in args.splits:
        process_split(model, processor, args.data_root, args.out_root, split, args.force)


if __name__ == "__main__":
    main()
