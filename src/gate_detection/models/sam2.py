"""SAM2 model wrapper used to generate ring segmentation masks from gate labels.

Kept separate from the GateNet / DeepLabV3+ models because SAM2 pulls in
``transformers`` (a heavy optional dependency only needed for the
mask-generation pipeline). For that reason it is **not** re-exported from
:mod:`gate_detection.models.__init__` — import it explicitly:

    from gate_detection.models.sam2 import load_sam2, run_sam2
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

DEFAULT_MODEL = "facebook/sam2.1-hiera-large"

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")


def load_sam2(model_id: str = DEFAULT_MODEL, device: str = DEVICE):
    """Load a SAM2 model + processor pair from the HuggingFace hub."""
    from transformers import Sam2Model, Sam2Processor

    print(f"Loading {model_id} on {device} ...")
    processor = Sam2Processor.from_pretrained(model_id)
    model     = Sam2Model.from_pretrained(model_id).to(device)
    model.eval()
    return model, processor


def run_sam2(model, processor, image_bgr: np.ndarray,
             boxes: list[list[float]]) -> np.ndarray:
    """Return a ``(H, W)`` uint8 mask (0/255) = union of best-IoU SAM masks per box."""
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
