"""On-the-fly training augmentations for image+mask pairs.

Each augmenter operates on a paired ``(img, mask)`` tuple in (H, W, C) and
(H, W) layout respectively, using OpenCV. Designed for synchronous calls
inside ``GateDataset.__getitem__``.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

# ImageNet-1k preprocessing constants for ResNet-style encoders. Matches
# segmentation_models_pytorch's resnet34 + imagenet pretrained_settings (see
# segmentation_models_pytorch/encoders/_legacy_pretrained_settings.py): RGB,
# inputs in [0, 1], mean / std subtracted per-channel. Apply via
# apply_imagenet_norm(); call at both training and inference for parity.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def apply_imagenet_norm(img_rgb_uint8: np.ndarray) -> np.ndarray:
    """Per-channel ImageNet mean/std normalisation.

    Input: ``(H, W, 3)`` RGB image, **uint8 in [0, 255]** (no auto-detect —
    callers must pass uint8 to keep this single-purpose). Output: ``(H, W, 3)``
    float32, scaled to ``[0, 1]`` and then mean/std-subtracted per channel.
    """
    if img_rgb_uint8.dtype != np.uint8:
        raise TypeError(
            f"apply_imagenet_norm expects uint8 input, got {img_rgb_uint8.dtype}"
        )
    x = img_rgb_uint8.astype(np.float32) / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def random_flip(img: np.ndarray, mask: np.ndarray, p: float = 0.5):
    """Random horizontal flip applied to image and mask together."""
    if random.random() < p:
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)
    return img, mask


def random_affine(
    img: np.ndarray,
    mask: np.ndarray,
    *,
    angle_deg: float = 15.0,
    translate_frac: float = 0.10,
    scale_range: tuple[float, float] = (0.9, 1.1),
):
    """Random rotation ± angle_deg, translation ± translate_frac of size, scale."""
    H, W = img.shape[:2]
    angle = random.uniform(-angle_deg, angle_deg)
    tx = random.uniform(-translate_frac, translate_frac) * W
    ty = random.uniform(-translate_frac, translate_frac) * H
    scale = random.uniform(*scale_range)

    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_REFLECT)
    return img, mask


def hsv_jitter(
    img: np.ndarray,
    *,
    h_range: float = 10.0,
    s_factor: tuple[float, float] = (0.8, 1.2),
    v_factor: tuple[float, float] = (0.8, 1.2),
) -> np.ndarray:
    """RGB-input HSV colour jitter (hue shift, saturation & value scaling)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + random.uniform(-h_range, h_range)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(*s_factor), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(*v_factor), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def motion_blur(img: np.ndarray, k_choices=range(5, 16, 2)) -> np.ndarray:
    """Random directional motion blur with odd kernel size in ``k_choices``.

    Build a horizontal line kernel through the centre, then rotate it by a
    random angle. Simulates camera motion during exposure — important
    because racing drones produce a lot of motion blur and a clean-trained
    model would not generalise.
    """
    k = random.choice(list(k_choices))
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    angle = random.uniform(0, 180)
    rot = cv2.getRotationMatrix2D((k // 2, k // 2), angle, 1)
    kernel = cv2.warpAffine(kernel, rot, (k, k))
    # Re-normalise after rotation: the rotated kernel may sum to slightly
    # less than 1 due to bilinear resampling losses at the corners.
    kernel /= kernel.sum() + 1e-6
    return cv2.filter2D(img, -1, kernel)


def gaussian_noise(img: np.ndarray, sigma_max: float = 15.0) -> np.ndarray:
    """Additive Gaussian noise with σ ~ U[0, sigma_max] (per-image)."""
    sigma = random.uniform(0, sigma_max)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def train_augment(img: np.ndarray, mask: np.ndarray):
    """Default training augmentation chain (flip → affine → HSV → blur → noise)."""
    img, mask = random_flip(img, mask)
    img, mask = random_affine(img, mask)
    img = hsv_jitter(img)
    img = motion_blur(img)
    img = gaussian_noise(img)
    return img, mask
