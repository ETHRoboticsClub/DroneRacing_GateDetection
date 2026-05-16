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


# ---------------------------------------------------------------------------
# Strong-aug primitives (report 14 — generalization preparation)
# ---------------------------------------------------------------------------
# These augmenters are gated by the ``augment_strong`` config flag (=False
# by default, preserving the v1 baseline). They address documented OOD
# failure modes — viewpoint shift (perspective), unknown venue lighting
# (brightness / contrast / gamma), partial occlusion (cutout — DeVries
# & Taylor 2017, arXiv:1708.04552; we implement plain cutout with black
# fill, NOT the learned/random fill of Zhong et al. 2017 random erasing),
# and compressed footage (JPEG). Each was picked over CLAHE / weather
# overlays because the cost is one cv2 call and the semantics are
# well-cited.

def _random_perspective_matrix(H: int, W: int, max_disp_frac: float):
    """Build a 3×3 perspective transform that displaces each corner by up
    to ``max_disp_frac`` of the corresponding image dimension."""
    max_dx = max_disp_frac * W
    max_dy = max_disp_frac * H
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    offsets = np.float32(
        [[random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy)]
         for _ in range(4)]
    )
    dst = src + offsets
    return cv2.getPerspectiveTransform(src, dst)


def random_perspective(
    img: np.ndarray, mask: np.ndarray, *, max_disp_frac: float = 0.08,
):
    """Random perspective warp on img+mask (corners shift independently by
    ±max_disp_frac of image size). ``BORDER_CONSTANT(0)`` — same rationale
    as ``random_affine_kp``: a reflected edge gate would teach the model
    to ignore real edge gates at deployment."""
    H, W = img.shape[:2]
    M = _random_perspective_matrix(H, W, max_disp_frac)
    img = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(mask, M, (W, H), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img, mask


def random_perspective_kp(
    img: np.ndarray, mask: np.ndarray, corners: np.ndarray, vis: np.ndarray,
    *, max_disp_frac: float = 0.08,
):
    """Single-gate perspective warp on img + mask + (K, 2) corners."""
    H, W = img.shape[:2]
    M = _random_perspective_matrix(H, W, max_disp_frac)
    img = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(mask, M, (W, H), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # [x, y, 1] @ M.T → [u, v, w]; divide u/w, v/w
    corners_h = np.hstack([corners, np.ones((corners.shape[0], 1), dtype=corners.dtype)])
    corners_w = corners_h @ M.T
    corners_t = corners_w[:, :2] / np.clip(corners_w[:, 2:3], 1e-6, None)
    in_frame = (
        (corners_t[:, 0] >= 0) & (corners_t[:, 0] < W) &
        (corners_t[:, 1] >= 0) & (corners_t[:, 1] < H)
    )
    vis = (vis.astype(np.int32) * in_frame.astype(np.int32)).astype(vis.dtype)
    return img, mask, corners_t.astype(corners.dtype), vis


def random_perspective_kp_multi(
    img: np.ndarray, mask: np.ndarray, corners: np.ndarray, vis: np.ndarray,
    *, max_disp_frac: float = 0.08,
):
    """Multi-gate perspective warp on img + mask + (G, K, 2) corners.

    Same H matrix applied to every gate's corners — they share the image.
    Per-corner in-frame visibility is recomputed.
    """
    H, W = img.shape[:2]
    M = _random_perspective_matrix(H, W, max_disp_frac)
    img = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(mask, M, (W, H), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    G, K = corners.shape[0], corners.shape[1]
    if G == 0:
        return img, mask, corners, vis
    flat = corners.reshape(-1, 2)
    flat_h = np.hstack([flat, np.ones((flat.shape[0], 1), dtype=flat.dtype)])
    flat_w = flat_h @ M.T
    flat_t = (flat_w[:, :2] / np.clip(flat_w[:, 2:3], 1e-6, None)).astype(corners.dtype)
    corners_t = flat_t.reshape(G, K, 2)
    in_frame = (
        (corners_t[..., 0] >= 0) & (corners_t[..., 0] < W) &
        (corners_t[..., 1] >= 0) & (corners_t[..., 1] < H)
    )
    vis = (vis.astype(np.int32) * in_frame.astype(np.int32)).astype(vis.dtype)
    return img, mask, corners_t, vis


def brightness_contrast_gamma(
    img: np.ndarray,
    *,
    b_range: float = 0.2,
    c_range: float = 0.3,
    g_range: tuple[float, float] = (0.7, 1.4),
) -> np.ndarray:
    """Photometric jitter — additive brightness shift, multiplicative contrast
    around the per-image mean, and a gamma curve.

    Standard photometric chain (matches albumentations RandomBrightnessContrast
    and RandomGamma defaults, scaled to the venue-lighting uncertainty we
    actually expect at competition).
    """
    out = img.astype(np.float32)
    out = out + random.uniform(-b_range, b_range) * 255.0
    mean = out.mean()
    out = (out - mean) * random.uniform(1.0 - c_range, 1.0 + c_range) + mean
    out = np.clip(out, 0, 255)
    gamma = random.uniform(*g_range)
    out = (out / 255.0) ** (1.0 / gamma) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def random_cutout(
    img: np.ndarray, mask: np.ndarray,
    *,
    n_range: tuple[int, int] = (1, 3),
    size_frac: tuple[float, float] = (0.05, 0.20),
):
    """Cut out N∈n_range black patches from the image; mask stays unchanged.

    Implements cutout (DeVries & Taylor 2017, arXiv:1708.04552). The mask
    is **not** punctured because the gate is still there in world — the
    pixels are just occluded; we want the model to learn to recover the
    full label from partial visibility. For the same reason, keypoint
    visibility flags are not changed (occlusion ≠ corner-leaves-frame).
    """
    H, W = img.shape[:2]
    n_patches = random.randint(*n_range)
    out = img.copy()
    for _ in range(n_patches):
        ph = max(1, int(random.uniform(*size_frac) * H))
        pw = max(1, int(random.uniform(*size_frac) * W))
        y = random.randint(0, max(0, H - ph))
        x = random.randint(0, max(0, W - pw))
        out[y:y + ph, x:x + pw] = 0
    return out, mask


def jpeg_compression(img: np.ndarray, *, q_range: tuple[int, int] = (60, 95)) -> np.ndarray:
    """Simulate JPEG compression artifacts at random quality q∈q_range.

    Input is RGB uint8; we convert RGB→BGR for cv2.imencode, then back.
    On encode failure (vanishingly rare), return the input verbatim — and
    log it per CLAUDE.md §6 so the silent path is visible.
    """
    q = random.randint(*q_range)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        print(
            f"[WARN] jpeg_compression: cv2.imencode failed (q={q}); "
            f"returning original image unchanged.",
            flush=True,
        )
        return img
    dec_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if dec_bgr is None:
        print(
            f"[WARN] jpeg_compression: cv2.imdecode returned None (q={q}); "
            f"returning original image unchanged.",
            flush=True,
        )
        return img
    return cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB)


def train_augment(img: np.ndarray, mask: np.ndarray, *, strong: bool = False):
    """Default training augmentation chain (flip → affine → HSV → blur → noise).

    When ``strong=True``, additionally applies perspective warp (after flip,
    before affine), stronger HSV (h_range 20 vs 10), brightness/contrast/gamma
    (after HSV), JPEG compression (after noise), and cutout (last). Strong
    chain is gated by the ``data.augment_strong`` config flag.
    """
    img, mask = random_flip(img, mask)
    if strong:
        img, mask = random_perspective(img, mask)
    img, mask = random_affine(img, mask)
    img = hsv_jitter(img, h_range=20.0 if strong else 10.0)
    if strong:
        img = brightness_contrast_gamma(img)
    img = motion_blur(img)
    img = gaussian_noise(img)
    if strong:
        img = jpeg_compression(img)
        img, mask = random_cutout(img, mask)
    return img, mask


# ---------------------------------------------------------------------------
# Keypoint-aware augmentation
# ---------------------------------------------------------------------------
# The keypoint-multi-task path (plan P1-1) needs the SAME spatial transforms
# applied to the inner corner pixel coordinates so the rendered heatmap
# targets line up with the augmented image. We deliberately track corners
# vectorially (not by warping a pre-rendered heatmap), because warping a
# Gaussian peak through bilinear interpolation smears it sub-pixel — fine
# for visualisation, bad for sub-pixel keypoint regression.
#
# Inner-corner channel order is the second half of CORNER_TYPES at
# data/labels.py:33-36 — [inner_TL, inner_TR, inner_BR, inner_BL].
_INNER_CORNER_HFLIP_PERM = [1, 0, 3, 2]


def random_flip_kp(
    img: np.ndarray,
    mask: np.ndarray,
    corners: np.ndarray,
    vis: np.ndarray,
    p: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random horizontal flip applied to img + mask + corner coords.

    Horizontal flip swaps left and right corners (TL↔TR, BL↔BR), so the
    channel order must be permuted to keep "channel k = inner corner k"
    consistent across batch.
    """
    if random.random() < p:
        H, W = img.shape[:2]
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)
        corners = corners.copy()
        corners[:, 0] = (W - 1) - corners[:, 0]
        corners = corners[_INNER_CORNER_HFLIP_PERM]
        vis = vis[_INNER_CORNER_HFLIP_PERM]
    return img, mask, corners, vis


def random_affine_kp(
    img: np.ndarray,
    mask: np.ndarray,
    corners: np.ndarray,
    vis: np.ndarray,
    *,
    angle_deg: float = 15.0,
    translate_frac: float = 0.10,
    scale_range: tuple[float, float] = (0.9, 1.1),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Affine warp applied identically to img / mask / corner coords.

    Corners that land outside the frame after the warp have their
    visibility flag set to 0 — the loss masks invisible channels so an
    empty target does not become a "no peak" training signal.
    """
    H, W = img.shape[:2]
    angle = random.uniform(-angle_deg, angle_deg)
    tx = random.uniform(-translate_frac, translate_frac) * W
    ty = random.uniform(-translate_frac, translate_frac) * H
    scale = random.uniform(*scale_range)

    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    # BORDER_CONSTANT (black) instead of BORDER_REFLECT here, intentionally:
    # under reflection a gate near the frame edge gets *reflected* into the
    # visible area while its corner labels are marked invisible (the
    # corner pixel is outside [0, W)). The model would learn to ignore
    # gate-like features near edges, which is the opposite of what we
    # want at deployment where edge gates are real. Black borders are
    # obviously not gates and the model learns no spurious pattern.
    img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Apply the same M to corners. cv2 affine matrices are 2×3 in the form
    # [[a, b, tx], [c, d, ty]]; for points we use [x, y, 1] @ M^T.
    corners_h = np.hstack([corners, np.ones((corners.shape[0], 1), dtype=corners.dtype)])
    corners_t = corners_h @ M.T  # (K, 2)
    in_frame = (
        (corners_t[:, 0] >= 0) & (corners_t[:, 0] < W) &
        (corners_t[:, 1] >= 0) & (corners_t[:, 1] < H)
    )
    # Multiplicative AND: a corner that was already invisible stays invisible.
    vis = (vis.astype(np.int32) * in_frame.astype(np.int32)).astype(vis.dtype)
    return img, mask, corners_t.astype(corners.dtype), vis


def train_augment_kp(
    img: np.ndarray,
    mask: np.ndarray,
    corners: np.ndarray,
    vis: np.ndarray,
    *,
    strong: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keypoint-aware training augmentation chain.

    Spatial transforms (flip, affine) are applied to img + mask + corner
    coords + vis flags simultaneously; appearance transforms (HSV, blur,
    noise) only touch the image. ``strong=True`` adds perspective + photo-
    metric jitter + cutout + JPEG (see :func:`train_augment` for the
    rationale).
    """
    img, mask, corners, vis = random_flip_kp(img, mask, corners, vis)
    if strong:
        img, mask, corners, vis = random_perspective_kp(img, mask, corners, vis)
    img, mask, corners, vis = random_affine_kp(img, mask, corners, vis)
    img = hsv_jitter(img, h_range=20.0 if strong else 10.0)
    if strong:
        img = brightness_contrast_gamma(img)
    img = motion_blur(img)
    img = gaussian_noise(img)
    if strong:
        img = jpeg_compression(img)
        # Cutout: image only — vis flags unchanged (occlusion ≠ corner OOF).
        img, _ = random_cutout(img, mask)
    return img, mask, corners, vis


# ---------------------------------------------------------------------------
# Multi-gate variant
# ---------------------------------------------------------------------------
# The single-gate train_augment_kp expects (corners (K, 2), vis (K,)). The
# multi-gate path threads (corners (G, K, 2), vis (G, K)) through the same
# spatial transforms. The L-R flip channel-permutation is identical per
# gate (CORNER_TYPES inner half [TL, TR, BR, BL] → [TR, TL, BL, BR]
# under hflip), and the affine matrix is shared across all gates because
# they live in the same image.

def random_flip_kp_multi(
    img: np.ndarray,
    mask: np.ndarray,
    corners: np.ndarray,
    vis: np.ndarray,
    p: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Horizontal flip on img + mask + (G, K, 2) corners + (G, K) vis."""
    if random.random() < p:
        H, W = img.shape[:2]
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)
        corners = corners.copy()
        corners[..., 0] = (W - 1) - corners[..., 0]
        # Per-gate channel permutation under hflip — same [1, 0, 3, 2]
        # as the single-gate path; broadcasts across the gate axis.
        corners = corners[:, _INNER_CORNER_HFLIP_PERM, :]
        vis = vis[:, _INNER_CORNER_HFLIP_PERM]
    return img, mask, corners, vis


def random_affine_kp_multi(
    img: np.ndarray,
    mask: np.ndarray,
    corners: np.ndarray,
    vis: np.ndarray,
    *,
    angle_deg: float = 15.0,
    translate_frac: float = 0.10,
    scale_range: tuple[float, float] = (0.9, 1.1),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Affine warp on img/mask + (G, K, 2) corners. Out-of-frame → vis=0.

    Same affine matrix applied to every gate's corners — they share the
    image, so a single warp is correct. Per-corner in-frame visibility
    is recomputed; gates that lose all visible corners simply contribute
    zero peaks to the heatmap (the loss masks zero-vis channels when no
    gate has them visible).
    """
    H, W = img.shape[:2]
    angle = random.uniform(-angle_deg, angle_deg)
    tx = random.uniform(-translate_frac, translate_frac) * W
    ty = random.uniform(-translate_frac, translate_frac) * H
    scale = random.uniform(*scale_range)

    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Apply M to every (g, k) corner. Reshape to (G*K, 2), homogenise,
    # multiply, reshape back — single matmul for the whole batch.
    G, K = corners.shape[0], corners.shape[1]
    flat = corners.reshape(-1, 2)
    flat_h = np.hstack([flat, np.ones((flat.shape[0], 1), dtype=flat.dtype)])
    flat_t = (flat_h @ M.T).astype(corners.dtype)
    corners_t = flat_t.reshape(G, K, 2)

    in_frame = (
        (corners_t[..., 0] >= 0) & (corners_t[..., 0] < W) &
        (corners_t[..., 1] >= 0) & (corners_t[..., 1] < H)
    )
    vis = (vis.astype(np.int32) * in_frame.astype(np.int32)).astype(vis.dtype)
    return img, mask, corners_t, vis


def train_augment_kp_multi(
    img: np.ndarray,
    mask: np.ndarray,
    corners: np.ndarray,
    vis: np.ndarray,
    *,
    strong: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Multi-gate keypoint-aware training augmentation chain.

    Mirrors :func:`train_augment_kp` for shape ``(G, K, 2)`` corners +
    ``(G, K)`` vis. Spatial transforms shared across all gates; appearance
    transforms touch only the image. ``strong=True`` adds perspective +
    photometric jitter + cutout + JPEG (see :func:`train_augment` for
    the rationale).
    """
    img, mask, corners, vis = random_flip_kp_multi(img, mask, corners, vis)
    if strong:
        img, mask, corners, vis = random_perspective_kp_multi(img, mask, corners, vis)
    img, mask, corners, vis = random_affine_kp_multi(img, mask, corners, vis)
    img = hsv_jitter(img, h_range=20.0 if strong else 10.0)
    if strong:
        img = brightness_contrast_gamma(img)
    img = motion_blur(img)
    img = gaussian_noise(img)
    if strong:
        img = jpeg_compression(img)
        # Cutout: image only — vis flags unchanged (occlusion ≠ corner OOF).
        img, _ = random_cutout(img, mask)
    return img, mask, corners, vis
