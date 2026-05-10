"""Keypoint heatmap loss + Gaussian target rendering + soft-argmax decode.

Used by the multi-task DeepLabV3+ keypoint variant (plan P1-1). The four
keypoints predicted are the inner gate corners in CORNER_TYPES order
(``inner_TL, inner_TR, inner_BR, inner_BL`` — see ``data/labels.py:33-36``).
Outer corners are NOT regressed; downstream code synthesises them as
``centroid + 1.4 × (inner − centroid)`` to match
``data/labels.py:parse_pixel_corners(synth_outer=True)``.

Design choices
--------------
* **Gaussian heatmap targets** (precedent: AlphaPilot, Foehn et al. 2020):
  one peak per visible inner corner, σ in pixels at the heatmap resolution.
  Invisible corners (``vis == 0``) get an all-zero channel AND the loss
  masks them out so the model is not pushed toward "no peak anywhere".
* **Weighted MSE** with ``fg_weight`` upweighting near-peak pixels — plain
  MSE is dominated by the background-zero pixels, so we follow the
  ``weight = fg_weight × target + 1`` recipe used by Newell et al. 2016
  (Stacked Hourglass, hence widely-used keypoint MSE convention).
* **Global soft-argmax** for sub-pixel decode at inference and for the
  validation corner-distance metric. With ``beta=100`` and σ=3 px Gaussian
  targets, the soft-argmax is numerically indistinguishable from a
  parabolic-fit-on-argmax sub-pixel estimator while staying vectorised.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def render_heatmap(
    corners_px: np.ndarray,
    vis: np.ndarray,
    *,
    size: int,
    sigma: float,
) -> np.ndarray:
    """Render a per-corner Gaussian heatmap target.

    Parameters
    ----------
    corners_px
        ``(K, 2)`` array of corner ``(x, y)`` pixel coordinates at the
        ``size × size`` resolution.
    vis
        ``(K,)`` integer/boolean visibility flags. Invisible corners
        (``vis[k] == 0``) produce an all-zero channel; the loss is expected
        to mask them out so the all-zero target does not become a training
        signal of "no peak ever".
    size
        Heatmap spatial resolution (square). Must match the model output
        size — caller's responsibility.
    sigma
        Gaussian peak sigma in pixels at the heatmap resolution. With
        size=384 and sigma=3 px the 3σ effective peak is ~9 px wide.

    Returns
    -------
    np.ndarray
        ``(K, size, size)`` float32 heatmap tensor. Peak value is exactly
        1.0 at the corner pixel (before any sub-pixel quantisation), 0
        elsewhere on a Gaussian falloff.
    """
    K = corners_px.shape[0]
    out = np.zeros((K, size, size), dtype=np.float32)
    yy, xx = np.meshgrid(
        np.arange(size, dtype=np.float32),
        np.arange(size, dtype=np.float32),
        indexing="ij",
    )
    two_sigma_sq = 2.0 * sigma * sigma
    for k in range(K):
        if vis[k] <= 0:
            continue
        cx, cy = float(corners_px[k, 0]), float(corners_px[k, 1])
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        out[k] = np.exp(-d2 / two_sigma_sq)
    return out


def render_heatmap_multi(
    corners_px: np.ndarray,
    vis: np.ndarray,
    *,
    size: int,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-gate version of :func:`render_heatmap`.

    Parameters
    ----------
    corners_px
        ``(G, K, 2)`` array of inner-corner pixel coordinates for ``G``
        gates × ``K`` corners each.
    vis
        ``(G, K)`` integer / boolean visibility flags. Per-(gate, corner)
        invisible entries do not contribute a peak.
    size
        Heatmap spatial resolution (square), as in :func:`render_heatmap`.
    sigma
        Gaussian σ in pixels.

    Returns
    -------
    out : np.ndarray
        ``(K, size, size)`` float32 heatmap. Peak per channel is the
        per-pixel **max** across gates (not sum) so the heatmap stays
        bounded in ``[0, 1]`` even when two gates' corners coincide —
        keeps the sigmoid head's loss surface well-conditioned.
    vis_per_channel : np.ndarray
        ``(K,)`` int32 mask, 1 if **any** gate has that corner visible,
        0 otherwise. The MSE loss masks invisible channels so a channel
        with no gate-visible corner contributes 0 — without this mask
        the loss would push the sigmoid toward 0 everywhere on those
        channels and hurt all-gates-occluded frames disproportionately.

    Rationale
    ---------
    Empirical motivation in ``docs/report/09_largest_gate_null_result``:
    single-gate-of-interest supervision (whether by file order or by
    bbox area) has frame-to-frame ambiguity that the network learns AS
    NOISE in multi-gate scenes. Multi-gate Gaussian targets remove the
    "which gate" ambiguity from the loss entirely; the network is
    rewarded for any peak at any gate's corner. Inference still does
    soft-argmax (single instance), so deployment behaviour is unchanged.
    The MonoRace paper (Bahnam et al. 2026) handles multi-gate by
    *adaptive cropping* at the input via an EKF prior; we don't have
    EKF state online yet, so this is the alternative path that also
    avoids contradictory loss gradients.
    """
    if corners_px.ndim != 3 or corners_px.shape[-1] != 2:
        raise ValueError(
            f"corners_px must have shape (G, K, 2); got {corners_px.shape}"
        )
    G, K, _ = corners_px.shape
    out = np.zeros((K, size, size), dtype=np.float32)
    vis_per_channel = np.zeros(K, dtype=np.int32)
    if G == 0:
        return out, vis_per_channel
    yy, xx = np.meshgrid(
        np.arange(size, dtype=np.float32),
        np.arange(size, dtype=np.float32),
        indexing="ij",
    )
    two_sigma_sq = 2.0 * sigma * sigma
    for g in range(G):
        for k in range(K):
            if vis[g, k] <= 0:
                continue
            cx, cy = float(corners_px[g, k, 0]), float(corners_px[g, k, 1])
            d2 = (xx - cx) ** 2 + (yy - cy) ** 2
            peak = np.exp(-d2 / two_sigma_sq).astype(np.float32)
            out[k] = np.maximum(out[k], peak)
            vis_per_channel[k] = 1
    return out, vis_per_channel


def weighted_heatmap_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    vis: torch.Tensor,
    *,
    fg_weight: float = 10.0,
) -> torch.Tensor:
    """Per-pixel-weighted MSE between predicted and target heatmaps.

    ``pred`` and ``target`` are ``(B, K, H, W)``. ``vis`` is ``(B, K)`` with
    1 for visible corners and 0 otherwise. The per-pixel weight is
    ``fg_weight * target + 1`` so that pixels near the Gaussian peak (where
    ``target → 1``) contribute ``fg_weight + 1`` times more than empty
    background pixels (where ``target == 0``).

    The reduction is ``sum(per-channel mean) / number of visible channels``;
    a batch where every corner is invisible returns 0 (clamped denominator)
    so the loss never NaNs on a degenerate batch.
    """
    err = (pred - target).pow(2)            # (B, K, H, W)
    weight = fg_weight * target + 1.0       # (B, K, H, W)
    per_chan = (err * weight).mean(dim=(2, 3))  # (B, K)
    valid = vis.float()                     # (B, K)
    n_valid = valid.sum().clamp(min=1.0)
    return (per_chan * valid).sum() / n_valid


def soft_argmax_2d(
    heatmap: torch.Tensor,
    *,
    beta: float = 100.0,
) -> torch.Tensor:
    """Differentiable spatial argmax → ``(B, K, 2)`` (x, y) pixel coords.

    Computes a soft-argmax via 2-D softmax with temperature ``beta``. The
    default ``beta=100`` is large enough that on the 384² grid the
    background-pixel weight (``exp(0) ≈ 1`` per pixel × ~150k pixels) is
    dwarfed by the peak weight (``exp(100) ≈ 2.7e43``), so there is no
    centre-of-image bias. The trade-off is that with σ=3 px Gaussian
    targets the immediate-neighbour relative weight is ``exp(-100·0.055)
    ≈ 0.4 %``, which makes the soft-argmax effectively integer-pixel for
    a single heatmap. Sub-pixel comparison still works for the validation
    metric because the same bias is applied to both pred and target
    heatmaps and cancels in the difference. For absolute sub-pixel
    inference, lower ``beta`` to ~20 (knee of the bias-vs-resolution
    trade-off for σ=3 px), at the cost of some background influence.

    NOTE: returns coordinates in ``(x, y)`` order, image convention (x =
    column = right, y = row = down). The corresponding ``CORNER_TYPES`` slot
    in ``data/labels.py:33-36`` also stores ``(x, y)``, so values here are
    directly comparable to GT corner arrays.
    """
    B, K, H, W = heatmap.shape
    flat = (heatmap * beta).reshape(B, K, -1)
    soft = flat.softmax(dim=-1)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=heatmap.device, dtype=heatmap.dtype),
        torch.arange(W, device=heatmap.device, dtype=heatmap.dtype),
        indexing="ij",
    )
    gy = grid_y.reshape(-1)  # (H*W,)
    gx = grid_x.reshape(-1)
    cx = (soft * gx).sum(dim=-1)  # (B, K)
    cy = (soft * gy).sum(dim=-1)
    return torch.stack([cx, cy], dim=-1)  # (B, K, 2)


def heatmap_peak_confidence(heatmap: torch.Tensor) -> torch.Tensor:
    """Per-channel max of the heatmap → ``(B, K)`` peak values.

    Useful as a confidence gate on inference: a flat heatmap with no clear
    peak (max << 1) means the model has no opinion and the corner should be
    treated as missing. Caller decides the threshold; we don't pick one.
    """
    B, K, H, W = heatmap.shape
    return heatmap.reshape(B, K, -1).amax(dim=-1)
