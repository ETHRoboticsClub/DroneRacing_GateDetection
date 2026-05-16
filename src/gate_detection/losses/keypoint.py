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


def windowed_soft_argmax_2d(
    heatmap: torch.Tensor,
    *,
    window_size: int = 15,
    beta: float = 100.0,
) -> torch.Tensor:
    """Per-channel **windowed** soft-argmax around the integer-argmax pixel.

    Designed for multi-modal targets (multiple peaks per channel — one per
    visible gate). Global soft-argmax averages across all peaks; with two
    near-equal peaks at distance ``d`` it returns the midpoint, not a
    single corner. The windowed form picks the strongest peak by integer
    argmax, then refines sub-pixel within a small window so distant peaks
    cannot pull the estimate.

    Parameters
    ----------
    heatmap : Tensor of shape (B, K, H, W).
    window_size : odd integer pixel side. 15 px around a σ=3 peak captures
        the entire 5σ envelope while staying well clear of any second-gate
        peak that is even one gate-spacing away.
    beta : softmax temperature inside the window. Same role as in
        :func:`soft_argmax_2d`; β=100 gives near-integer behaviour with
        sub-pixel correction from the immediate ±1 neighbourhood.

    Returns
    -------
    Tensor of shape (B, K, 2) in (x, y) order, image-pixel units.
    """
    B, K, H, W = heatmap.shape
    flat = heatmap.reshape(B, K, -1)
    peak_idx = flat.argmax(dim=-1)            # (B, K)
    peak_y = (peak_idx // W).long()           # (B, K)
    peak_x = (peak_idx % W).long()
    half = window_size // 2

    out = torch.zeros(B, K, 2, device=heatmap.device, dtype=heatmap.dtype)
    # B and K are both small (B=1 for single-image inference, K=4 for
    # inner corners), so a Python loop here is fine; vectorisation would
    # only help for batched validation, which uses the global soft_argmax_2d.
    for b in range(B):
        for k in range(K):
            cy = int(peak_y[b, k].item())
            cx = int(peak_x[b, k].item())
            y0 = max(0, cy - half)
            y1 = min(H, cy + half + 1)
            x0 = max(0, cx - half)
            x1 = min(W, cx + half + 1)
            window = heatmap[b, k, y0:y1, x0:x1]
            wflat = window.reshape(-1)
            sm = (wflat * beta).softmax(dim=0)
            ys = torch.arange(y0, y1, device=heatmap.device, dtype=heatmap.dtype)
            xs = torch.arange(x0, x1, device=heatmap.device, dtype=heatmap.dtype)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            out[b, k, 0] = (sm * xx.reshape(-1)).sum()
            out[b, k, 1] = (sm * yy.reshape(-1)).sum()
    return out


def _find_top_peaks_2d(channel: np.ndarray, n: int = 3, nms_radius: int = 30) -> list[tuple[float, float]]:
    """Top-``n`` local maxima of a single (H, W) heatmap channel with NMS.

    NMS = no two returned peaks within ``nms_radius`` px. Returns
    ``[(x, y), ...]`` in descending peak-value order; may be shorter
    than ``n`` if the channel doesn't have that many distinct local
    maxima above background.
    """
    H, W = channel.shape
    flat = channel.reshape(-1)
    # Look at the top ~200 candidates — comfortably more than n × any
    # reasonable cluster size given σ=3 peaks.
    sorted_idx = np.argsort(flat)[::-1][:200]
    peaks: list[tuple[float, float]] = []
    r2 = nms_radius * nms_radius
    for idx in sorted_idx:
        y, x = int(idx // W), int(idx % W)
        ok = True
        for px, py in peaks:
            if (x - px) * (x - px) + (y - py) * (y - py) < r2:
                ok = False
                break
        if ok:
            peaks.append((float(x), float(y)))
            if len(peaks) >= n:
                break
    return peaks


def coherent_quad_candidates(
    heatmap: np.ndarray, top_n: int = 3,
) -> list[dict]:
    """Return ALL spatially-ordered (TL,TR,BR,BL) corner combos from the
    top-``top_n`` peaks per channel, each with its rectangularity score.

    Used by the runtime gate-tracker (report 14 / Path B) when more than
    one valid candidate is needed — the tracker picks by combining
    proximity-to-previous-track with rectangularity, rather than purely
    by rectangularity. ``coherent_quad_decode`` is a thin wrapper around
    this that just keeps the minimum-score winner.

    Each returned dict has keys:
      * ``quad``     — (4, 2) float32 array, channel order TL/TR/BR/BL
      * ``rect_score`` — ``|top_w-bot_w| + |left_h-right_h|`` (px)
      * ``centroid`` — (2,) float32, mean of the four corners
      * ``diagonal`` — average of the two diagonals (px), used as a
        proxy for "how close is this gate" (large = close)

    Sorted by ``rect_score`` ascending (most rectangular first). Empty
    list when no spatially-ordered combo exists.
    """
    if heatmap.shape[0] != 4:
        raise ValueError(
            f"coherent_quad_candidates expects K=4 channels (TL, TR, BR, BL); "
            f"got K={heatmap.shape[0]}"
        )
    peaks = [_find_top_peaks_2d(heatmap[k], n=top_n) for k in range(4)]
    if any(len(p) == 0 for p in peaks):
        return []
    candidates: list[dict] = []
    from itertools import product
    for combo in product(*peaks):
        TL, TR, BR, BL = combo
        if not (TR[0] > TL[0] and BL[1] > TL[1]
                and BR[0] > BL[0] and BR[1] > TR[1]):
            continue
        top_w   = TR[0] - TL[0]
        bot_w   = BR[0] - BL[0]
        left_h  = BL[1] - TL[1]
        right_h = BR[1] - TR[1]
        rect_score = abs(top_w - bot_w) + abs(left_h - right_h)
        quad = np.array(combo, dtype=np.float32)
        # Diagonal magnitude — proxy for gate size. Average of TL-BR and
        # TR-BL diagonals so an extreme aspect ratio doesn't dominate.
        diag = 0.5 * (
            float(np.linalg.norm(quad[2] - quad[0])) +
            float(np.linalg.norm(quad[3] - quad[1]))
        )
        candidates.append({
            "quad": quad,
            "rect_score": float(rect_score),
            "centroid": quad.mean(axis=0).astype(np.float32),
            "diagonal": diag,
        })
    candidates.sort(key=lambda c: c["rect_score"])
    return candidates


def coherent_quad_decode(heatmap: np.ndarray, top_n: int = 3) -> np.ndarray | None:
    """Pick a (K=4, 2) corner set whose 4 peaks all belong to the **same** gate.

    Designed for multi-modal heatmaps from multi-gate-trained heads
    where per-channel argmax independently picks the strongest peak —
    which can come from different gates per channel and produces a
    Frankenstein quad that matches no single GT gate (see report 10).

    Strategy: enumerate up to ``top_n`` local maxima per channel
    (with NMS), then over the ``top_n^4`` combinations check the
    spatial-order constraints (TR right of TL, BL below TL, BR right
    of BL and below TR) and pick the most rectangular combination by
    scoring ``|top_width - bot_width| + |left_height - right_height|``.

    Parameters
    ----------
    heatmap : (K, H, W) float32 array. K must be 4 with channel order
        TL, TR, BR, BL (matches CORNER_TYPES inner half).
    top_n : peaks per channel to enumerate. With G=2-3 typical visible
        gates, 3 covers the strongest peak from each gate plus a noise
        buffer. The 3^4 = 81 combinations evaluate in microseconds.

    Returns
    -------
    (4, 2) float32 array of (x, y) coords, or ``None`` if no
    spatially-ordered combination exists (e.g., heatmap has no clear
    peaks). Caller (predictor) should fall back appropriately.
    """
    # Thin wrapper around ``coherent_quad_candidates``: the original
    # "best by rectangularity" winner is the first element of the
    # sorted candidate list. Keeping a single source of truth so the
    # tracker (Path B) and the legacy decoder cannot drift apart.
    cands = coherent_quad_candidates(heatmap, top_n=top_n)
    if not cands:
        return None
    return cands[0]["quad"]


def heatmap_peak_confidence(heatmap: torch.Tensor) -> torch.Tensor:
    """Per-channel max of the heatmap → ``(B, K)`` peak values.

    Useful as a confidence gate on inference: a flat heatmap with no clear
    peak (max << 1) means the model has no opinion and the corner should be
    treated as missing. Caller decides the threshold; we don't pick one.
    """
    B, K, H, W = heatmap.shape
    return heatmap.reshape(B, K, -1).amax(dim=-1)
