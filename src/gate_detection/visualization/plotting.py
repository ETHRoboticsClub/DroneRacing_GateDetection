"""Matplotlib helpers for the inference / IoU / flight-benchmark grids.

Every public function here writes a PNG to ``out_path`` (creating parent
directories) and prints the saved path. They are intentionally pure-plotting
— no model inference happens inside.
"""

from __future__ import annotations

import os

import matplotlib

# Headless backend so this module is import-safe in non-GUI contexts (e.g. CI).
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np                # noqa: E402

from gate_detection.data.labels import CORNER_TYPES         # noqa: E402
from gate_detection.postprocessing.quadgate import detect_corners_debug  # noqa: E402
from .colors import CORNER_COLORS              # noqa: E402


# ---------------------------------------------------------------------------
# Internal save helper
# ---------------------------------------------------------------------------
def _savefig(fig, out_path: str) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Grid saved → {out_path}")


# ---------------------------------------------------------------------------
# QuAdGate step renderers
# ---------------------------------------------------------------------------
def _render_lines(ax, mask_rot, lines_ext):
    ax.imshow(mask_rot, cmap="gray", vmin=0, vmax=255)
    if lines_ext is not None and len(lines_ext):
        for x1, y1, x2, y2 in lines_ext:
            ax.plot([x1, x2], [y1, y2], color="cyan", linewidth=0.8, alpha=0.85)
        ax.set_title(f"{len(lines_ext)} lines", fontsize=7, pad=2)
    else:
        ax.set_title("no lines", fontsize=7, color="red", pad=2)
    ax.axis("off")


def _render_candidates(ax, mask_rot, cand_pts):
    ax.imshow(mask_rot, cmap="gray", vmin=0, vmax=255)
    if cand_pts is not None and len(cand_pts):
        ax.scatter(cand_pts[:, 0], cand_pts[:, 1],
                   s=6, c="lime", marker=".", linewidths=0, alpha=0.75)
        ax.set_title(f"{len(cand_pts)} candidates", fontsize=7, pad=2)
    else:
        ax.set_title("no candidates", fontsize=7, color="red", pad=2)
    ax.axis("off")


def _render_matches(ax, mask_rot, priors_rot, matches):
    ax.imshow(mask_rot, cmap="gray", vmin=0, vmax=255)
    if priors_rot is not None:
        for i, ctype in enumerate(CORNER_TYPES):
            color = CORNER_COLORS[ctype]
            px, py = priors_rot[i]
            ax.plot(px, py, "x", color=color, markersize=5, markeredgewidth=1.2)
            if ctype in matches:
                mx, my = matches[ctype]
                ax.plot(mx, my, "o", color=color, markersize=4, markeredgewidth=0)
                ax.plot([px, mx], [py, my], "-", color=color, linewidth=0.7, alpha=0.6)
    n = len(matches)
    color = "white" if n >= 3 else "red"
    ax.set_title(f"{n}/8 matched", fontsize=7, color=color, pad=2)
    ax.axis("off")


def _render_corners(ax, img_np, result):
    ax.imshow(img_np)
    if result is not None:
        ax.scatter(result[:, 0], result[:, 1], s=25, c="red", zorder=5, linewidths=0)
    else:
        ax.set_title("failed", fontsize=7, color="red", pad=2)
    ax.axis("off")


# ---------------------------------------------------------------------------
# Inference grid (simple — 5 columns)
# ---------------------------------------------------------------------------
_COLS_SIMPLE = ["Image", "GT mask", "Prediction", "Pred B&W", "QuAdGate"]


def plot_inference_grid(rows, ckpt_meta, out_path: str) -> None:
    """5-column inference grid.

    Each ``row`` is a dict with keys::

        img_rgb   : (H, W, 3) uint8
        gt_mask   : (H, W) uint8 (binary)
        pred_prob : (H, W) float in [0, 1]
        priors    : list of (8, 2) float arrays  (may be empty)
        fname     : str
    """
    n_rows = len(rows)
    n_cols = len(_COLS_SIMPLE)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    if n_rows == 1:
        axes = axes[None, :]

    fig.suptitle(_ckpt_title(ckpt_meta), fontsize=12)
    for title, ax in zip(_COLS_SIMPLE, axes[0]):
        ax.set_title(title, fontsize=9)

    for ri, row in enumerate(rows):
        img_np = row["img_rgb"]
        mask_np = row["gt_mask"]
        pred_np = row["pred_prob"]
        priors = row["priors"]

        axes[ri, 0].imshow(img_np)
        axes[ri, 0].set_ylabel(row["fname"], fontsize=6, rotation=0,
                               labelpad=60, va="center")
        axes[ri, 0].axis("off")

        axes[ri, 1].imshow(img_np)
        axes[ri, 1].imshow(mask_np, cmap="Greens", alpha=0.6, vmin=0, vmax=1)
        axes[ri, 1].axis("off")

        axes[ri, 2].imshow(img_np)
        axes[ri, 2].imshow(pred_np, cmap="Reds", alpha=0.6, vmin=0, vmax=1)
        axes[ri, 2].axis("off")

        axes[ri, 3].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
        axes[ri, 3].axis("off")

        ax_qag = axes[ri, 4]
        ax_qag.imshow(img_np)
        ax_qag.imshow(pred_np, cmap="gray", alpha=0.4, vmin=0, vmax=1)
        any_ok = False
        for prior in priors:
            debug = detect_corners_debug(pred_np, prior)
            if debug["lines_ext"] is not None:
                for x1, y1, x2, y2 in debug["lines_ext"]:
                    ax_qag.plot([x1, x2], [y1, y2], color="cyan",
                                linewidth=0.8, alpha=0.7)
            if debug["result"] is not None:
                any_ok = True
                pts = debug["result"]
                ax_qag.scatter(pts[:, 0], pts[:, 1],
                               s=25, c="red", zorder=5, linewidths=0)
        if not priors:
            ax_qag.set_title("no label", fontsize=8, color="gray")
        elif not any_ok:
            ax_qag.set_title("failed", fontsize=8, color="red")
        ax_qag.axis("off")

    _savefig(fig, out_path)


# ---------------------------------------------------------------------------
# Inference grid (debug — every QuAdGate step in its own column)
# ---------------------------------------------------------------------------
_COLS_DEBUG = [
    "Image", "GT mask", "Pred mask\n(QAG input)",
    "QAG 1\nlines", "QAG 2\ncandidates", "QAG 3\nmatches", "QAG 4\ncorners",
]


def plot_inference_grid_with_qag_steps(rows, ckpt_meta, out_path: str) -> None:
    """7-column grid showing every QuAdGate intermediate step."""
    n_rows = len(rows)
    n_cols = len(_COLS_DEBUG)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    if n_rows == 1:
        axes = axes[None, :]

    fig.suptitle(_ckpt_title(ckpt_meta), fontsize=12)
    for title, ax in zip(_COLS_DEBUG, axes[0]):
        ax.set_title(title, fontsize=9)

    for ri, row in enumerate(rows):
        img_np = row["img_rgb"]
        mask_np = row["gt_mask"]
        pred_np = row["pred_prob"]
        priors = row["priors"]

        axes[ri, 0].imshow(img_np)
        axes[ri, 0].set_ylabel(row["fname"], fontsize=6, rotation=0,
                               labelpad=60, va="center")
        axes[ri, 0].axis("off")

        axes[ri, 1].imshow(img_np)
        axes[ri, 1].imshow(mask_np, cmap="Greens", alpha=0.6, vmin=0, vmax=1)
        axes[ri, 1].axis("off")

        if priors:
            ax_pred, ax_lines, ax_cands, ax_matches, ax_final = axes[ri, 2:]
            ax_pred.imshow(pred_np, cmap="gray", vmin=0, vmax=1)
            ax_pred.axis("off")
            debug = detect_corners_debug(pred_np, priors[0])
            mask_rot = debug["mask_rot"]
            _render_lines(ax_lines, mask_rot, debug["lines_ext"])
            _render_candidates(ax_cands, mask_rot, debug["cand_pts"])
            _render_matches(ax_matches, mask_rot, debug["priors_rot"], debug["matches"])
            _render_corners(ax_final, img_np, debug["result"])
        else:
            for ax in axes[ri, 2:]:
                ax.imshow(pred_np, cmap="gray", vmin=0, vmax=1)
                ax.set_title("no label", fontsize=7, color="gray", pad=2)
                ax.axis("off")

    _savefig(fig, out_path)


# ---------------------------------------------------------------------------
# IoU evaluation grid
# ---------------------------------------------------------------------------
def plot_iou_grid(visuals, ckpt_meta, mean_iou: float, median_iou: float,
                  threshold: float, out_path: str) -> None:
    """4-column grid: image | GT mask | pred prob | pred binary."""
    n_rows = len(visuals)
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 3.2))
    if n_rows == 1:
        axes = axes[None, :]

    col_titles = ["Image", "GT mask", "Pred (prob)", f"Pred ≥{threshold}"]
    for title, ax in zip(col_titles, axes[0]):
        ax.set_title(title, fontsize=9)

    fig.suptitle(
        f"{_ckpt_title(ckpt_meta)}  |  "
        f"mean IoU = {mean_iou:.3f}   median = {median_iou:.3f}   n = {n_rows}",
        fontsize=10,
    )

    for ri, s in enumerate(visuals):
        color = "green" if s.iou >= 0.75 else ("orange" if s.iou >= 0.5 else "red")

        axes[ri, 0].imshow(s.img_rgb)
        axes[ri, 0].set_ylabel(f"IoU={s.iou:.3f}\n{s.fname}",
                               fontsize=6, rotation=0, labelpad=72,
                               va="center", color=color)
        axes[ri, 0].axis("off")

        axes[ri, 1].imshow(s.gt_mask,   cmap="gray", vmin=0, vmax=1); axes[ri, 1].axis("off")
        axes[ri, 2].imshow(s.pred_prob, cmap="hot",  vmin=0, vmax=1); axes[ri, 2].axis("off")
        axes[ri, 3].imshow(s.pred_bin,  cmap="gray", vmin=0, vmax=1); axes[ri, 3].axis("off")

    _savefig(fig, out_path)


# ---------------------------------------------------------------------------
# Flight benchmark grid
# ---------------------------------------------------------------------------
def _draw_corners(ax, corners, gt_gates):
    """Green + = GT inner corners, red x = detected inner corners."""
    for gt in gt_gates:
        for pt in gt[4:8]:
            ax.plot(pt[0], pt[1], "g+", markersize=7, markeredgewidth=1.5)
    if corners is not None:
        for pt in corners[4:8]:
            ax.plot(pt[0], pt[1], "rx", markersize=7, markeredgewidth=1.5)


def plot_flight_grid(visuals, flight_name: str, fps: float,
                     n_total: int, n_detected: int, out_path: str) -> None:
    """3-column grid: image+corners | pred mask | textual status."""
    if not visuals:
        print("No frames to visualise.")
        return

    n_rows = len(visuals)
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.2))
    if n_rows == 1:
        axes = axes[None, :]

    for title, ax in zip(["Image  (green=GT  red=detected)", "Pred mask", "Result"], axes[0]):
        ax.set_title(title, fontsize=8)

    fig.suptitle(
        f"{flight_name}  |  detected {n_detected}/{n_total} frames "
        f"({100 * n_detected / max(n_total, 1):.1f}%)  {fps:.1f} fps",
        fontsize=10,
    )

    for ri, s in enumerate(visuals):
        det_ok = s.detected is not None
        color = "green" if det_ok else "red"
        ce_str = f"{s.cerr:.1f} px" if s.cerr is not None else "—"

        axes[ri, 0].imshow(s.img_rgb)
        _draw_corners(axes[ri, 0], s.detected, s.gt_gates)
        axes[ri, 0].set_ylabel(
            f"{'OK' if det_ok else 'miss'}  cerr={ce_str}\n{s.fname}",
            fontsize=5.5, rotation=0, labelpad=90, va="center", color=color,
        )
        axes[ri, 0].axis("off")

        axes[ri, 1].imshow(s.pred_prob, cmap="hot", vmin=0, vmax=1)
        axes[ri, 1].axis("off")

        axes[ri, 2].axis("off")
        axes[ri, 2].text(0.5, 0.58, "DETECTED" if det_ok else "missed",
                         ha="center", va="center", fontsize=11,
                         color=color, transform=axes[ri, 2].transAxes)
        axes[ri, 2].text(0.5, 0.38, ce_str,
                         ha="center", va="center", fontsize=9,
                         color="black", transform=axes[ri, 2].transAxes)

    _savefig(fig, out_path)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def _ckpt_title(ckpt_meta: dict) -> str:
    name = ckpt_meta.get("name", "checkpoint")
    return (f"{name}  |  epoch {ckpt_meta.get('epoch', '?')}  |"
            f"  val_loss={ckpt_meta.get('val_loss', float('nan')):.4f}")
