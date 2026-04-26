"""Label I/O for the YOLO-keypoint format used by the tii-racing dataset.

On-disk format
--------------
Each line is one gate. Two variants exist:

    17-col (original):
        cls cx cy w h  (inner_x inner_y vis)*4

    29-col (synthetic, with outer corners added by ``generate_labels.py``):
        cls cx cy w h  (inner_x inner_y vis)*4  (outer_x outer_y vis)*4

All coordinates are normalised to ``[0, 1]``.

Corner ordering
---------------
The pipeline addresses corners by the canonical order

    CORNER_TYPES = [outer_TL, outer_TR, outer_BR, outer_BL,
                    inner_TL, inner_TR, inner_BR, inner_BL]

which is also the order produced by :func:`assign_corner_types` and
consumed downstream by QuAdGate / PnP.
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np


CORNER_TYPES = [
    "outer_TL", "outer_TR", "outer_BR", "outer_BL",
    "inner_TL", "inner_TR", "inner_BR", "inner_BL",
]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_gates(lbl_path: str) -> list[dict[str, Any]]:
    """Parse a label file into a list of gate dicts.

    Each dict has::

        cls       : int
        cx, cy    : float        # normalised bbox centre
        w, h      : float        # normalised bbox size
        inner     : (4, 2) float  normalised inner corners (xy)
        inner_vis : list[int]    # length 4
        outer     : (4, 2) float  normalised outer corners — None if 17-col
        outer_vis : list[int]    # length 4 — None if 17-col

    Lines with fewer than 17 columns are skipped; otherwise outer info is
    set when the line has ≥ 29 columns.
    """
    gates: list[dict[str, Any]] = []
    if not lbl_path or not os.path.exists(lbl_path):
        return gates

    with open(lbl_path) as f:
        for line in f:
            vals = list(map(float, line.split()))
            if len(vals) < 17:
                continue

            inner = np.array([[vals[5 + 3 * k], vals[6 + 3 * k]] for k in range(4)],
                             dtype=np.float32)
            inner_vis = [int(vals[7 + 3 * k]) for k in range(4)]

            if len(vals) >= 29:
                outer = np.array([[vals[17 + 3 * k], vals[18 + 3 * k]] for k in range(4)],
                                 dtype=np.float32)
                outer_vis = [int(vals[19 + 3 * k]) for k in range(4)]
            else:
                outer = None
                outer_vis = None

            gates.append({
                "cls": int(vals[0]),
                "cx": vals[1], "cy": vals[2], "w": vals[3], "h": vals[4],
                "inner": inner, "inner_vis": inner_vis,
                "outer": outer, "outer_vis": outer_vis,
            })
    return gates


def all4_visible(lbl_path: str) -> bool:
    """Return True iff every gate in the file has all 4 inner corners visible."""
    gates = parse_gates(lbl_path)
    if not gates:
        return False
    for g in gates:
        if not all(v > 0 for v in g["inner_vis"]):
            return False
    return True


# ---------------------------------------------------------------------------
# Outer-corner synthesis
# ---------------------------------------------------------------------------
def synthesize_outer(inner_norm: np.ndarray, scale: float = 1.4) -> np.ndarray:
    """Synthesise outer corners by radial scaling from the inner centroid."""
    centroid = inner_norm.mean(axis=0)
    return centroid + scale * (inner_norm - centroid)


def make_synthetic_label_lines(lbl_path: str, scale: float = 1.4) -> list[str]:
    """Read a 17-col label file and return the 29-col line strings.

    Outer-corner visibility is set to 2 only when (a) the matching inner
    corner is visible and (b) the synthesised outer point falls inside the
    [0, 1] image bounds; otherwise the outer entry is ``"0.0 0.0 0"``.
    """
    out: list[str] = []
    with open(lbl_path) as f:
        for line in f:
            vals = list(map(float, line.split()))
            if len(vals) < 17:
                continue
            cls = int(vals[0])
            cx, cy, w, h = vals[1:5]

            inner = np.array([[vals[5 + 3 * k], vals[6 + 3 * k]] for k in range(4)],
                             dtype=np.float32)
            vis_flags = [vals[7 + 3 * k] > 0 for k in range(4)]

            head = f"{cls} {cx} {cy} {w} {h}"
            for k in range(4):
                head += f" {vals[5 + 3 * k]} {vals[6 + 3 * k]} {int(vals[7 + 3 * k])}"

            outer = synthesize_outer(inner, scale=scale)
            tail = ""
            for (ox, oy), v_inner in zip(outer, vis_flags):
                in_bounds = 0.0 <= ox <= 1.0 and 0.0 <= oy <= 1.0
                vis_out = 2 if (v_inner and in_bounds) else 0
                tail += f" {float(ox):.10f} {float(oy):.10f} {vis_out}" if vis_out \
                        else " 0.0 0.0 0"

            out.append(head + tail)
    return out


def gates_to_lines(gates: list[dict[str, Any]]) -> list[str]:
    """Serialise a list of parsed gates back to label lines (29-col when outer present).

    The bbox (cx, cy, w, h) is *recomputed* from the visible corners so that
    coordinate transforms (warp, scale) propagate to the bbox automatically.
    """
    lines: list[str] = []
    for g in gates:
        all_pts = []
        if g.get("outer") is not None:
            all_pts.extend(pt for pt, v in zip(g["outer"], g["outer_vis"]) if v > 0)
        all_pts.extend(pt for pt, v in zip(g["inner"], g["inner_vis"]) if v > 0)
        if not all_pts:
            continue
        all_pts = np.asarray(all_pts)
        cx = float(all_pts[:, 0].mean())
        cy = float(all_pts[:, 1].mean())
        w  = float(all_pts[:, 0].max() - all_pts[:, 0].min())
        h  = float(all_pts[:, 1].max() - all_pts[:, 1].min())

        line = f"{g['cls']} {cx} {cy} {w} {h}"
        for pt, v in zip(g["inner"], g["inner_vis"]):
            line += f" {pt[0]:.8f} {pt[1]:.8f} {v}"
        if g.get("outer") is not None:
            for pt, v in zip(g["outer"], g["outer_vis"]):
                line += f" {pt[0]:.8f} {pt[1]:.8f} {v}"
        lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# Mask building
# ---------------------------------------------------------------------------
def corners_to_mask(lbl_path: str, out_size: int) -> np.ndarray:
    """Build a binary segmentation mask from a label file.

    * 29-col format → ring mask (outer polygon filled, then inner polygon erased).
    * 17-col format → inner polygon filled.

    Returns a uint8 array of shape ``(out_size, out_size)`` with values in {0, 1}.
    """
    mask = np.zeros((out_size, out_size), dtype=np.uint8)
    if not lbl_path or not os.path.exists(lbl_path):
        return mask

    def visible_pts(corners: np.ndarray, vis: list[int]) -> list[list[int]]:
        pts = []
        for (x, y), v in zip(corners, vis):
            if v > 0:
                px = int(np.clip(x * out_size, 0, out_size - 1))
                py = int(np.clip(y * out_size, 0, out_size - 1))
                pts.append([px, py])
        return pts

    for g in parse_gates(lbl_path):
        inner_pts = visible_pts(g["inner"], g["inner_vis"])

        if g["outer"] is not None:
            # Ring mask: fill outer with 1, then erase inner with 0. The
            # ORDER matters — the inner erase must come second to "punch a
            # hole" through the previously-filled outer polygon.
            outer_pts = visible_pts(g["outer"], g["outer_vis"])
            if len(outer_pts) >= 3:
                cv2.fillPoly(mask, [np.array(outer_pts, np.int32)], 1)
            if len(inner_pts) >= 3:
                cv2.fillPoly(mask, [np.array(inner_pts, np.int32)], 0)
        else:
            # 17-col format has no outer corners; fall back to filling the
            # inner quadrilateral. This is a degraded supervision target
            # (no ring → no descriptor signal for QuAdGate) and exists only
            # for debugging old data.
            if len(inner_pts) >= 3:
                cv2.fillPoly(mask, [np.array(inner_pts, np.int32)], 1)
            elif len(inner_pts) == 2:
                cv2.line(mask, tuple(inner_pts[0]), tuple(inner_pts[1]), 1, thickness=4)

    return mask


# ---------------------------------------------------------------------------
# Pixel-corner extraction (for QuAdGate priors and ground-truth corners)
# ---------------------------------------------------------------------------
def assign_corner_types(corners_outer: np.ndarray, corners_inner: np.ndarray) -> np.ndarray:
    """Sort outer (4,2) and inner (4,2) into [TL, TR, BR, BL] each, then stack.

    Sorting rule (image coords, +y down):
        TL = argmin(x + y), TR = argmax(x - y),
        BR = argmax(x + y), BL = argmin(x - y).

    Returns an ``(8, 2)`` array in :data:`CORNER_TYPES` order.
    """
    def sort_corners(pts: np.ndarray) -> np.ndarray:
        # Identify the four corners of an axis-aligned-ish quadrilateral:
        #   x+y is smallest at TL (small x AND small y → both negative-ish)
        #   x+y is largest  at BR (large x AND large y)
        #   x-y is largest  at TR (large x, small y)
        #   x-y is smallest at BL (small x, large y)
        # Robust to mild rotation; would need a different rule if the gate
        # were rotated past ~45° (which is why QuAdGate derotates first).
        s = pts[:, 0] + pts[:, 1]
        d = pts[:, 0] - pts[:, 1]
        TL = pts[np.argmin(s)]
        TR = pts[np.argmax(d)]
        BR = pts[np.argmax(s)]
        BL = pts[np.argmin(d)]
        return np.stack([TL, TR, BR, BL])

    return np.vstack([sort_corners(corners_outer), sort_corners(corners_inner)])


def parse_pixel_corners(
    lbl_path: str,
    img_size: int,
    outer_scale: float = 1.4,
) -> list[np.ndarray]:
    """Return one ``(8, 2)`` corner array per gate in pixel coordinates.

    For 17-col labels (no outer), outer corners are synthesised from the
    inner centroid using :func:`synthesize_outer` with ``scale=outer_scale``.
    Gates with fewer than 3 visible inner corners are skipped.
    """
    if not lbl_path or not os.path.exists(lbl_path):
        return []

    out: list[np.ndarray] = []
    for g in parse_gates(lbl_path):
        inner_visible = np.array(
            [[x, y] for (x, y), v in zip(g["inner"], g["inner_vis"]) if v > 0],
            dtype=np.float32,
        )
        if len(inner_visible) < 3:
            continue
        inner_px = inner_visible * img_size

        if g["outer"] is not None:
            outer_visible = np.array(
                [[x, y] for (x, y), v in zip(g["outer"], g["outer_vis"]) if v > 0],
                dtype=np.float32,
            )
            if len(outer_visible) < 3:
                continue
            outer_px = outer_visible * img_size
        else:
            outer_px = synthesize_outer(inner_px, scale=outer_scale)

        out.append(assign_corner_types(outer_px, inner_px))
    return out
