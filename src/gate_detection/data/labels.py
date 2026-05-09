"""Label I/O for the YOLO-keypoint format used by the tii-racing dataset.

On-disk format
--------------
Each line is one gate. Two variants exist:

    17-col (original):
        cls cx cy w h  (inner_x inner_y vis)*4

    29-col (with outer corners):
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
import cv2
import numpy as np


CORNER_TYPES = [
    "outer_TL", "outer_TR", "outer_BR", "outer_BL",
    "inner_TL", "inner_TR", "inner_BR", "inner_BL",
]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_gates(lbl_path: str) -> list[dict]:
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
    *,
    synth_outer: bool = False,
    synth_scale: float = 1.4,
) -> list[np.ndarray]:
    """Return one ``(8, 2)`` corner array per gate in pixel coordinates.

    By default, gates without outer corners (17-col labels) or fewer than 3
    visible inner/outer corners are skipped. Set ``synth_outer=True`` to
    synthesise outer corners as ``centroid + synth_scale × (inner - centroid)``
    when only inner corners are present — matches the convention used in
    scripts/benchmark_pnp.py:79-81 and unblocks 17-col labels (the format
    TII v3.0.0 actually ships) without requiring scripts/prepare_labels.py
    (which is a known-missing script tracked as plan P0-4).

    Returns an empty list if no gate parses; emits a single ``[WARN]`` line
    when that empty return is the result of skipping (per CLAUDE.md §6,
    no silent fallbacks).
    """
    if not lbl_path or not os.path.exists(lbl_path):
        return []

    raw = parse_gates(lbl_path)
    skipped_no_outer = 0
    skipped_too_few_visible = 0
    out: list[np.ndarray] = []
    for g in raw:
        inner_visible_pts = np.array(
            [[x, y] for (x, y), v in zip(g["inner"], g["inner_vis"]) if v > 0],
            dtype=np.float32,
        )
        if g["outer"] is None:
            if not synth_outer:
                skipped_no_outer += 1
                continue
            # assign_corner_types sorts by quadrant via argmin/argmax over
            # x±y and only produces a non-degenerate (4, 2) when ALL four
            # corners are present — with three points one of them gets
            # picked for two slots. Require 4 visible to keep priors clean.
            if len(inner_visible_pts) < 4:
                skipped_too_few_visible += 1
                continue
            # Synthesise outer corners from the inner ones, matching the
            # offline 17→29-col conversion in benchmark_pnp.py:79-81.
            centroid = inner_visible_pts.mean(axis=0)
            outer_visible_pts = centroid + synth_scale * (inner_visible_pts - centroid)
        else:
            outer_visible_pts = np.array(
                [[x, y] for (x, y), v in zip(g["outer"], g["outer_vis"]) if v > 0],
                dtype=np.float32,
            )
            # Same 4-visibility requirement as above; assign_corner_types
            # silently duplicates when fewer than 4 are passed.
            if len(inner_visible_pts) < 4 or len(outer_visible_pts) < 4:
                skipped_too_few_visible += 1
                continue

        out.append(assign_corner_types(
            outer_visible_pts * img_size, inner_visible_pts * img_size,
        ))

    if not out and raw and (skipped_no_outer or skipped_too_few_visible):
        # Silent-empty-return is exactly the failure mode CLAUDE.md §6 forbids;
        # downstream callers like evaluation/benchmark.py infer "no gate label"
        # and report 0% detection without telling the user the LABEL was the
        # problem, not the detector. Announce the skip reason once per call.
        print(
            f"[WARN] parse_pixel_corners({os.path.basename(lbl_path)}, synth_outer={synth_outer}): "
            f"all {len(raw)} gates skipped "
            f"(no_outer={skipped_no_outer}, too_few_visible={skipped_too_few_visible}). "
            "If this is a TII 17-col label, pass synth_outer=True.",
            flush=True,
        )
    return out
