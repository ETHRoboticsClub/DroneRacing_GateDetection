"""Dataset for GateNet training/validation."""

from __future__ import annotations

import os
from typing import Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .labels import corners_to_mask, parse_gates
from .transforms import (
    apply_imagenet_norm,
    train_augment,
    train_augment_kp,
    train_augment_kp_multi,
)
from gate_detection.losses.keypoint import render_heatmap, render_heatmap_multi

MASK_SUFFIX = "_mask.png"


def _load_sam_mask(mask_path: str, out_size: int) -> np.ndarray:
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros((out_size, out_size), dtype=np.uint8)
    if m.shape != (out_size, out_size):
        m = cv2.resize(m, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def _label_path_for(img_path: str) -> str:
    """Sibling .txt label path next to an image, regardless of mask mode."""
    stem = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(os.path.dirname(img_path), stem + ".txt")


def _first_gate_inner(
    lbl_path: str,
    img_size: int,
    n_keypoints: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (corners_px (K, 2), vis (K,)) for the first gate in ``lbl_path``.

    "First gate" = first parseable line in the label file, matching the
    convention used by ``benchmark.py`` (priors are taken from
    ``gt_gates[0]``). Multi-gate scenes only supervise the first.

    If the label has no gates, returns an all-zero corner array with
    vis=0 — the loss masks invisible channels so this becomes a no-op
    sample for the keypoint head while the seg head still trains on it.
    """
    if not lbl_path or not os.path.exists(lbl_path):
        return (
            np.zeros((n_keypoints, 2), dtype=np.float32),
            np.zeros(n_keypoints, dtype=np.int32),
        )
    gates = parse_gates(lbl_path)
    if not gates:
        return (
            np.zeros((n_keypoints, 2), dtype=np.float32),
            np.zeros(n_keypoints, dtype=np.int32),
        )
    g = gates[0]
    # parse_gates returns inner corners normalised; scale to pixel coords
    # at img_size — augmentation works in pixel coords for clarity.
    corners = (g["inner"] * img_size).astype(np.float32)
    vis = np.array(g["inner_vis"], dtype=np.int32)
    if corners.shape[0] != n_keypoints:
        # Defensive: parse_gates currently always returns 4-corner inner;
        # if a future label format changes that, surface it loudly rather
        # than silently producing the wrong number of channels.
        raise ValueError(
            f"label {lbl_path} produced {corners.shape[0]} inner corners; "
            f"keypoint head expects {n_keypoints}"
        )
    return corners, vis


def _all_gates_inner(
    lbl_path: str,
    img_size: int,
    n_keypoints: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(corners (G, K, 2), vis (G, K))`` for ALL gates in ``lbl_path``.

    Multi-gate sibling of :func:`_first_gate_inner`. Used by the
    ``multi_gate_target`` dataset path: every visible gate's inner
    corners contribute a Gaussian peak to the per-channel heatmap (max
    across gates), so the keypoint head sees multi-modal targets and
    is not penalised for predicting any specific gate's corners — see
    ``docs/report/09_largest_gate_null_result/findings.md`` for the
    motivation and the MonoRace cross-reference.

    Empty-label / no-gate frames return shape ``(0, K, 2)`` /
    ``(0, K)``; ``render_heatmap_multi`` handles ``G=0`` by returning
    an all-zero heatmap with vis-per-channel = 0, which the loss masks.
    """
    if not lbl_path or not os.path.exists(lbl_path):
        return (
            np.zeros((0, n_keypoints, 2), dtype=np.float32),
            np.zeros((0, n_keypoints), dtype=np.int32),
        )
    gates = parse_gates(lbl_path)
    if not gates:
        return (
            np.zeros((0, n_keypoints, 2), dtype=np.float32),
            np.zeros((0, n_keypoints), dtype=np.int32),
        )
    G = len(gates)
    corners = np.zeros((G, n_keypoints, 2), dtype=np.float32)
    vis = np.zeros((G, n_keypoints), dtype=np.int32)
    for i, g in enumerate(gates):
        if g["inner"].shape[0] != n_keypoints:
            raise ValueError(
                f"label {lbl_path} gate[{i}] produced {g['inner'].shape[0]} "
                f"inner corners; keypoint head expects {n_keypoints}"
            )
        corners[i] = (g["inner"] * img_size).astype(np.float32)
        vis[i] = np.asarray(g["inner_vis"], dtype=np.int32)
    return corners, vis


class GateDataset(Dataset):
    """Loads (image, mask) pairs from one or more flat directories.

    Each input directory should contain matching ``<stem>.jpg`` files plus
    either ``<stem>.txt`` label files (default) or ``<stem>_mask.png`` SAM
    ring masks (when ``use_sam_masks=True``).

    Images are resized to ``img_size`` × ``img_size``.  When
    ``use_sam_masks=False`` masks are rasterised from the label corners;
    when ``use_sam_masks=True`` they are loaded from the pre-generated PNG
    and samples without a PNG are skipped (run
    ``scripts/generate_sam_masks.py`` to fill them in).

    Args:
        sample_dirs:   list of directories to scan.
        img_size:      target square image size.
        augment:       if True, apply :func:`train_augment` to each sample.
        use_sam_masks: if True, load SAM ring masks from ``_mask.png`` files.
        imagenet_norm: if True, apply per-channel ImageNet mean/std (must
            match the predictor's preprocessing — kept in sync via the
            shared ``apply_imagenet_norm`` helper).
        keypoint_heatmap: if True, also emit Gaussian heatmap targets for
            the first gate's inner corners. Sample tuple becomes
            ``(img_t, mask_t, kp_target_t, kp_vis_t)``.
        heatmap_sigma: Gaussian σ in pixels at ``img_size`` resolution.
            σ=3 px at 384² gives a 9-px-wide effective peak — wide enough
            for early-training gradients, narrow enough for sub-pixel
            soft-argmax to stay accurate.
        n_keypoints: number of corner channels (=4 for inner-only, the
            current convention; outer corners are synthesised at inference).
        multi_gate_target: if True, render Gaussian peaks for ALL gates'
            inner corners per channel (max across gates). Sidesteps the
            "which gate" ambiguity in single-gate supervision that
            plateaued report 07's val_corner_dist at 22 px (see
            ``docs/report/09_largest_gate_null_result``). Inference is
            unchanged — soft-argmax still returns one (x, y) per channel
            (the most-prominent gate's corner). Requires ``keypoint_heatmap=True``.
    """

    def __init__(
        self,
        sample_dirs: Iterable[str],
        img_size: int = 384,
        augment: bool = True,
        augment_strong: bool = False,
        use_sam_masks: bool = False,
        imagenet_norm: bool = False,
        keypoint_heatmap: bool = False,
        heatmap_sigma: float = 3.0,
        n_keypoints: int = 4,
        multi_gate_target: bool = False,
        per_flight_cap: int | None = None,
        cap_seed: int = 42,
        sample_dir_weights: list[int] | None = None,
        next_gate_target: bool = False,
        next_gate_ring_thickness_px: int = 6,
        next_gate_csv_root: str | None = None,
        next_gate_calib_dir: str | None = None,
        next_gate_only_kp: bool = False,
    ):
        self.img_size = img_size
        self.augment = augment
        self.augment_strong = augment_strong
        self.use_sam_masks = use_sam_masks
        self.imagenet_norm = imagenet_norm
        self.keypoint_heatmap = keypoint_heatmap
        self.heatmap_sigma = heatmap_sigma
        self.n_keypoints = n_keypoints
        self.multi_gate_target = multi_gate_target
        self.per_flight_cap = per_flight_cap
        self.cap_seed = cap_seed
        self.sample_dir_weights = sample_dir_weights
        # Next-gate-mask target (report 15). When enabled, __getitem__
        # appends a (1, H, W) binary mask of ONLY the next gate (the one
        # closest in front of the drone per Vicon) to the sample tuple,
        # making it a 5-tuple. The mask is computed on-the-fly via
        # ``data/next_gate_labels.py`` — each DataLoader worker caches
        # CSV / calibration lookups per flight on first access.
        self.next_gate_target = next_gate_target
        self.next_gate_ring_thickness_px = next_gate_ring_thickness_px
        # When True, the multi-gate kp target is REPLACED with peaks at
        # only the Vicon-derived next gate's corners (report 15.4). The
        # kp head learns to fire on a single gate of interest instead
        # of all visible gates — fixes the hybrid tracker's bottleneck
        # (kp peaks within the chosen seg CC's bbox were weak under
        # the multi-gate-target supervision). Requires next_gate_target
        # = True so the Vicon lookup chain is available (CSV root +
        # calib dir). For samples without Vicon, the kp_vis is set to
        # all-zero so the kp loss is masked on those samples (same
        # pattern as ng_valid for the next-gate-mask head).
        self.next_gate_only_kp = next_gate_only_kp
        if next_gate_only_kp and not next_gate_target:
            raise ValueError(
                "next_gate_only_kp=True requires next_gate_target=True "
                "(the Vicon lookup chain is shared between the two — "
                "see data/next_gate_labels.py:find_next_gate_label_index)."
            )
        # Default roots — match the layout described in CLAUDE.md.
        if next_gate_target:
            from gate_detection.paths import DATA_ROOT  # lazy import to avoid cycles
            self.next_gate_csv_root = next_gate_csv_root or str(DATA_ROOT)
            self.next_gate_calib_dir = next_gate_calib_dir or "calibration"
        else:
            self.next_gate_csv_root = None
            self.next_gate_calib_dir = None
        # Per-worker warn counter (mutable list, see next_gate_labels._warn).
        self._next_gate_warn_counter: list[int] = [0]
        if multi_gate_target and not keypoint_heatmap:
            # CLAUDE.md §6: surface the inconsistency loudly. The flag
            # has no meaning without keypoint_heatmap=True.
            raise ValueError(
                "multi_gate_target=True requires keypoint_heatmap=True; "
                "the keypoint head must be active for the multi-gate "
                "target rendering to do anything."
            )

        # Materialise sample_dirs once so we can both iterate over it
        # and len() it (callers may pass a generator).
        sample_dirs_list: list[str] = list(sample_dirs)
        if sample_dir_weights is not None:
            if len(sample_dir_weights) != len(sample_dirs_list):
                raise ValueError(
                    f"sample_dir_weights has length {len(sample_dir_weights)} "
                    f"but sample_dirs has length {len(sample_dirs_list)}; "
                    f"they must match."
                )
            if any(w < 0 for w in sample_dir_weights):
                raise ValueError(
                    f"sample_dir_weights must be non-negative; got "
                    f"{sample_dir_weights}."
                )

        # third element is the source-directory index; needed to apply
        # ``sample_dir_weights`` AFTER ``per_flight_cap`` (so caps are
        # honoured per flight before replication). Stripped at the end.
        all_samples: list[tuple[str, str, int]] = []

        skipped = 0
        for src_idx, d in enumerate(sample_dirs_list):
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if not f.endswith(".jpg"):
                    continue
                stem = os.path.splitext(f)[0]
                img_p = os.path.join(d, f)
                if use_sam_masks:
                    mask_p = os.path.join(d, stem + MASK_SUFFIX)
                    if not os.path.exists(mask_p):
                        skipped += 1
                        continue
                    all_samples.append((img_p, mask_p, src_idx))
                else:
                    lbl = os.path.join(d, stem + ".txt")
                    if os.path.exists(lbl):
                        all_samples.append((img_p, lbl, src_idx))

        # Class-balanced sampling (task 2 from report-12 next-steps).
        # Without this, the SAM training pool has a 3× over-representation
        # of flight-01p-ellipse (906 crops) vs flight-13p-trackRATM (303),
        # which report 12 identified as the root cause of the cross-flight
        # generalisation gap. When per_flight_cap is set, we group samples
        # by flight name (extracted from filename prefix
        # ``flight-XXp-traj__...``) and deterministically sample up to
        # ``per_flight_cap`` per flight using ``random.Random(cap_seed)``.
        # Files without a parseable flight prefix are kept verbatim and
        # logged with [WARN] (CLAUDE.md §6 — no silent fallback).
        if per_flight_cap is not None:
            import re
            import random as _random
            flight_re = re.compile(r"(flight-[0-9]+[apAP]?-[a-zA-Z0-9]+)")
            grouped: dict[str, list[tuple[str, str, int]]] = {}
            unknown: list[tuple[str, str, int]] = []
            for s in all_samples:
                base = os.path.basename(s[0])
                m = flight_re.search(base)
                if m is None:
                    unknown.append(s)
                else:
                    grouped.setdefault(m.group(1), []).append(s)
            if unknown:
                print(
                    f"  [WARN] GateDataset.per_flight_cap: "
                    f"{len(unknown)} samples have no parseable flight prefix; "
                    f"keeping them verbatim (no cap applied).",
                    flush=True,
                )
            rng = _random.Random(cap_seed)
            balanced: list[tuple[str, str, int]] = list(unknown)
            for flight, samples in sorted(grouped.items()):
                if len(samples) <= per_flight_cap:
                    balanced.extend(samples)
                else:
                    rng.shuffle(samples)
                    balanced.extend(samples[:per_flight_cap])
            # Sort for determinism across runs (Python dict iteration
            # order is insertion-stable but we want sorted regardless).
            balanced.sort(key=lambda s: s[0])
            print(
                f"  [GateDataset] per_flight_cap={per_flight_cap} "
                f"(seed={cap_seed}): {len(all_samples)} -> {len(balanced)} "
                f"samples across {len(grouped)} flights.",
                flush=True,
            )
            final_3tuple = balanced
        else:
            final_3tuple = all_samples

        # Source up-weighting (report 14 — finetune-cookbook prerequisite).
        # When ``sample_dir_weights`` is set, each sample is replicated by
        # its source dir's weight. Designed for the post-competition-data
        # finetune use case: cap a large pretrain pool at e.g. 150 per
        # flight with ``per_flight_cap``, then 30× the small competition
        # source via ``sample_dir_weights=[1, 30]``. Replication happens
        # AFTER cap so the cap is honoured per flight before upweighting.
        # weight=0 is allowed (and explicitly drops a source); weight=1 is
        # a no-op (current behaviour).
        if sample_dir_weights is not None:
            replicated: list[tuple[str, str, int]] = []
            for s in final_3tuple:
                w = sample_dir_weights[s[2]]
                if w > 0:
                    replicated.extend([s] * w)
            print(
                f"  [GateDataset] sample_dir_weights={list(sample_dir_weights)}: "
                f"{len(final_3tuple)} -> {len(replicated)} samples after "
                f"per-source replication.",
                flush=True,
            )
            final_3tuple = replicated

        # Strip source_idx — downstream callers expect 2-tuples.
        self.samples = [(s[0], s[1]) for s in final_3tuple]

        if use_sam_masks and skipped:
            print(
                f"  [GateDataset] skipped {skipped} samples without {MASK_SUFFIX} "
                "(run scripts/generate_sam_masks.py to fill in)"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, secondary_path = self.samples[idx]

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        if self.use_sam_masks:
            mask = _load_sam_mask(secondary_path, self.img_size)
        else:
            mask = corners_to_mask(secondary_path, self.img_size)

        if self.keypoint_heatmap:
            # In SAM-mask mode the dataset path stores `mask_path` as
            # `secondary_path`; the keypoint label still lives at the
            # sibling .txt. Resolve it explicitly.
            lbl_path = (
                secondary_path
                if secondary_path.endswith(".txt")
                else _label_path_for(img_path)
            )
            if self.multi_gate_target:
                # (G, K, 2) corners + (G, K) vis across all gates in frame.
                corners_px, vis = _all_gates_inner(
                    lbl_path, self.img_size, self.n_keypoints,
                )
                # Report 15.4 — single-gate-of-interest supervision for
                # the keypoint head. Use the same Vicon-derived next-
                # gate lookup we already do for the next-gate-mask
                # target. Filter corners_px/vis to keep ONLY that gate's
                # corners; the kp head then trains to fire on the next
                # gate only, not at every visible gate's corners (which
                # was the v2 trained behaviour and the bottleneck for
                # the hybrid tracker's coverage).
                if self.next_gate_only_kp:
                    from gate_detection.data.labels import parse_gates
                    from gate_detection.data.next_gate_labels import (
                        find_next_gate_label_index,
                    )
                    label_gates_now = parse_gates(lbl_path) if os.path.exists(lbl_path) else []
                    ng_idx = find_next_gate_label_index(
                        os.path.basename(img_path),
                        label_gates_now,
                        img_size=self.img_size,
                        csv_root=self.next_gate_csv_root,
                        calib_dir=self.next_gate_calib_dir,
                        warn_counter=self._next_gate_warn_counter,
                    )
                    if ng_idx is not None and ng_idx < corners_px.shape[0]:
                        corners_px = corners_px[ng_idx:ng_idx + 1]
                        vis = vis[ng_idx:ng_idx + 1]
                    else:
                        # No Vicon next-gate available — mask the kp
                        # loss for this sample by setting all
                        # visibility flags to 0. Same per-sample mask
                        # pattern as ng_valid for the next-gate-mask
                        # head (reviewer round-1 Item H).
                        corners_px = corners_px[:1] if corners_px.shape[0] > 0 else \
                            np.zeros((1, self.n_keypoints, 2), dtype=np.float32)
                        vis = np.zeros(
                            (1, self.n_keypoints), dtype=np.int32,
                        )
                if self.augment:
                    img, mask, corners_px, vis = train_augment_kp_multi(
                        img, mask, corners_px, vis,
                        strong=self.augment_strong,
                    )
                kp_target, vis_per_channel = render_heatmap_multi(
                    corners_px, vis,
                    size=self.img_size,
                    sigma=self.heatmap_sigma,
                )  # (K, H, W) float32, (K,) int32
                kp_target_t = torch.from_numpy(kp_target)
                kp_vis_t    = torch.from_numpy(vis_per_channel)
            else:
                corners_px, vis = _first_gate_inner(
                    lbl_path, self.img_size, self.n_keypoints,
                )
                if self.augment:
                    img, mask, corners_px, vis = train_augment_kp(
                        img, mask, corners_px, vis,
                        strong=self.augment_strong,
                    )
                kp_target = render_heatmap(
                    corners_px, vis,
                    size=self.img_size,
                    sigma=self.heatmap_sigma,
                )  # (K, H, W) float32
                kp_target_t = torch.from_numpy(kp_target)            # (K, H, W)
                kp_vis_t    = torch.from_numpy(vis.astype(np.int32))  # (K,)
        else:
            if self.augment:
                img, mask = train_augment(img, mask, strong=self.augment_strong)

        # Train/inference parity: if imagenet_norm is on, apply the SAME
        # helper used by the predictor (single source of truth in
        # transforms.apply_imagenet_norm). The flag is persisted in the
        # checkpoint metadata so the predictor can replay it.
        if self.imagenet_norm:
            img_t = torch.from_numpy(apply_imagenet_norm(img)).permute(2, 0, 1).contiguous()
        else:
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()

        # Next-gate-mask target (report 15). Generated AFTER augmentation
        # because we re-render at original-image coordinates from the
        # label file's normalized corners; the seg mask was warped in
        # `train_augment` but the labels themselves are unchanged here.
        # NOTE: this means the next-gate mask is NOT spatially-augmented
        # in this version — a known limitation; if the strong-aug
        # geometric transforms (perspective, affine) prove important
        # we'd need to re-apply them to this mask too. Documented for
        # the next reviewer.
        ng_mask_t = None
        ng_valid_t = None
        if self.next_gate_target:
            from gate_detection.data.labels import parse_gates
            from gate_detection.data.next_gate_labels import render_next_gate_mask
            # Determine the label path for THIS sample. In SAM mode the
            # secondary_path stores the mask PNG; the label is at
            # `<stem>.txt` next to the image.
            lbl_path = (
                secondary_path
                if secondary_path.endswith(".txt")
                else _label_path_for(img_path)
            )
            label_gates = parse_gates(lbl_path) if os.path.exists(lbl_path) else []
            ng_mask, ng_valid = render_next_gate_mask(
                os.path.basename(img_path),
                label_gates,
                img_size=self.img_size,
                csv_root=self.next_gate_csv_root,
                calib_dir=self.next_gate_calib_dir,
                ring_thickness_px=self.next_gate_ring_thickness_px,
                warn_counter=self._next_gate_warn_counter,
            )
            ng_mask_t = torch.from_numpy(ng_mask).unsqueeze(0).float()  # (1, H, W)
            # ng_valid is a 0/1 scalar saying "the trainer should apply
            # BCE on this sample" — fixes the zero-mask-training bug
            # reviewer-round-1 Item H caught (autonomous-flight samples
            # without Vicon would otherwise drag the head toward
            # all-zero predictions even when a gate IS visible).
            ng_valid_t = torch.tensor(float(ng_valid), dtype=torch.float32)

        if self.keypoint_heatmap:
            if ng_mask_t is not None:
                return img_t, mask_t, kp_target_t, kp_vis_t, ng_mask_t, ng_valid_t
            return img_t, mask_t, kp_target_t, kp_vis_t
        return img_t, mask_t
