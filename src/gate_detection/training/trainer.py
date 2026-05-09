"""GateNet training loop.

Decomposed into:

    build_dataloaders(...)       — train + val dataloaders from data/synthetic/
    train_epoch(...)             — one epoch of optimisation
    validate(...)                — one epoch of evaluation (no grad)
    train(...)                   — top-level driver, saves best checkpoint

Hyperparameter defaults mirror ``configs/default.yaml``; the CLI in
``scripts/train.py`` reads that YAML and overrides via flags.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryJaccardIndex
from tqdm import tqdm
from transformers import get_polynomial_decay_schedule_with_warmup

from gate_detection.config import get_device
from gate_detection.data.dataset import GateDataset
from gate_detection.losses.keypoint import soft_argmax_2d, weighted_heatmap_mse
from gate_detection.losses.segmentation import gate_loss
from gate_detection.models import ModelConfig, SegmentationModelWrapper, build_model
from gate_detection.paths import CHECKPOINTS_ROOT, SAM_ROOT, SYN_ROOT


def _base_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying model when wrapped by torch.compile.

    ``torch.compile`` wraps the original module in an ``OptimizedModule``
    whose state-dict keys are prefixed with ``_orig_mod.``. We unwrap before
    saving so checkpoints stay portable to non-compiled inference (and the
    predictor doesn't have to deal with the prefix).
    """
    return getattr(model, "_orig_mod", model)


# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------
def build_dataloaders(
    data_root: str | os.PathLike = SAM_ROOT,
    *,
    img_size: int = 384,
    batch_size: int = 16,
    augment: bool = True,
    use_sam_masks: bool = True,
    imagenet_norm: bool = False,
    keypoint_heatmap: bool = False,
    heatmap_sigma: float = 3.0,
    n_keypoints: int = 4,
    num_workers_train: int = 12,
    num_workers_val: int = 12,
    prefetch_factor: int = 4,
) -> tuple[DataLoader, DataLoader, GateDataset, GateDataset]:
    """Build train+val loaders from ``<data_root>/{train,val}``."""
    data_root = Path(data_root)

    train_dirs = [str(data_root / "train")]
    val_dirs   = [str(data_root / "val")]

    train_ds = GateDataset(
        train_dirs, img_size=img_size, augment=augment,
        use_sam_masks=use_sam_masks, imagenet_norm=imagenet_norm,
        keypoint_heatmap=keypoint_heatmap, heatmap_sigma=heatmap_sigma,
        n_keypoints=n_keypoints,
    )
    val_ds = GateDataset(
        val_dirs, img_size=img_size, augment=False,
        use_sam_masks=use_sam_masks, imagenet_norm=imagenet_norm,
        keypoint_heatmap=keypoint_heatmap, heatmap_sigma=heatmap_sigma,
        n_keypoints=n_keypoints,
    )

    # persistent_workers avoids re-spawning the worker pool every epoch
    # (cheap dataset, expensive fork startup); prefetch_factor keeps the
    # GPU fed when augmentations dominate the per-batch CPU time.
    # Both are only valid when num_workers > 0.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers_train,
        pin_memory=True,
        persistent_workers=(num_workers_train > 0),
        prefetch_factor=(prefetch_factor if num_workers_train > 0 else None),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers_val,
        pin_memory=True,
        persistent_workers=(num_workers_val > 0),
        prefetch_factor=(prefetch_factor if num_workers_val > 0 else None),
    )
    return train_loader, val_loader, train_ds, val_ds


# ---------------------------------------------------------------------------
# Single-epoch driver
# ---------------------------------------------------------------------------
def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    *,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    deep_supervision_weights=(4, 2, 1, 1, 1),
    bce_weight: float = 2.0,
    iou_threshold: float = 0.5,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    channels_last: bool = False,
    keypoint_loss_weight: float = 0.0,
    keypoint_fg_weight: float = 10.0,
    desc: str = "train",
) -> tuple[float, float, float]:
    """One epoch of training.

    When ``keypoint_loss_weight > 0`` the loader is expected to yield 4-tuples
    ``(imgs, masks, kp_targets, kp_vis)`` and the combined loss
    ``L_seg + keypoint_loss_weight × L_heatmap`` is back-propagated. The
    third return value is the mean per-batch keypoint loss (0.0 in the
    seg-only branch) so the caller can log both components.
    """
    multi_task = keypoint_loss_weight > 0.0
    model.train()
    total_loss = 0.0
    total_kp_loss = 0.0
    n_samples = 0
    iou_metric = BinaryJaccardIndex(threshold=iou_threshold).to(device)
    progress = tqdm(loader, desc=desc, leave=False)
    for batch in progress:
        if multi_task:
            imgs, masks, kp_targets, kp_vis = batch
            kp_targets = kp_targets.to(device, non_blocking=True)
            kp_vis = kp_vis.to(device, non_blocking=True)
        else:
            imgs, masks = batch
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if channels_last and imgs.ndim == 4:
            imgs = imgs.contiguous(memory_format=torch.channels_last)
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            if multi_task:
                outputs, kp_pred = model.forward_multi(imgs)
                if kp_pred is None:
                    raise RuntimeError(
                        "keypoint_loss_weight > 0 but model has no keypoint "
                        "head — set model.keypoint_head=true in the config."
                    )
                seg_loss = gate_loss(
                    outputs, masks,
                    deep_supervision_weights=deep_supervision_weights,
                    bce_weight=bce_weight,
                )
                # Heatmap MSE runs in fp32 like BCE — bf16 has plenty of
                # range, but the upweighting near the peak amplifies any
                # rounding noise and the 0.x → 0.0 collapse breaks gradients.
                with torch.autocast(device_type=device, enabled=False):
                    kp_loss = weighted_heatmap_mse(
                        kp_pred.float(), kp_targets.float(), kp_vis,
                        fg_weight=keypoint_fg_weight,
                    )
                loss = seg_loss + keypoint_loss_weight * kp_loss
            else:
                outputs = model(imgs)
                loss = gate_loss(outputs, masks,
                                 deep_supervision_weights=deep_supervision_weights,
                                 bce_weight=bce_weight)
                kp_loss = torch.tensor(0.0, device=device)
        # GradScaler is only meaningful for fp16 (small dynamic range);
        # for bf16 the caller passes scaler.is_enabled() == False and we
        # take the plain backward path.
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        # Scheduler steps PER BATCH, not per epoch — required because the
        # polynomial-decay schedule is parametrised by total training steps.
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * imgs.size(0)
        total_kp_loss += float(kp_loss.item()) * imgs.size(0)
        n_samples += imgs.size(0)
        # Online IoU is computed on the full-resolution head output (last
        # entry of the deep-supervision tuple), to match deployment.
        pred_full = outputs[-1]
        iou_metric.update(pred_full, masks.to(torch.int64))
        progress.set_postfix(
            loss=f"{total_loss / max(n_samples, 1):.4f}",
            iou=f"{float(iou_metric.compute()):.4f}",
        )
    return (
        total_loss / max(n_samples, 1),
        float(iou_metric.compute()),
        total_kp_loss / max(n_samples, 1),
    )


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    *,
    deep_supervision_weights=(4, 2, 1, 1, 1),
    bce_weight: float = 2.0,
    iou_threshold: float = 0.5,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    channels_last: bool = False,
    keypoint_loss_weight: float = 0.0,
    keypoint_fg_weight: float = 10.0,
    soft_argmax_beta: float = 100.0,
    desc: str = "val",
) -> tuple[float, float, float, float]:
    """One validation epoch.

    Returns ``(val_loss, val_iou, val_kp_loss, val_corner_dist)`` where
    ``val_corner_dist`` is the mean Euclidean pixel distance between
    soft-argmax-decoded predicted corners and the soft-argmax of the GT
    heatmap, masked by visibility. Both keypoint quantities are 0.0 when
    ``keypoint_loss_weight == 0``.
    """
    multi_task = keypoint_loss_weight > 0.0
    model.eval()
    total_loss = 0.0
    total_kp_loss = 0.0
    total_corner_dist = 0.0
    n_visible_corners = 0
    n_samples = 0
    iou_metric = BinaryJaccardIndex(threshold=iou_threshold).to(device)
    progress = tqdm(loader, desc=desc, leave=False)
    for batch in progress:
        if multi_task:
            imgs, masks, kp_targets, kp_vis = batch
            kp_targets = kp_targets.to(device, non_blocking=True)
            kp_vis = kp_vis.to(device, non_blocking=True)
        else:
            imgs, masks = batch
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if channels_last and imgs.ndim == 4:
            imgs = imgs.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            if multi_task:
                outputs, kp_pred = model.forward_multi(imgs)
                seg_loss = gate_loss(
                    outputs, masks,
                    deep_supervision_weights=deep_supervision_weights,
                    bce_weight=bce_weight,
                )
                with torch.autocast(device_type=device, enabled=False):
                    kp_loss = weighted_heatmap_mse(
                        kp_pred.float(), kp_targets.float(), kp_vis,
                        fg_weight=keypoint_fg_weight,
                    )
                loss = seg_loss + keypoint_loss_weight * kp_loss
            else:
                outputs = model(imgs)
                loss = gate_loss(outputs, masks,
                                 deep_supervision_weights=deep_supervision_weights,
                                 bce_weight=bce_weight)
                kp_loss = torch.tensor(0.0, device=device)
                kp_pred = None
        total_loss += loss.item() * imgs.size(0)
        total_kp_loss += float(kp_loss.item()) * imgs.size(0)
        n_samples += imgs.size(0)
        pred_full = outputs[-1]
        iou_metric.update(pred_full, masks.to(torch.int64))

        if multi_task and kp_pred is not None:
            # Mean visibility-masked Euclidean distance between
            # soft-argmax(pred) and soft-argmax(target). The GT heatmap
            # is a clean σ=3 Gaussian so soft-argmax recovers the centre
            # to far-sub-pixel — the metric isolates pred error.
            pred_xy = soft_argmax_2d(kp_pred.float(), beta=soft_argmax_beta)   # (B, K, 2)
            tgt_xy  = soft_argmax_2d(kp_targets.float(), beta=soft_argmax_beta)
            d = (pred_xy - tgt_xy).pow(2).sum(dim=-1).sqrt()                   # (B, K)
            valid = kp_vis.float()                                              # (B, K)
            total_corner_dist += float((d * valid).sum().item())
            n_visible_corners += int(valid.sum().item())

        progress.set_postfix(
            loss=f"{total_loss / max(n_samples, 1):.4f}",
            iou=f"{float(iou_metric.compute()):.4f}",
        )
    val_corner_dist = (
        total_corner_dist / n_visible_corners
        if n_visible_corners > 0 else 0.0
    )
    return (
        total_loss / max(n_samples, 1),
        float(iou_metric.compute()),
        total_kp_loss / max(n_samples, 1),
        val_corner_dist,
    )


# ---------------------------------------------------------------------------
# Top-level training driver
# ---------------------------------------------------------------------------
def train(
    *,
    epochs: int = 100,
    batch_size: int = 16,
    base_lr: float = 1e-3,
    augment: bool = True,
    img_size: int = 384,
    model_name: str = "gatenet",
    width_factor: int = 4,
    encoder_name: str = "resnet34",
    encoder_weights: str | None = None,
    dropout_p: float = 0.0,
    optimizer_name: str = "adamw",
    sgd_momentum: float = 0.9,
    weight_decay: float = 1e-4,
    deep_supervision_weights=(4, 2, 1, 1, 1),
    bce_weight: float = 2.0,
    iou_threshold: float = 0.5,
    warmup_ratio: float = 0.1,
    polynomial_power: float = 1.0,
    polynomial_lr_end: float = 1e-6,
    compile_model: bool = True,
    compile_mode: str = "max-autotune",
    use_amp: bool = True,
    amp_dtype: str = "bf16",
    channels_last: bool = True,
    cudnn_benchmark: bool = True,
    matmul_precision: str = "high",
    num_workers_train: int = 4,
    num_workers_val: int = 2,
    prefetch_factor: int = 4,
    use_sam_masks: bool = True,
    imagenet_norm: bool = False,
    keypoint_head: bool = False,
    n_keypoints: int = 4,
    keypoint_loss_weight: float = 0.0,
    keypoint_fg_weight: float = 10.0,
    heatmap_sigma: float = 3.0,
    seed: int = 42,
    data_root: str | os.PathLike | None = None,
    checkpoint_dir: str | os.PathLike = CHECKPOINTS_ROOT,
    checkpoint_name: str = "best_gatenet.pth",
    checkpoint_interval: int = 0,
    device: str | None = None,
) -> torch.nn.Module:
    """Train GateNet end-to-end.

    Saves fixed-interval checkpoints during training and writes the single best
    validation-loss model once at the end.
    """
    device = device or get_device()
    print(f"Device: {device}")
    # Reproducibility: seed PyTorch (model init, dataloader shuffle), NumPy
    # (kornia internals, label rasterisation), and Python's random module
    # (augmentations in data/transforms.py). The 28/8 flight split is
    # independently seeded inside data/discovery.py:31 so it does not depend
    # on this call. cudnn deterministic is intentionally NOT set — too
    # expensive (~30% slower) for the run-to-run consistency it would buy.
    print(f"Seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device == "cuda":
        # cudnn.benchmark caches the best conv algorithm per input shape —
        # huge win when input shapes are static (always 384×384 here).
        torch.backends.cudnn.benchmark = cudnn_benchmark
        # "high" enables TF32 matmul on Ampere+ (3-4× speedup vs strict fp32
        # with negligible accuracy impact for segmentation).
        if matmul_precision in {"high", "medium"}:
            torch.set_float32_matmul_precision(matmul_precision)

    amp_dtype_map = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    resolved_amp_dtype = amp_dtype_map.get(amp_dtype.lower(), torch.bfloat16)
    use_amp = use_amp and device in {"cuda", "cpu"}
    # CPU autocast only supports bf16; silently coerce so stray fp16 configs
    # don't crash with an opaque "unsupported dtype" error.
    if device == "cpu" and resolved_amp_dtype != torch.bfloat16:
        resolved_amp_dtype = torch.bfloat16

    model_cfg = ModelConfig(
        name=model_name,
        width_factor=width_factor,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        dropout_p=dropout_p,
        keypoint_head=keypoint_head,
        n_keypoints=n_keypoints,
    )
    # Both directions are inconsistent at the dataloader / forward boundary
    # (dataset emits 4-tuples iff keypoint_head; train_epoch unpacks 4-tuples
    # iff keypoint_loss_weight > 0). Fail fast at config-time with a clear
    # message rather than crash on the first batch with a tuple-unpacking
    # error or a "model has no keypoint head" RuntimeError.
    if keypoint_head and keypoint_loss_weight <= 0.0:
        raise ValueError(
            "keypoint_head=True requires keypoint_loss_weight > 0 — an "
            "unsupervised keypoint head still emits 4-tuples from the "
            "dataloader, which the trainer's seg-only path does not unpack. "
            "Set loss.keypoint_weight (typical 1.0) in the config."
        )
    if (not keypoint_head) and keypoint_loss_weight > 0.0:
        raise ValueError(
            "keypoint_loss_weight > 0 requires model.keypoint_head=true; "
            "the trainer would call model.forward_multi() which returns "
            "kp_pred=None on a single-head model and crash with a clear "
            "RuntimeError on the first batch."
        )
    model: SegmentationModelWrapper = build_model(model_cfg).to(device)
    # Backbones that produce a single output (e.g. DeepLabV3+) carry
    # ds_weights=(1.0,); only GateNet uses real deep supervision and thus
    # honours the user-provided per-scale weights.
    train_ds_weights = tuple(model.ds_weights)
    if model_name.lower() == "gatenet":
        train_ds_weights = tuple(float(w) for w in deep_supervision_weights)
    # channels_last is a no-op for non-conv ops but ~30% faster for conv
    # nets on Ampere+ when paired with AMP. Both model AND inputs must be
    # in channels_last (see train_epoch / validate).
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    # torch.compile traces + JIT-compiles the forward graph. "max-autotune"
    # picks the fastest kernels per input shape (one-time cost, big payoff
    # given our static 384×384 inputs).
    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model, mode=compile_mode)

    resolved_data_root = data_root if data_root is not None else (SAM_ROOT if use_sam_masks else SYN_ROOT)
    train_loader, val_loader, train_ds, val_ds = build_dataloaders(
        resolved_data_root,
        img_size=img_size,
        batch_size=batch_size,
        augment=augment,
        use_sam_masks=use_sam_masks,
        imagenet_norm=imagenet_norm,
        keypoint_heatmap=keypoint_head,
        heatmap_sigma=heatmap_sigma,
        n_keypoints=n_keypoints,
        num_workers_train=num_workers_train,
        num_workers_val=num_workers_val,
        prefetch_factor=prefetch_factor,
    )
    print(f"Train: {len(train_ds)} samples  Val: {len(val_ds)} samples")
    print(f"Data : use_sam_masks={use_sam_masks}, imagenet_norm={imagenet_norm}, augment={augment}")
    if keypoint_head:
        print(
            f"Multi-task: keypoint_head=True n_keypoints={n_keypoints} "
            f"loss_weight={keypoint_loss_weight} fg_weight={keypoint_fg_weight} "
            f"sigma_px={heatmap_sigma}"
        )

    # GradScaler enabled ONLY for CUDA + fp16. bf16 has the same exponent
    # range as fp32, so gradients never underflow and scaling is unnecessary.
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device == "cuda" and use_amp and resolved_amp_dtype == torch.float16),
    )

    opt_name = optimizer_name.lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=base_lr,
            momentum=sgd_momentum,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Use 'adamw' or 'sgd'.")
    # Polynomial decay with linear warmup, parameterised in *steps* (not
    # epochs). Linear warmup from 0 → base_lr over the first warmup_ratio
    # of training, then polynomial decay base_lr → polynomial_lr_end with
    # the given power (1.0 = linear, <1.0 = decays faster early).
    steps_per_epoch = max(len(train_loader), 1)
    total_train_steps = max(epochs * steps_per_epoch, 1)
    warmup_steps = int(total_train_steps * warmup_ratio)
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
        lr_end=polynomial_lr_end,
        power=polynomial_power,
    )

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / checkpoint_name

    best_val_loss = float("inf")
    best_val_iou = float("-inf")
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        train_loss, train_iou, train_kp_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scheduler=scheduler,
            scaler=scaler,
            deep_supervision_weights=train_ds_weights,
            bce_weight=bce_weight,
            iou_threshold=iou_threshold,
            use_amp=use_amp,
            amp_dtype=resolved_amp_dtype,
            channels_last=channels_last,
            keypoint_loss_weight=keypoint_loss_weight,
            keypoint_fg_weight=keypoint_fg_weight,
            desc=f"Epoch {epoch:3d}/{epochs} [train]",
        )

        val_loss, val_iou, val_kp_loss, val_corner_dist = validate(
            model,
            val_loader,
            device,
            deep_supervision_weights=train_ds_weights,
            bce_weight=bce_weight,
            iou_threshold=iou_threshold,
            use_amp=use_amp,
            amp_dtype=resolved_amp_dtype,
            channels_last=channels_last,
            keypoint_loss_weight=keypoint_loss_weight,
            keypoint_fg_weight=keypoint_fg_weight,
            desc=f"Epoch {epoch:3d}/{epochs} [val]  ",
        )

        lr_now = float(optimizer.param_groups[0]["lr"])
        print(
            f"Epoch {epoch:3d}/{epochs}  lr={lr_now:.2e}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}",
        )
        print(f"  IoU: train={train_iou:.4f}  val={val_iou:.4f}")
        if keypoint_loss_weight > 0.0:
            print(
                f"  KP : train_kp_loss={train_kp_loss:.4f}  val_kp_loss={val_kp_loss:.4f}  "
                f"val_corner_dist={val_corner_dist:.2f} px"
            )

        # Primary selection metric: validation IoU.
        # Tie-breaker: lower validation loss.
        # We snapshot to CPU to avoid holding a duplicate state-dict on GPU
        # for the whole run; the checkpoint is written once at the end.
        if (val_iou > best_val_iou) or (val_iou == best_val_iou and val_loss < best_val_loss):
            best_val_iou = val_iou
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in _base_model(model).state_dict().items()
            }

        # Periodic snapshots are independent of best-model selection; they
        # exist for resume / inspection and include the optimizer state.
        if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
            interval_name = (
                f"{Path(checkpoint_name).stem}_epoch{epoch:03d}"
                f"{Path(checkpoint_name).suffix}"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model": _base_model(model).state_dict(),
                    # model_config must round-trip every architectural field
                    # — the predictor uses it to rebuild the exact same
                    # state-dict layout on load (notably dropout_p, which
                    # toggles the DeepLabV3+ head wrapper).
                    "model_config": {
                        "name": model_cfg.name,
                        "width_factor": model_cfg.width_factor,
                        "encoder_name": model_cfg.encoder_name,
                        "encoder_weights": model_cfg.encoder_weights,
                        "dropout_p": model_cfg.dropout_p,
                        "keypoint_head": model_cfg.keypoint_head,
                        "n_keypoints": model_cfg.n_keypoints,
                    },
                    # Data-preprocessing fields the predictor must replay
                    # at inference for parity. Kept separate from
                    # model_config because they are not architectural.
                    "data_config": {
                        "imagenet_norm": imagenet_norm,
                        "heatmap_sigma": heatmap_sigma,
                    },
                    "seed": seed,
                    "optimizer": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "train_iou": train_iou,
                    "val_iou": val_iou,
                },
                checkpoint_dir / interval_name,
            )
            print(f" checkpoint -> {checkpoint_dir / interval_name}")

    if best_state_dict is None:
        best_state_dict = {
            k: v.detach().cpu().clone()
            for k, v in _base_model(model).state_dict().items()
        }

    torch.save(
        {
            "epoch": best_epoch,
            "model": best_state_dict,
            "model_config": {
                "name": model_cfg.name,
                "width_factor": model_cfg.width_factor,
                "encoder_name": model_cfg.encoder_name,
                "encoder_weights": model_cfg.encoder_weights,
                "dropout_p": model_cfg.dropout_p,
                "keypoint_head": model_cfg.keypoint_head,
                "n_keypoints": model_cfg.n_keypoints,
            },
            "data_config": {
                "imagenet_norm": imagenet_norm,
                "heatmap_sigma": heatmap_sigma,
            },
            "seed": seed,
            "val_loss": best_val_loss,
            "val_iou": best_val_iou,
        },
        ckpt_path,
    )
    print(f" best_gatenet -> {ckpt_path} (epoch {best_epoch}, val_iou={best_val_iou:.4f})")

    print(f"\nDone. Best val IoU: {best_val_iou:.4f} (val_loss={best_val_loss:.4f})")
    return model
