"""Train GateNet.

    python scripts/train.py
    python scripts/train.py --epochs 50 --batch-size 8 --no-aug
    python scripts/train.py --config configs/my_experiment.yaml
"""

from __future__ import annotations

import argparse
import os

from gate_detection.config import load_config
from gate_detection.paths import DATA_ROOT, SAM_ROOT, SYN_ROOT
from gate_detection.training.trainer import train


def main() -> None:
    p = argparse.ArgumentParser(description="Train GateNet")
    p.add_argument("--config",     type=str, default=None, help="YAML config (default: configs/default.yaml)")
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--no-aug",     action="store_true",
                   help="disable online per-batch augmentation")
    p.add_argument("--seed",       type=int, default=None,
                   help="reproducibility seed (overrides cfg['train']['seed']; default 42)")
    args = p.parse_args()

    cfg = load_config(args.config)
    sched = cfg["train"]["scheduler"]
    perf = cfg["train"].get("performance", {})
    use_sam_masks = cfg["data"].get("use_sam_masks", True)
    data_root = str(SAM_ROOT if use_sam_masks else SYN_ROOT)
    loss_cfg = cfg["loss"]

    # Pre-flight validation for the finetune-cookbook config surface
    # (report 14, Tier 3.6). The dataset silently skips directories that
    # don't exist (GateDataset.__init__: ``if not os.path.isdir(d): continue``),
    # which means a stale ``TODO_PATH_...`` placeholder in the template
    # would silently produce a TII-only training run — the opposite of
    # what the user intended. Fail loudly here so the mistake is caught
    # at config-load time, not after a 6-hour run.
    extra_train_dirs = cfg["data"].get("extra_train_dirs") or []
    for d in extra_train_dirs:
        if "TODO" in str(d).upper():
            raise ValueError(
                f"data.extra_train_dirs contains an un-filled template "
                f"placeholder: {d!r}. Replace it with a real path or "
                f"remove the key entirely before training."
            )
        if not os.path.isdir(d):
            raise FileNotFoundError(
                f"data.extra_train_dirs[i]={d!r} does not exist; the "
                f"dataset would silently skip it. Fix the path."
            )
    init_from = cfg["train"].get("init_from")
    if init_from is not None and not os.path.isfile(init_from):
        raise FileNotFoundError(
            f"train.init_from={init_from!r} does not exist; cannot resume."
        )

    train(
        epochs=             args.epochs     if args.epochs is not None     else cfg["train"]["epochs"],
        batch_size=         args.batch_size if args.batch_size is not None else cfg["train"]["batch_size"],
        base_lr=            args.lr         if args.lr is not None         else cfg["train"]["lr"],
        augment=            not args.no_aug,
        augment_strong=     cfg["data"].get("augment_strong", False),
        seed=               args.seed if args.seed is not None else cfg["train"].get("seed", 42),
        img_size=           cfg["data"]["img_size"],
        use_sam_masks=      use_sam_masks,
        imagenet_norm=      cfg["data"].get("imagenet_norm", False),
        data_root=          data_root,
        model_name=         cfg["model"].get("name", "gatenet"),
        width_factor=       cfg["model"]["width_factor"],
        encoder_name=       cfg["model"].get("encoder_name", "resnet34"),
        encoder_weights=    cfg["model"].get("encoder_weights", None),
        dropout_p=          cfg["model"].get("dropout_p", 0.0),
        keypoint_head=      cfg["model"].get("keypoint_head", False),
        n_keypoints=        cfg["model"].get("n_keypoints", 4),
        keypoint_loss_weight=loss_cfg.get("keypoint_weight", 0.0),
        keypoint_fg_weight= loss_cfg.get("keypoint_fg_weight", 10.0),
        heatmap_sigma=      cfg["data"].get("heatmap_sigma", 3.0),
        multi_gate_target=  cfg["data"].get("multi_gate_target", False),
        per_flight_cap=     cfg["data"].get("per_flight_cap", None),
        cap_seed=           cfg["data"].get("cap_seed", 42),
        sample_dir_weights= cfg["data"].get("sample_dir_weights", None),
        extra_train_dirs=   cfg["data"].get("extra_train_dirs", None),
        init_from=          cfg["train"].get("init_from", None),
        freeze_encoder=     cfg["train"].get("freeze_encoder", False),
        next_gate_head=     cfg["model"].get("next_gate_head", False),
        next_gate_loss_weight=loss_cfg.get("next_gate_weight", 0.0),
        next_gate_ring_thickness_px=cfg["data"].get("next_gate_ring_thickness_px", 6),
        next_gate_only_kp=  cfg["data"].get("next_gate_only_kp", False),
        optimizer_name=     cfg["train"].get("optimizer", "adamw"),
        sgd_momentum=       cfg["train"].get("sgd_momentum", 0.9),
        weight_decay=       cfg["train"].get("weight_decay", 1e-4),
        deep_supervision_weights=tuple(loss_cfg["deep_supervision_weights"]),
        bce_weight=         loss_cfg["bce_weight"],
        iou_threshold=      cfg["eval"]["threshold"],
        warmup_ratio=       sched.get("warmup_ratio", 0.1),
        polynomial_power=   sched.get("power", 1.0),
        polynomial_lr_end=  sched.get("lr_end", 1e-6),
        num_workers_train=  cfg["train"]["num_workers_train"],
        num_workers_val=    cfg["train"]["num_workers_val"],
        prefetch_factor=    perf.get("prefetch_factor", 4),
        compile_model=      perf.get("compile", True),
        compile_mode=       perf.get("compile_mode", "max-autotune"),
        use_amp=            perf.get("amp", True),
        amp_dtype=          perf.get("amp_dtype", "bf16"),
        channels_last=      perf.get("channels_last", True),
        cudnn_benchmark=    perf.get("cudnn_benchmark", True),
        matmul_precision=   perf.get("matmul_precision", "high"),
        checkpoint_name=    cfg["train"]["checkpoint_name"],
    )


if __name__ == "__main__":
    main()
