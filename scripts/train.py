"""Train GateNet.

    python scripts/train.py
    python scripts/train.py --epochs 50 --batch-size 8 --no-aug
    python scripts/train.py --config configs/my_experiment.yaml
"""

from __future__ import annotations

import argparse

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
    args = p.parse_args()

    cfg = load_config(args.config)
    sched = cfg["train"]["scheduler"]
    perf = cfg["train"].get("performance", {})
    use_sam_masks = cfg["data"].get("use_sam_masks", True)
    data_root = str(SAM_ROOT if use_sam_masks else SYN_ROOT)

    train(
        epochs=             args.epochs     if args.epochs is not None     else cfg["train"]["epochs"],
        batch_size=         args.batch_size if args.batch_size is not None else cfg["train"]["batch_size"],
        base_lr=            args.lr         if args.lr is not None         else cfg["train"]["lr"],
        augment=            not args.no_aug,
        img_size=           cfg["data"]["img_size"],
        use_sam_masks=      use_sam_masks,
        imagenet_norm=      cfg["data"].get("imagenet_norm", False),
        data_root=          data_root,
        model_name=         cfg["model"].get("name", "gatenet"),
        width_factor=       cfg["model"]["width_factor"],
        encoder_name=       cfg["model"].get("encoder_name", "resnet34"),
        encoder_weights=    cfg["model"].get("encoder_weights", None),
        dropout_p=          cfg["model"].get("dropout_p", 0.0),
        optimizer_name=     cfg["train"].get("optimizer", "adamw"),
        sgd_momentum=       cfg["train"].get("sgd_momentum", 0.9),
        weight_decay=       cfg["train"].get("weight_decay", 1e-4),
        deep_supervision_weights=tuple(cfg["loss"]["deep_supervision_weights"]),
        bce_weight=         cfg["loss"]["bce_weight"],
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
