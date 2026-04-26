"""Project paths.

All paths are resolved relative to the repository root (the directory
that contains ``src/``, ``configs/``, ``data/``, ``checkpoints/``).
Override any of them via environment variables of the same name (e.g.
``DATA_ROOT=/mnt/disk/data``).
"""

from __future__ import annotations

import os
from pathlib import Path


_PKG_DIR  = Path(__file__).resolve().parent          # src/gate_detection
_SRC_DIR  = _PKG_DIR.parent                          # src
REPO_ROOT = _SRC_DIR.parent                          # repository root


def _env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default)))


DATA_ROOT        = _env("DATA_ROOT",        REPO_ROOT / "data")
CHECKPOINTS_ROOT = _env("CHECKPOINTS_ROOT", REPO_ROOT / "checkpoints")
CONFIGS_ROOT     = _env("CONFIGS_ROOT",     REPO_ROOT / "configs")

SYN_ROOT         = DATA_ROOT / "synthetic"           # train/, val/, test/, train_aug/, val_aug/
BG_ROOT          = DATA_ROOT / "raw" / "backgrounds"

DEFAULT_CONFIG   = CONFIGS_ROOT / "default.yaml"
