"""Minimal YAML config loader.

Usage:
    from gate_detection.config import load_config
    cfg = load_config()                          # configs/default.yaml
    cfg = load_config("configs/experiment.yaml") # custom file
    epochs = cfg["train"]["epochs"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .paths import DEFAULT_CONFIG


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config file. Falls back to ``configs/default.yaml``."""
    p = Path(path) if path is not None else DEFAULT_CONFIG
    with open(p) as f:
        return yaml.safe_load(f)


def get_device() -> str:
    """Pick the best available torch device (cuda > mps > cpu)."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
