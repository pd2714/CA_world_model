"""Config composition utilities."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from src.utils.io import load_yaml


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries."""
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load an experiment config and merge referenced sub-configs."""
    cfg = load_yaml(path)
    root = Path(path).resolve().parents[2]
    merged: dict[str, Any] = {}
    for section in ("data", "model", "train"):
        section_value = cfg.get(section)
        if isinstance(section_value, str):
            merged = deep_update(merged, load_yaml(root / section_value))
        elif isinstance(section_value, dict):
            merged = deep_update(merged, section_value)
    merged = deep_update(merged, {k: v for k, v in cfg.items() if k not in {"data", "model", "train"}})
    merged["config_path"] = str(Path(path).resolve())
    return merged
