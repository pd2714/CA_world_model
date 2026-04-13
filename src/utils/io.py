"""Filesystem and serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_npz(path: str | Path, **arrays: Any) -> None:
    np.savez_compressed(Path(path), **arrays)


def torch_save(obj: Any, path: str | Path) -> None:
    torch.save(obj, Path(path))
