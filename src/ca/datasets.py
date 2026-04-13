"""Dataset wrappers for CA trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def load_npz_dataset(path: str | Path) -> dict[str, Any]:
    data = np.load(Path(path), allow_pickle=True)
    return {key: data[key] for key in data.files}


@dataclass
class CADataset(Dataset):
    """Base dataset that wraps saved trajectories."""

    trajectories: torch.Tensor
    metadata: dict[str, Any]

    @classmethod
    def from_npz(cls, path: str | Path) -> "CADataset":
        raw = load_npz_dataset(path)
        metadata = raw.get("metadata", np.array({}, dtype=object)).item()
        return cls(trajectories=torch.tensor(raw["trajectories"]).float(), metadata=metadata)

    def __len__(self) -> int:
        return int(self.trajectories.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        traj = self.trajectories[idx]
        return {"trajectory": traj}


class NextStepDataset(CADataset):
    """Flatten trajectories into next-step pairs."""

    def __len__(self) -> int:
        return int(self.trajectories.shape[0] * (self.trajectories.shape[1] - 1))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        num_steps = self.trajectories.shape[1] - 1
        traj_idx = idx // num_steps
        step_idx = idx % num_steps
        current_state = self.trajectories[traj_idx, step_idx]
        next_state = self.trajectories[traj_idx, step_idx + 1]
        return {"x": current_state, "y": next_state, "traj_idx": traj_idx, "step_idx": step_idx}


class SequenceWindowDataset(CADataset):
    """Sample fixed-length sequence windows for rollout training."""

    def __init__(self, trajectories: torch.Tensor, metadata: dict[str, Any], window: int) -> None:
        super().__init__(trajectories=trajectories, metadata=metadata)
        self.window = window

    @classmethod
    def from_npz(cls, path: str | Path, window: int) -> "SequenceWindowDataset":
        base = CADataset.from_npz(path)
        return cls(base.trajectories, base.metadata, window=window)

    def __len__(self) -> int:
        return int(self.trajectories.shape[0] * (self.trajectories.shape[1] - self.window))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        max_offset = self.trajectories.shape[1] - self.window
        traj_idx = idx // max_offset
        start = idx % max_offset
        window = self.trajectories[traj_idx, start : start + self.window]
        return {"window": window, "traj_idx": traj_idx, "start": start}
