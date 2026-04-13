"""Observable extraction for CA states and trajectories."""

from __future__ import annotations

import numpy as np
import torch


def density(state: torch.Tensor) -> torch.Tensor:
    return state.float().mean(dim=tuple(range(1, state.ndim)))


def domain_walls_1d(state: torch.Tensor) -> torch.Tensor:
    right = torch.roll(state, shifts=-1, dims=-1)
    return (state != right).float()


def domain_wall_density_1d(state: torch.Tensor) -> torch.Tensor:
    return domain_walls_1d(state).mean(dim=tuple(range(1, state.ndim)))


def numpy_density(trajectory: np.ndarray) -> np.ndarray:
    axes = tuple(range(1, trajectory.ndim))
    return trajectory.mean(axis=axes)


def numpy_domain_wall_density_1d(trajectory: np.ndarray) -> np.ndarray:
    right = np.roll(trajectory, shift=-1, axis=-1)
    return (trajectory != right).mean(axis=tuple(range(1, trajectory.ndim)))


def simple_event_score_1d(trajectory: np.ndarray) -> np.ndarray:
    walls = np.roll(trajectory, -1, axis=-1) ^ trajectory.astype(np.uint8)
    temporal_change = np.abs(np.diff(walls.astype(np.float32), axis=0))
    return temporal_change.mean(axis=-1)
