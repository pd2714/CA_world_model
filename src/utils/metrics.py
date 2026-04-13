"""Core numeric metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def bce_with_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets)


def binary_accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return (preds == targets).float().mean()


def probs_to_binary(probs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (probs >= threshold).float()


def hamming_distance(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return (preds != targets).float().mean(dim=tuple(range(1, preds.ndim)))


def shift_aligned_hamming_distance(preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    length = preds.shape[-1]
    best = None
    best_shift = None
    for shift in range(length):
        shifted = torch.roll(preds, shifts=shift, dims=-1)
        current = hamming_distance(shifted, targets)
        if best is None:
            best = current
            best_shift = torch.full_like(current, shift, dtype=torch.long)
            continue
        mask = current < best
        best = torch.where(mask, current, best)
        best_shift = torch.where(mask, torch.full_like(best_shift, shift), best_shift)
    if best is None or best_shift is None:
        raise ValueError("shift_aligned_hamming_distance requires non-empty inputs")
    return best, best_shift


def count_ones(state: torch.Tensor) -> torch.Tensor:
    return state.float().sum(dim=tuple(range(1, state.ndim)))


def count_drift(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return count_ones(preds) - count_ones(targets)


def density_drift(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return preds.float().mean(dim=tuple(range(1, preds.ndim))) - targets.float().mean(dim=tuple(range(1, targets.ndim)))


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def to_numpy(value: torch.Tensor | np.ndarray | Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
