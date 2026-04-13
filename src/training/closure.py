"""Helpers for latent-closure training and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class ClosureStep:
    tau: int
    x_t: torch.Tensor
    x_true_next: torch.Tensor
    z_t: torch.Tensor
    z_pred_next: torch.Tensor
    z_true_next: torch.Tensor
    z_true_next_for_loss: torch.Tensor


@dataclass
class LatentRolloutStep:
    tau: int
    latent: torch.Tensor
    prev_latent: torch.Tensor
    logits: torch.Tensor
    state: torch.Tensor


def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    return {
        "shape": list(detached.shape),
        "mean": float(detached.mean().item()),
        "std": float(detached.std().item()) if detached.numel() > 1 else 0.0,
        "min": float(detached.min().item()),
        "max": float(detached.max().item()),
        "abs_mean": float(detached.abs().mean().item()),
    }


def build_closure_steps(
    model: torch.nn.Module,
    x_t: torch.Tensor,
    future_states: torch.Tensor,
    horizon: int,
    detach_target: bool = True,
) -> list[ClosureStep]:
    """Build closure supervision pairs z_pred_next vs z_true_next."""
    effective_horizon = min(int(horizon), int(future_states.shape[1]))
    steps: list[ClosureStep] = []
    z_t = model.encode(x_t)
    z_roll = z_t
    for tau in range(1, effective_horizon + 1):
        z_pred_next = model.step_latent(z_roll)
        x_true_next = future_states[:, tau - 1]
        z_true_next = model.encode(x_true_next)
        steps.append(
            ClosureStep(
                tau=tau,
                x_t=x_t,
                x_true_next=x_true_next,
                z_t=z_t,
                z_pred_next=z_pred_next,
                z_true_next=z_true_next,
                z_true_next_for_loss=z_true_next.detach() if detach_target else z_true_next,
            )
        )
        z_roll = z_pred_next
    return steps


def latent_closure_loss(predicted_latent: torch.Tensor, target_latent: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(predicted_latent, target_latent)


def closure_debug_report(steps: list[ClosureStep]) -> list[dict[str, Any]]:
    """Return shape/stat summaries that make the closure comparison explicit."""
    report: list[dict[str, Any]] = []
    for step in steps:
        report.append(
            {
                "tau": step.tau,
                "x_t": _tensor_stats(step.x_t),
                "x_true_next": _tensor_stats(step.x_true_next),
                "z_t": _tensor_stats(step.z_t),
                "z_pred_next": _tensor_stats(step.z_pred_next),
                "z_true_next": _tensor_stats(step.z_true_next),
                "target_detached": bool(not step.z_true_next_for_loss.requires_grad),
                "closure_mse": float(latent_closure_loss(step.z_pred_next, step.z_true_next_for_loss).detach().item()),
            }
        )
    return report


def latent_rollout_steps(
    model: torch.nn.Module,
    x_t: torch.Tensor,
    horizon: int,
    threshold: bool = False,
) -> list[LatentRolloutStep]:
    z = model.encode(x_t)
    steps: list[LatentRolloutStep] = []
    current = z
    for tau in range(1, int(horizon) + 1):
        prev = current
        current = model.step_latent(current)
        logits = model.decode(current)
        state = (torch.sigmoid(logits) >= 0.5).float() if threshold else torch.sigmoid(logits)
        steps.append(LatentRolloutStep(tau=tau, latent=current, prev_latent=prev, logits=logits, state=state))
    return steps
