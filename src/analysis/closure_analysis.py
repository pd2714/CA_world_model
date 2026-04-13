"""Approximate closure analysis for latent dynamics."""

from __future__ import annotations

import numpy as np
import torch

from src.training.closure import build_closure_steps, closure_debug_report


@torch.no_grad()
def latent_closure_error(model: torch.nn.Module, trajectories: torch.Tensor, max_tau: int = 10) -> list[dict[str, float]]:
    """Compare rolled-out latent states to re-encoded future latents."""
    if not hasattr(model, "encode") or not hasattr(model, "step_latent"):
        return []
    records: list[dict[str, float]] = []
    max_horizon = min(max_tau, trajectories.shape[1] - 1)
    for tau in range(1, max_horizon + 1):
        step_errors = []
        rel_errors = []
        for t in range(trajectories.shape[1] - tau):
            steps = build_closure_steps(
                model=model,
                x_t=trajectories[:, t],
                future_states=trajectories[:, t + 1 : t + tau + 1],
                horizon=tau,
                detach_target=True,
            )
            if not steps:
                continue
            final_step = steps[-1]
            error = (final_step.z_pred_next - final_step.z_true_next).pow(2).mean(dim=tuple(range(1, final_step.z_pred_next.ndim)))
            denom = final_step.z_true_next.pow(2).mean(dim=tuple(range(1, final_step.z_true_next.ndim))).clamp_min(1e-8)
            step_errors.append(error)
            rel_errors.append(error / denom)
        if step_errors:
            records.append(
                {
                    "tau": tau,
                    "closure_mse": float(torch.cat(step_errors).mean().item()),
                    "closure_rel_mse": float(torch.cat(rel_errors).mean().item()),
                }
            )
    return records


@torch.no_grad()
def closure_debug_snapshot(
    model: torch.nn.Module,
    x_t: torch.Tensor,
    x_next: torch.Tensor,
) -> list[dict[str, float | list[int] | bool]]:
    steps = build_closure_steps(model=model, x_t=x_t, future_states=x_next[:, None], horizon=1, detach_target=True)
    return closure_debug_report(steps)


def latent_predictivity_score(features: np.ndarray, targets: np.ndarray) -> float:
    coef = np.linalg.lstsq(features, targets, rcond=None)[0]
    preds = features @ coef
    return float(((preds - targets) ** 2).mean())
