"""Loss functions for CA world models."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from src.ca.observables import density, domain_wall_density_1d
from src.training.closure import build_closure_steps, latent_closure_loss, latent_rollout_steps
from src.utils.metrics import hamming_distance, probs_to_binary


def reconstruction_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets)


def observable_loss(logits: torch.Tensor, targets: torch.Tensor, use_domain_walls: bool = True) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    loss = F.mse_loss(density(probs), density(targets))
    if use_domain_walls and probs.ndim == 3:
        loss = loss + F.mse_loss(domain_wall_density_1d(probs), domain_wall_density_1d(targets))
    return loss


def latent_regularization(latent: torch.Tensor, weight: float) -> torch.Tensor:
    if weight <= 0.0:
        return torch.zeros((), device=latent.device)
    return weight * latent.pow(2).mean()


def slot_usage_penalty(attention: torch.Tensor, weight: float) -> torch.Tensor:
    if attention is None or weight <= 0.0:
        return torch.zeros((), device=attention.device if attention is not None else "cpu")
    probs = attention.mean(dim=-1)
    entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
    return -weight * entropy


def rollout_step_weights(horizon: int, gamma: float = 1.0) -> torch.Tensor:
    indices = torch.arange(horizon).float()
    weights = gamma**indices
    return weights / weights.sum()


def resolve_scheduled_value(base_value: float, schedule: list[dict[str, Any]] | None, epoch: int | None) -> float:
    value = float(base_value)
    if epoch is None or not schedule:
        return value
    for entry in sorted(schedule, key=lambda item: int(item.get("start_epoch", 1))):
        if epoch >= int(entry.get("start_epoch", 1)):
            value = float(entry["value"])
    return value


def resolve_rollout_horizon(config: dict[str, Any], targets: torch.Tensor, epoch: int | None) -> int:
    return min(
        targets.shape[1],
        int(
            resolve_scheduled_value(
                float(config["train"].get("rollout_horizon", 1)),
                config["train"].get("rollout_horizon_schedule"),
                epoch,
            )
        ),
    )


def compute_training_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    epoch: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute task-specific losses from a sequence window."""
    sequence = batch["window"].to(device)
    x0 = sequence[:, 0]
    targets = sequence[:, 1:]
    loss_cfg = config.get("losses", {})
    rollout_horizon = resolve_rollout_horizon(config, targets, epoch)
    weights = rollout_step_weights(rollout_horizon, gamma=float(loss_cfg.get("rollout_gamma", 1.0))).to(device)

    if config["model"]["name"] == "pixel_predictor":
        current = x0
        step_losses = []
        for t in range(rollout_horizon):
            logits = model(current)
            target = targets[:, t]
            step_losses.append(reconstruction_bce(logits, target))
            current = target
        loss = torch.stack(step_losses).dot(weights)
        metrics = {"loss_total": float(loss.item()), "loss_bce": float(loss.item())}
        return loss, metrics

    if config["model"]["name"] == "dense_world_model":
        outputs = model(x0)
        loss = reconstruction_bce(outputs.reconstruction_logits, x0) * float(loss_cfg.get("recon_weight", 0.2))
        pred_loss = reconstruction_bce(outputs.prediction_logits, targets[:, 0])
        loss = loss + pred_loss
        closure_weight = resolve_scheduled_value(
            float(loss_cfg.get("closure_weight", 0.0)),
            loss_cfg.get("closure_weight_schedule"),
            epoch,
        )
        obs_loss = observable_loss(outputs.prediction_logits, targets[:, 0], use_domain_walls=x0.ndim == 3)
        loss = loss + float(loss_cfg.get("observable_weight", 0.0)) * obs_loss
        loss = loss + latent_regularization(outputs.latent, float(loss_cfg.get("latent_l2_weight", 0.0)))
        metrics = {
            "loss_total": float(loss.item()),
            "loss_pred": float(pred_loss.item()),
            "loss_obs": float(obs_loss.item()),
        }
        latent_steps = latent_rollout_steps(model, x0, horizon=rollout_horizon, threshold=False)
        teacher_forced_latent_error = F.mse_loss(outputs.next_latent, model.encode(targets[:, 0]).detach())
        metrics["latent_teacher_forced_error"] = float(teacher_forced_latent_error.item())
        if closure_weight > 0.0:
            closure_steps = build_closure_steps(
                model=model,
                x_t=x0,
                future_states=targets[:, :rollout_horizon],
                horizon=rollout_horizon,
                detach_target=True,
            )
            rollout_closure = torch.zeros((), device=device)
            for t, step in enumerate(closure_steps):
                rollout_closure = rollout_closure + weights[t] * latent_closure_loss(
                    step.z_pred_next,
                    step.z_true_next_for_loss,
                )
            loss = loss + closure_weight * rollout_closure
            metrics["loss_closure"] = float(rollout_closure.item())
            metrics["loss_total"] = float(loss.item())
        latent_rollout_weight = float(loss_cfg.get("latent_rollout_weight", loss_cfg.get("rollout_weight", 1.0)))
        rollout_loss = torch.zeros((), device=device)
        latent_cycle_loss = torch.zeros((), device=device)
        latent_step_penalty = torch.zeros((), device=device)
        final_hamming = torch.zeros((), device=device)
        for t, step in enumerate(latent_steps):
            rollout_loss = rollout_loss + weights[t] * reconstruction_bce(step.logits, targets[:, t])
            reencoded = model.encode(torch.sigmoid(model.decode(step.latent)))
            latent_cycle_loss = latent_cycle_loss + weights[t] * F.mse_loss(step.latent, reencoded)
            step_delta = step.latent - step.prev_latent
            latent_step_penalty = latent_step_penalty + weights[t] * step_delta.pow(2).mean()
            final_hamming = hamming_distance(probs_to_binary(step.state), targets[:, t]).mean()
            metrics[f"latent_rollout_hamming_h{t + 1}"] = float(final_hamming.item())
        if rollout_horizon > 0:
            loss = loss + latent_rollout_weight * rollout_loss
            metrics["loss_rollout"] = float(rollout_loss.item())
            metrics["pure_latent_rollout_hamming"] = float(final_hamming.item())
        latent_cycle_weight = float(loss_cfg.get("latent_cycle_weight", 0.0))
        if latent_cycle_weight > 0.0 and rollout_horizon > 0:
            loss = loss + latent_cycle_weight * latent_cycle_loss
            metrics["loss_latent_cycle"] = float(latent_cycle_loss.item())
        latent_step_weight = float(loss_cfg.get("latent_step_weight", 0.0))
        if latent_step_weight > 0.0 and rollout_horizon > 0:
            loss = loss + latent_step_weight * latent_step_penalty
            metrics["loss_latent_step"] = float(latent_step_penalty.item())
        if rollout_horizon > 0:
            metrics["latent_step_norm"] = float(latent_step_penalty.sqrt().item())
        metrics["loss_total"] = float(loss.item())
        return loss, metrics

    if config["model"]["name"] == "object_world_model":
        outputs = model(x0)
        pred_loss = reconstruction_bce(outputs.prediction_logits, targets[:, 0])
        recon_loss = reconstruction_bce(outputs.reconstruction_logits, x0)
        obs_loss = observable_loss(outputs.prediction_logits, targets[:, 0], use_domain_walls=True)
        loss = pred_loss + float(loss_cfg.get("recon_weight", 0.2)) * recon_loss
        loss = loss + float(loss_cfg.get("observable_weight", 0.0)) * obs_loss
        loss = loss + slot_usage_penalty(outputs.attention, float(loss_cfg.get("slot_entropy_weight", 0.0)))
        return loss, {
            "loss_total": float(loss.item()),
            "loss_pred": float(pred_loss.item()),
            "loss_obs": float(obs_loss.item()),
        }

    raise ValueError(f"Unsupported model name: {config['model']['name']}")
