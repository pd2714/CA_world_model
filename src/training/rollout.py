"""Autonomous rollout helpers."""

from __future__ import annotations

from typing import Any

import torch


def feedback_mode_to_state(logits: torch.Tensor, feedback_mode: str = "hard") -> torch.Tensor:
    probs = torch.sigmoid(logits)
    if feedback_mode == "soft":
        return probs
    if feedback_mode == "hard":
        return (probs >= 0.5).float()
    raise ValueError(f"Unsupported feedback mode: {feedback_mode}")


def dense_rollout(
    model: torch.nn.Module,
    x0: torch.Tensor,
    steps: int,
    feedback_mode: str = "hard",
    rollout_mode: str = "latent",
) -> dict[str, Any]:
    """Roll out a dense model either in latent space or via decode/re-encode feedback."""
    states = [x0]
    z = model.encode(x0)
    logits_seq = [model.decode(z)]
    current_state = x0
    for _ in range(steps):
        if rollout_mode == "reencode":
            z = model.encode(current_state)
        elif rollout_mode != "latent":
            raise ValueError(f"Unsupported rollout mode: {rollout_mode}")
        z = model.step_latent(z)
        logits = model.decode(z)
        current_state = feedback_mode_to_state(logits, feedback_mode=feedback_mode)
        logits_seq.append(logits)
        states.append(current_state)
    return {
        "states": torch.stack(states, dim=1),
        "logits": torch.stack(logits_seq, dim=1),
        "extras": {"feedback_mode": feedback_mode, "rollout_mode": rollout_mode},
    }


@torch.no_grad()
def model_rollout(
    model: torch.nn.Module,
    x0: torch.Tensor,
    steps: int,
    feedback_mode: str = "hard",
    rollout_mode: str = "latent",
) -> dict[str, Any]:
    if hasattr(model, "encode") and hasattr(model, "decode") and hasattr(model, "step_latent"):
        return dense_rollout(model, x0, steps=steps, feedback_mode=feedback_mode, rollout_mode=rollout_mode)
    if hasattr(model, "rollout"):
        rollout = model.rollout(x0, steps=steps)
        if isinstance(rollout, tuple):
            if len(rollout) == 2 and isinstance(rollout[1], dict):
                return {"states": rollout[0], "extras": rollout[1]}
            return {"states": rollout[0], "logits": rollout[1]}
        return {"states": rollout}
    raise ValueError("Model does not implement rollout().")
