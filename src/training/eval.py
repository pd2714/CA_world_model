"""Evaluation routines for models and exact CA rollouts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ca.elementary_1d import ElementaryCA
from src.ca.observables import density, domain_wall_density_1d
from src.training.rollout import model_rollout
from src.utils.metrics import (
    binary_accuracy_from_logits,
    count_drift,
    density_drift,
    hamming_distance,
    probs_to_binary,
    shift_aligned_hamming_distance,
)


@torch.no_grad()
def evaluate_one_step(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_name: str,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    accs: list[float] = []
    criterion = torch.nn.BCEWithLogitsLoss()
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        if model_name == "pixel_predictor":
            logits = model(x)
        else:
            outputs = model(x)
            logits = outputs.prediction_logits
        losses.append(float(criterion(logits, y).item()))
        accs.append(float(binary_accuracy_from_logits(logits, y).item()))
    return {"one_step_bce": float(np.mean(losses)), "one_step_acc": float(np.mean(accs))}


@torch.no_grad()
def evaluate_exact_local_rule(
    model: torch.nn.Module,
    device: torch.device,
    model_name: str,
    rule: int,
    input_size: int,
) -> pd.DataFrame:
    model.eval()
    automaton = ElementaryCA(rule=rule)
    records: list[dict[str, float | int | str]] = []
    center = input_size // 2
    neighborhoods = ["111", "110", "101", "100", "011", "010", "001", "000"]

    for neighborhood in neighborhoods:
        x = torch.zeros(1, 1, input_size, device=device)
        x[0, 0, center - 1 : center + 2] = torch.tensor([int(bit) for bit in neighborhood], device=device)
        target = int(automaton.step(np.array([int(bit) for bit in neighborhood], dtype=np.uint8))[1])
        if model_name == "pixel_predictor":
            logits = model(x)
        else:
            outputs = model(x)
            logits = outputs.prediction_logits
        center_logit = float(logits[0, 0, center].item())
        center_prob = float(torch.sigmoid(logits[0, 0, center]).item())
        center_pred = int(center_prob >= 0.5)
        records.append(
            {
                "neighborhood": neighborhood,
                "target": target,
                "pred": center_pred,
                "correct": int(center_pred == target),
                "prob_1": center_prob,
                "logit": center_logit,
            }
        )
    return pd.DataFrame(records)


@torch.no_grad()
def rollout_with_optional_logits(
    model: torch.nn.Module,
    x0: torch.Tensor,
    steps: int,
    model_name: str,
    feedback_mode: str = "hard",
    rollout_mode: str = "latent",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    model.eval()
    if model_name == "dense_world_model":
        threshold = feedback_mode == "hard"
        states, logits = model.rollout(x0, steps=steps, threshold=threshold, rollout_mode=rollout_mode)
        return states, logits
    if model_name == "pixel_predictor":
        states = [x0]
        logits_seq = [torch.full_like(x0, float("nan"))]
        current = x0
        for _ in range(steps):
            logits = model(current)
            current = torch.sigmoid(logits) if feedback_mode == "soft" else (torch.sigmoid(logits) >= 0.5).float()
            logits_seq.append(logits)
            states.append(current)
        return torch.stack(states, dim=1), torch.stack(logits_seq, dim=1)
    rollout = model_rollout(model, x0, steps=steps, feedback_mode=feedback_mode, rollout_mode=rollout_mode)
    return rollout["states"], rollout.get("logits")


@torch.no_grad()
def evaluate_rollout(
    model: torch.nn.Module,
    trajectories: torch.Tensor,
    device: torch.device,
    horizon: int,
    model_name: str,
    feedback_mode: str = "hard",
    rollout_mode: str = "latent",
) -> pd.DataFrame:
    model.eval()
    records: list[dict[str, float]] = []
    for idx in tqdm(range(len(trajectories)), desc="rollout-eval", leave=False):
        seq = trajectories[idx : idx + 1].to(device)
        effective_horizon = min(horizon, seq.shape[1] - 1)
        x0 = seq[:, 0]
        target = seq[:, : effective_horizon + 1]
        pred, rollout_logits = rollout_with_optional_logits(
            model,
            x0,
            steps=effective_horizon,
            model_name=model_name,
            feedback_mode=feedback_mode,
            rollout_mode=rollout_mode,
        )
        pred = pred[:, : effective_horizon + 1]
        for t in range(1, pred.shape[1]):
            target_t = target[:, t]
            pred_t = pred[:, t]
            pred_binary = probs_to_binary(pred_t) if pred_t.dtype.is_floating_point else pred_t.float()
            aligned_hamming, best_shift = shift_aligned_hamming_distance(
                pred_binary,
                target_t,
            )
            record = {
                "sample": idx,
                "horizon": t,
                "hamming": float(hamming_distance(pred_binary, target_t).mean().item()),
                "density_error": float((density(pred_t) - density(target_t)).abs().mean().item()),
                "count_drift": float(count_drift(pred_binary, target_t).mean().item()),
                "density_drift": float(density_drift(pred_t, target_t).mean().item()),
                "shift_aligned_hamming": float(aligned_hamming.mean().item()),
                "best_shift": float(best_shift.float().mean().item()),
                "domain_wall_error": float(
                    (domain_wall_density_1d(pred_binary) - domain_wall_density_1d(target_t)).abs().mean().item()
                )
                if pred_t.ndim == 3
                else 0.0,
                "feedback_mode": feedback_mode,
                "rollout_mode": rollout_mode,
            }
            if rollout_logits is not None:
                record["rollout_bce"] = float(F.binary_cross_entropy_with_logits(rollout_logits[:, t], target_t).item())
            records.append(
                record
            )
    return pd.DataFrame(records)


def save_eval_summary(
    metrics: dict[str, float],
    rollout_df: pd.DataFrame,
    output_dir: str | Path,
    local_rule_df: pd.DataFrame | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(output_dir / "one_step_metrics.csv", index=False)
    rollout_df.to_csv(output_dir / "rollout_metrics.csv", index=False)
    if local_rule_df is not None:
        local_rule_df.to_csv(output_dir / "local_rule_metrics.csv", index=False)
