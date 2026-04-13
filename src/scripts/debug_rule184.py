"""Focused rollout debugging for 1D CA world models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from src.ca.datasets import CADataset
from src.training.rollout import model_rollout
from src.utils.config import load_config
from src.utils.factory import build_model
from src.utils.metrics import hamming_distance
from src.utils.runtime import configure_runtime


def extract_prediction_logits(model: torch.nn.Module, x: torch.Tensor, model_name: str) -> torch.Tensor:
    if model_name == "pixel_predictor":
        return model(x)
    return model(x).prediction_logits


def shift_aligned_hamming(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    length = pred.shape[-1]
    best = None
    best_shift = None
    for shift in range(length):
        shifted = torch.roll(pred, shifts=shift, dims=-1)
        current = hamming_distance(shifted, target)
        if best is None:
            best = current
            best_shift = torch.full_like(current, shift, dtype=torch.long)
            continue
        mask = current < best
        best = torch.where(mask, current, best)
        best_shift = torch.where(mask, torch.full_like(best_shift, shift), best_shift)
    return best, best_shift


@torch.no_grad()
def rollout_debug_table(
    model: torch.nn.Module,
    trajectories: torch.Tensor,
    model_name: str,
    horizon: int,
) -> pd.DataFrame:
    records: list[dict[str, float]] = []
    for idx in range(len(trajectories)):
        seq = trajectories[idx : idx + 1]
        effective_horizon = min(horizon, seq.shape[1] - 1)
        x0 = seq[:, 0]
        target = seq[:, : effective_horizon + 1]

        hard_rollout = model.rollout(x0, steps=effective_horizon, threshold=True)
        soft_rollout = model.rollout(x0, steps=effective_horizon, threshold=False)
        if isinstance(hard_rollout, tuple):
            hard_states, _ = hard_rollout
        else:
            hard_states = hard_rollout
        if isinstance(soft_rollout, tuple):
            soft_states, soft_logits = soft_rollout
        else:
            soft_states = soft_rollout
            soft_logits = None

        initial_ones = float(x0.sum().item())
        for t in range(1, effective_horizon + 1):
            target_t = target[:, t]
            hard_t = hard_states[:, t]
            soft_t = soft_states[:, t]
            aligned_hamming, best_shift = shift_aligned_hamming(hard_t, target_t)
            record = {
                "sample": idx,
                "horizon": t,
                "hard_hamming": float(hamming_distance(hard_t, target_t).item()),
                "soft_hamming": float(hamming_distance((soft_t >= 0.5).float(), target_t).item()),
                "shift_aligned_hamming": float(aligned_hamming.item()),
                "best_shift": float(best_shift.item()),
                "target_ones": float(target_t.sum().item()),
                "hard_ones": float(hard_t.sum().item()),
                "soft_mass": float(soft_t.sum().item()),
                "hard_count_drift_from_target": float(hard_t.sum().item() - target_t.sum().item()),
                "hard_count_drift_from_initial": float(hard_t.sum().item() - initial_ones),
                "soft_mass_drift_from_target": float(soft_t.sum().item() - target_t.sum().item()),
                "soft_mass_drift_from_initial": float(soft_t.sum().item() - initial_ones),
            }
            if soft_logits is not None:
                record["soft_rollout_bce"] = float(F.binary_cross_entropy_with_logits(soft_logits[:, t], target_t).item())
            records.append(record)
    return pd.DataFrame(records)


@torch.no_grad()
def closure_table(model: torch.nn.Module, trajectories: torch.Tensor, max_tau: int) -> pd.DataFrame:
    if not hasattr(model, "encode") or not hasattr(model, "step_latent"):
        return pd.DataFrame()
    records: list[dict[str, float]] = []
    num_steps = trajectories.shape[1]
    for tau in range(1, min(max_tau + 1, num_steps)):
        mses = []
        rels = []
        for t in range(num_steps - tau):
            z = model.encode(trajectories[:, t])
            z_roll = z
            for _ in range(tau):
                z_roll = model.step_latent(z_roll)
            z_true = model.encode(trajectories[:, t + tau])
            err = (z_roll - z_true).pow(2).mean(dim=tuple(range(1, z_roll.ndim)))
            denom = z_true.pow(2).mean(dim=tuple(range(1, z_true.ndim))).clamp_min(1e-8)
            mses.append(err)
            rels.append(err / denom)
        mse_tau = torch.cat(mses).mean().item()
        rel_tau = torch.cat(rels).mean().item()
        records.append({"tau": tau, "closure_mse": float(mse_tau), "closure_rel_mse": float(rel_tau)})
    return pd.DataFrame(records)


@torch.no_grad()
def self_consistency_table(model: torch.nn.Module, trajectories: torch.Tensor, model_name: str) -> pd.DataFrame:
    if not hasattr(model, "encode") or not hasattr(model, "decode"):
        return pd.DataFrame()
    flat_states = trajectories[:, :-1].reshape(-1, *trajectories.shape[2:])
    z = model.encode(flat_states)
    recon_logits = model.decode(z)
    recon_probs = torch.sigmoid(recon_logits)
    recon_hard = (recon_probs >= 0.5).float()

    x_bce = F.binary_cross_entropy_with_logits(recon_logits, flat_states).item()
    x_acc = (recon_hard == flat_states).float().mean().item()

    z_reencoded_soft = model.encode(recon_probs)
    z_reencoded_hard = model.encode(recon_hard)
    z_mse_soft = (z_reencoded_soft - z).pow(2).mean().item()
    z_mse_hard = (z_reencoded_hard - z).pow(2).mean().item()

    return pd.DataFrame(
        [
            {
                "recon_bce": float(x_bce),
                "recon_acc": float(x_acc),
                "latent_cycle_mse_soft": float(z_mse_soft),
                "latent_cycle_mse_hard": float(z_mse_hard),
            }
        ]
    )


def summarize_tables(
    rollout_df: pd.DataFrame,
    closure_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
) -> dict[str, Any]:
    horizon_summary = rollout_df.groupby("horizon").mean(numeric_only=True)
    focus = [h for h in (1, 2, 4, 8, 16, 32) if h in horizon_summary.index]
    return {
        "hard_vs_soft_rollout": horizon_summary.loc[focus, ["hard_hamming", "soft_hamming", "shift_aligned_hamming"]]
        .reset_index()
        .to_dict(orient="records"),
        "count_drift": horizon_summary.loc[focus, ["hard_count_drift_from_initial", "soft_mass_drift_from_initial"]]
        .reset_index()
        .to_dict(orient="records"),
        "closure": closure_df.to_dict(orient="records"),
        "consistency": consistency_df.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--max_tau", type=int, default=8)
    parser.add_argument("--cpu_threads", type=int, default=2)
    args = parser.parse_args()

    configure_runtime(cpu_threads=args.cpu_threads, interop_threads=1)
    config = load_config(args.config)
    model = build_model(config).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset_prefix = config.get("dataset_name", config["experiment_name"])
    dataset = CADataset.from_npz(Path(args.data_dir) / f"{dataset_prefix}_test.npz")
    trajectories = dataset.trajectories.to(args.device)

    output_dir = Path(args.output_dir or Path(args.checkpoint).resolve().parent / "debug")
    output_dir.mkdir(parents=True, exist_ok=True)

    rollout_df = rollout_debug_table(model, trajectories, config["model"]["name"], horizon=args.horizon)
    closure_df = closure_table(model, trajectories[:, : min(16, trajectories.shape[1])], max_tau=args.max_tau)
    consistency_df = self_consistency_table(model, trajectories[:, : min(16, trajectories.shape[1])], config["model"]["name"])
    summary = summarize_tables(rollout_df, closure_df, consistency_df)

    rollout_df.to_csv(output_dir / "rollout_debug.csv", index=False)
    closure_df.to_csv(output_dir / "closure_debug.csv", index=False)
    consistency_df.to_csv(output_dir / "self_consistency.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(output_dir)


if __name__ == "__main__":
    main()
