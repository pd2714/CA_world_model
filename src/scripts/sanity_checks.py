"""Sanity checks and rollout diagnostics for CA world models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.analysis.closure_analysis import closure_debug_snapshot, latent_closure_error
from src.analysis.compare_rollouts import plot_multi_metric_vs_horizon
from src.ca.datasets import CADataset
from src.ca.observables import density
from src.training.eval import evaluate_exact_local_rule, evaluate_one_step
from src.training.rollout import model_rollout
from src.utils.config import load_config
from src.utils.factory import build_model, build_next_step_loader
from src.utils.io import ensure_dir, save_json
from src.utils.metrics import count_drift, density_drift, hamming_distance, probs_to_binary, shift_aligned_hamming_distance
from src.utils.plotting import save_figure, set_default_style
from src.utils.runtime import configure_runtime


def load_checkpoint_model(checkpoint: str, config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model = build_model(config).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_rollout_mode(
    model: torch.nn.Module,
    trajectories: torch.Tensor,
    horizon: int,
    feedback_mode: str,
    rollout_mode: str,
) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    records: list[dict[str, float | int | str]] = []
    first_pred = None
    first_target = None
    first_logits = None
    for idx in range(len(trajectories)):
        seq = trajectories[idx : idx + 1]
        effective_horizon = min(horizon, seq.shape[1] - 1)
        rollout = model_rollout(
            model,
            seq[:, 0],
            steps=effective_horizon,
            feedback_mode=feedback_mode,
            rollout_mode=rollout_mode,
        )
        pred = rollout["states"][:, : effective_horizon + 1]
        logits = rollout.get("logits")
        target = seq[:, : effective_horizon + 1]
        if idx == 0:
            first_pred = pred.detach().cpu()
            first_target = target.detach().cpu()
            first_logits = logits.detach().cpu() if logits is not None else None
        initial_state = target[:, 0]
        for t in range(1, pred.shape[1]):
            pred_t = pred[:, t]
            pred_binary = probs_to_binary(pred_t)
            target_t = target[:, t]
            aligned_hamming, best_shift = shift_aligned_hamming_distance(pred_binary, target_t)
            step_logits = logits[:, t] if logits is not None else None
            abs_logits = step_logits.abs() if step_logits is not None else None
            records.append(
                {
                    "sample": idx,
                    "horizon": t,
                    "feedback_mode": feedback_mode,
                    "rollout_mode": rollout_mode,
                    "hamming": float(hamming_distance(pred_binary, target_t).mean().item()),
                    "shift_aligned_hamming": float(aligned_hamming.mean().item()),
                    "best_shift": float(best_shift.float().mean().item()),
                    "count_drift_from_target": float(count_drift(pred_binary, target_t).mean().item()),
                    "count_drift_from_initial": float(count_drift(pred_binary, initial_state).mean().item()),
                    "density_drift_from_target": float(density_drift(pred_t, target_t).mean().item()),
                    "density_drift_from_initial": float(density_drift(pred_t, initial_state).mean().item()),
                    "target_density": float(density(target_t).mean().item()),
                    "pred_density": float(density(pred_t).mean().item()),
                    "rollout_bce": float(F.binary_cross_entropy_with_logits(step_logits, target_t).item())
                    if step_logits is not None
                    else float("nan"),
                    "logit_abs_mean": float(abs_logits.mean().item()) if abs_logits is not None else float("nan"),
                    "logit_abs_gt_5_frac": float((abs_logits > 5.0).float().mean().item()) if abs_logits is not None else float("nan"),
                }
            )
    if first_pred is None or first_target is None:
        raise ValueError("No trajectories available for sanity checks")
    return pd.DataFrame(records), first_pred, first_target, first_logits


@torch.no_grad()
def encoder_decoder_consistency(model: torch.nn.Module, trajectories: torch.Tensor) -> pd.DataFrame:
    flat_states = trajectories[:, :-1].reshape(-1, *trajectories.shape[2:])
    z = model.encode(flat_states)
    recon_logits = model.decode(z)
    recon_probs = torch.sigmoid(recon_logits)
    recon_hard = probs_to_binary(recon_probs)
    z_reencoded_soft = model.encode(recon_probs)
    z_reencoded_hard = model.encode(recon_hard)
    return pd.DataFrame(
        [
            {
                "recon_bce": float(F.binary_cross_entropy_with_logits(recon_logits, flat_states).item()),
                "recon_accuracy": float((recon_hard == flat_states).float().mean().item()),
                "latent_cycle_mse_soft": float((z_reencoded_soft - z).pow(2).mean().item()),
                "latent_cycle_mse_hard": float((z_reencoded_hard - z).pow(2).mean().item()),
            }
        ]
    )


@torch.no_grad()
def logit_histogram_table(logits: torch.Tensor) -> pd.DataFrame:
    records: list[dict[str, float | int]] = []
    for step in range(1, logits.shape[1]):
        step_logits = logits[:, step].reshape(-1)
        records.append(
            {
                "horizon": step,
                "logit_mean": float(step_logits.mean().item()),
                "logit_std": float(step_logits.std().item()) if step_logits.numel() > 1 else 0.0,
                "logit_abs_mean": float(step_logits.abs().mean().item()),
                "logit_abs_gt_5_frac": float((step_logits.abs() > 5.0).float().mean().item()),
                "prob_near_binary_frac": float(((torch.sigmoid(step_logits) < 0.01) | (torch.sigmoid(step_logits) > 0.99)).float().mean().item()),
            }
        )
    return pd.DataFrame(records)


def save_spacetime_plot(target: torch.Tensor, pred: torch.Tensor, path: Path, title: str) -> None:
    target_np = target[0, :, 0].numpy()
    pred_np = pred[0, :, 0].numpy()
    diff_np = np.abs(pred_np - target_np)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, image, name, cmap, vmin, vmax in [
        (axes[0], target_np, "Exact", "gray_r", 0, 1),
        (axes[1], pred_np, "Predicted", "gray_r", 0, 1),
        (axes[2], diff_np, "Difference", "magma", 0, 1),
    ]:
        ax.imshow(image, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.set_xlabel("Cell")
    axes[0].set_ylabel("Time")
    fig.suptitle(title)
    save_figure(fig, path)


def save_logit_histogram_plot(logits: torch.Tensor, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    max_steps = min(logits.shape[1] - 1, 8)
    for step in range(1, max_steps + 1):
        values = logits[:, step].reshape(-1).numpy()
        ax.hist(values, bins=40, alpha=0.35, density=True, label=f"t={step}")
    ax.set_xlabel("Decoder logit")
    ax.set_ylabel("Density")
    ax.set_title("Decoder logit histogram over rollout")
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, path)


def save_summary_csv(summary: dict[str, Any], path: Path) -> None:
    rows = []
    for key, value in summary.items():
        if isinstance(value, (int, float, str, bool)):
            rows.append({"metric": key, "value": value})
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--max_tau", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--cpu_threads", type=int, default=2)
    args = parser.parse_args()

    configure_runtime(cpu_threads=args.cpu_threads, interop_threads=1)
    set_default_style()
    config = load_config(args.config)
    device = torch.device(args.device)
    model = load_checkpoint_model(args.checkpoint, config, device)
    dataset_prefix = config.get("dataset_name", config["experiment_name"])
    test_path = Path(args.data_dir) / f"{dataset_prefix}_test.npz"
    trajectories = CADataset.from_npz(test_path).trajectories[: args.max_samples].to(device)
    next_loader = build_next_step_loader(
        test_path,
        batch_size=int(config["train"].get("batch_size", 32)),
        num_workers=int(config["train"].get("num_workers", 0)),
    )
    output_dir = ensure_dir(args.output_dir or Path(args.checkpoint).resolve().parent / "sanity_checks")

    local_rule_df = evaluate_exact_local_rule(
        model,
        device,
        config["model"]["name"],
        rule=int(config["ca"]["rule"]),
        input_size=int(config["ca"]["size"]),
    )
    one_step = evaluate_one_step(model, next_loader, device, config["model"]["name"])
    consistency_df = encoder_decoder_consistency(model, trajectories[:, : min(16, trajectories.shape[1])])
    closure_df = pd.DataFrame(latent_closure_error(model, trajectories[:, : min(args.max_tau + 1, trajectories.shape[1])], max_tau=args.max_tau))
    closure_debug = closure_debug_snapshot(model, trajectories[:1, 0], trajectories[:1, 1])

    hard_df, hard_pred, hard_target, hard_logits = evaluate_rollout_mode(
        model,
        trajectories,
        horizon=args.horizon,
        feedback_mode="hard",
        rollout_mode="reencode",
    )
    soft_df, soft_pred, soft_target, soft_logits = evaluate_rollout_mode(
        model,
        trajectories,
        horizon=args.horizon,
        feedback_mode="soft",
        rollout_mode="reencode",
    )
    latent_df, latent_pred, latent_target, latent_logits = evaluate_rollout_mode(
        model,
        trajectories,
        horizon=args.horizon,
        feedback_mode="hard",
        rollout_mode="latent",
    )

    rollout_df = pd.concat([latent_df, hard_df, soft_df], ignore_index=True)
    logits_for_hist = latent_logits if latent_logits is not None else hard_logits
    if logits_for_hist is None:
        raise ValueError("Expected logits from rollout for sanity checks")
    logit_df = logit_histogram_table(logits_for_hist)

    local_rule_df.to_csv(output_dir / "local_rule_metrics.csv", index=False)
    pd.DataFrame([one_step]).to_csv(output_dir / "one_step_metrics.csv", index=False)
    consistency_df.to_csv(output_dir / "encoder_decoder_consistency.csv", index=False)
    closure_df.to_csv(output_dir / "latent_closure.csv", index=False)
    rollout_df.to_csv(output_dir / "rollout_sanity.csv", index=False)
    logit_df.to_csv(output_dir / "logit_saturation.csv", index=False)
    save_json({"closure_debug": closure_debug}, output_dir / "closure_debug.json")

    plot_multi_metric_vs_horizon(
        closure_df,
        output_dir / "latent_closure_vs_horizon.png",
        ["closure_mse", "closure_rel_mse"],
        ["Closure MSE", "Closure Rel MSE"],
        x="tau",
    )
    latent_summary = latent_df.groupby("horizon").mean(numeric_only=True).reset_index()
    plot_multi_metric_vs_horizon(
        latent_summary,
        output_dir / "conservation_drift_vs_horizon.png",
        ["count_drift_from_initial", "density_drift_from_initial"],
        ["Count Drift", "Density Drift"],
    )
    plot_multi_metric_vs_horizon(
        latent_summary,
        output_dir / "raw_vs_shift_aligned_hamming.png",
        ["hamming", "shift_aligned_hamming"],
        ["Raw Hamming", "Shift-Aligned Hamming"],
    )
    save_spacetime_plot(latent_target, latent_pred, output_dir / "spacetime_latent_rollout.png", "Latent Rollout")
    save_spacetime_plot(hard_target, hard_pred, output_dir / "spacetime_hard_feedback.png", "Hard Feedback Rollout")
    save_spacetime_plot(soft_target, soft_pred, output_dir / "spacetime_soft_feedback.png", "Soft Feedback Rollout")
    save_logit_histogram_plot(logits_for_hist.cpu(), output_dir / "decoder_logit_histogram.png")

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "config": str(Path(args.config).resolve()),
        "dataset": str(test_path.resolve()),
        "teacher_forced_one_step_bce": one_step["one_step_bce"],
        "teacher_forced_one_step_acc": one_step["one_step_acc"],
        "local_rule_all_correct": bool(local_rule_df["correct"].all()),
        "local_rule_num_correct": int(local_rule_df["correct"].sum()),
        "latent_recon_bce": float(consistency_df["recon_bce"].iloc[0]),
        "latent_recon_accuracy": float(consistency_df["recon_accuracy"].iloc[0]),
        "latent_cycle_mse_soft": float(consistency_df["latent_cycle_mse_soft"].iloc[0]),
        "latent_cycle_mse_hard": float(consistency_df["latent_cycle_mse_hard"].iloc[0]),
        "closure_tau1_mse": float(closure_df.loc[closure_df["tau"] == 1, "closure_mse"].iloc[0]) if not closure_df.empty else float("nan"),
        "closure_tau_max_mse": float(closure_df["closure_mse"].iloc[-1]) if not closure_df.empty else float("nan"),
        "latent_rollout_final_hamming": float(latent_summary["hamming"].iloc[-1]),
        "latent_rollout_final_shift_aligned_hamming": float(latent_summary["shift_aligned_hamming"].iloc[-1]),
        "latent_rollout_final_count_drift": float(latent_summary["count_drift_from_initial"].iloc[-1]),
        "latent_rollout_final_density_drift": float(latent_summary["density_drift_from_initial"].iloc[-1]),
        "latent_rollout_final_logit_abs_mean": float(latent_summary["logit_abs_mean"].iloc[-1]),
        "latent_rollout_final_logit_abs_gt_5_frac": float(latent_summary["logit_abs_gt_5_frac"].iloc[-1]),
        "hard_feedback_final_hamming": float(hard_df.groupby("horizon")["hamming"].mean().iloc[-1]),
        "soft_feedback_final_hamming": float(soft_df.groupby("horizon")["hamming"].mean().iloc[-1]),
        "hard_feedback_mode": "hard",
        "soft_feedback_mode": "soft",
        "latent_rollout_mode": "latent",
        "feedback_rollout_mode": "reencode",
        "closure_target_detached": bool(closure_debug[0]["target_detached"]) if closure_debug else False,
    }
    save_json(summary, output_dir / "summary.json")
    save_summary_csv(summary, output_dir / "summary.csv")
    print(output_dir)


if __name__ == "__main__":
    main()
