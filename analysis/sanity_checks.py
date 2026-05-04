"""Rollout sanity checks, baselines, and OOD/generalization evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from analysis.baselines import build_baselines
from analysis.plotting import plot_metric_curves
from analysis.utils import (
    OOD_DENSITIES,
    ROLLOUT_STEPS,
    RunSpec,
    clone_config_with_length,
    default_density_for_config,
    generate_trajectories,
    log,
    prepare_output_dir,
    rollout_main_model,
    save_skip,
    supports_arbitrary_length,
    temporal_count,
    temporal_density,
)
from src.utils.metrics import count_ones, hamming_distance, probs_to_binary


def _effective_steps(steps: tuple[int, ...], available_horizon: int) -> list[int]:
    return [step for step in steps if step <= available_horizon]


def summarize_rollout_metrics(
    pred_states: torch.Tensor,
    target_states: torch.Tensor,
    steps: list[int],
    name: str,
    logits: torch.Tensor | None = None,
    extra: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_states = pred_states[:, : target_states.shape[1]]
    records: list[dict[str, Any]] = []
    for step in steps:
        pred_t = pred_states[:, step]
        target_t = target_states[:, step]
        pred_binary = probs_to_binary(pred_t)
        hamming = hamming_distance(pred_binary, target_t)
        accuracy = 1.0 - hamming
        density_error = (pred_binary.float().mean(dim=tuple(range(1, pred_binary.ndim))) - target_t.float().mean(dim=tuple(range(1, target_t.ndim)))).abs()
        count_error = (count_ones(pred_binary) - count_ones(target_t)).abs()
        bce = (
            F.binary_cross_entropy_with_logits(logits[:, step], target_t, reduction="none")
            .mean(dim=tuple(range(1, target_t.ndim)))
            .detach()
            if logits is not None and step < logits.shape[1]
            else None
        )
        for sample_index in range(target_states.shape[0]):
            row: dict[str, Any] = {
                "model": name,
                "sample": sample_index,
                "step": step,
                "accuracy": float(accuracy[sample_index].item()),
                "hamming_error": float(hamming[sample_index].item()),
                "density_error": float(density_error[sample_index].item()),
                "count_error": float(count_error[sample_index].item()),
                "bce": float(bce[sample_index].item()) if bce is not None else float("nan"),
            }
            if extra:
                row.update(extra)
            records.append(row)
    detailed = pd.DataFrame(records)
    summary = detailed.groupby([column for column in detailed.columns if column not in {"sample", "accuracy", "hamming_error", "density_error", "count_error", "bce"}], dropna=False).mean(numeric_only=True).reset_index()
    return detailed, summary


@torch.no_grad()
def evaluate_main_model(
    spec: RunSpec,
    model: torch.nn.Module,
    trajectories: torch.Tensor,
    device: torch.device,
    steps: tuple[int, ...] = ROLLOUT_STEPS,
) -> tuple[pd.DataFrame, pd.DataFrame, torch.Tensor, torch.Tensor]:
    max_step = min(max(steps), trajectories.shape[1] - 1)
    effective_steps = _effective_steps(steps, max_step)
    target = trajectories[:, : max_step + 1].to(device)
    pred, logits = rollout_main_model(
        model,
        target[:, 0],
        steps=max_step,
        model_name=spec.model_name,
        feedback_mode="hard",
        rollout_mode="latent",
    )
    detailed, summary = summarize_rollout_metrics(
        pred_states=pred,
        target_states=target,
        steps=effective_steps,
        name=spec.model_name,
        logits=logits,
        extra={"source": "main_model"},
    )
    return detailed, summary, pred.detach().cpu(), target.detach().cpu()


def run_rollout_sanity_checks(
    spec: RunSpec,
    model: torch.nn.Module,
    trajectories: torch.Tensor,
    device: torch.device,
    output_dir: str | Path,
    steps: tuple[int, ...] = ROLLOUT_STEPS,
) -> dict[str, Any]:
    output_dir = prepare_output_dir(output_dir)
    detailed, summary, _, _ = evaluate_main_model(spec, model, trajectories, device, steps=steps)
    detailed.to_csv(output_dir / "rollout_sanity_detailed.csv", index=False)
    summary.to_csv(output_dir / "rollout_sanity_summary.csv", index=False)
    plot_metric_curves(summary, output_dir / "accuracy_vs_step.png", "accuracy", "model", "Rollout accuracy vs step", ylabel="Accuracy")
    plot_metric_curves(summary, output_dir / "hamming_vs_step.png", "hamming_error", "model", "Rollout Hamming error vs step", ylabel="Hamming error")
    if summary["bce"].notna().any():
        plot_metric_curves(summary, output_dir / "bce_vs_step.png", "bce", "model", "Rollout BCE vs step", ylabel="BCE")
    final_row = summary.sort_values("step").iloc[-1]
    return {
        "final_accuracy": float(final_row["accuracy"]),
        "final_hamming_error": float(final_row["hamming_error"]),
        "final_bce": float(final_row["bce"]) if not pd.isna(final_row["bce"]) else float("nan"),
    }


def run_baseline_comparison(
    spec: RunSpec,
    model: torch.nn.Module,
    train_trajectories: torch.Tensor,
    test_trajectories: torch.Tensor,
    data_dir: str | Path,
    device: torch.device,
    output_dir: str | Path,
    steps: tuple[int, ...] = ROLLOUT_STEPS,
) -> dict[str, Any]:
    output_dir = prepare_output_dir(output_dir)
    cache_dir = prepare_output_dir(Path(output_dir).parent / "_baseline_cache")
    detailed_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []

    model_detailed, model_summary, _, _ = evaluate_main_model(spec, model, test_trajectories, device, steps=steps)
    detailed_frames.append(model_detailed)
    summary_frames.append(model_summary)

    max_step = min(max(steps), test_trajectories.shape[1] - 1)
    effective_steps = _effective_steps(steps, max_step)
    targets = test_trajectories[:, : max_step + 1].to(device)

    for baseline_name, baseline in build_baselines(spec, train_trajectories, data_dir, device, cache_dir):
        log(f"{spec.run_name}: evaluating baseline {baseline_name}")
        pred, logits = baseline.rollout(targets[:, 0], steps=max_step)
        detailed, summary = summarize_rollout_metrics(
            pred_states=pred,
            target_states=targets,
            steps=effective_steps,
            name=baseline_name,
            logits=logits,
            extra={"source": "baseline"},
        )
        detailed_frames.append(detailed)
        summary_frames.append(summary)

    detailed_df = pd.concat(detailed_frames, ignore_index=True)
    summary_df = pd.concat(summary_frames, ignore_index=True).sort_values(["step", "model"]).reset_index(drop=True)
    detailed_df.to_csv(output_dir / "baseline_comparison_detailed.csv", index=False)
    summary_df.to_csv(output_dir / "baseline_comparison_summary.csv", index=False)
    plot_metric_curves(summary_df, output_dir / "baseline_accuracy_vs_step.png", "accuracy", "model", "Baseline comparison: accuracy", ylabel="Accuracy")
    plot_metric_curves(summary_df, output_dir / "baseline_hamming_vs_step.png", "hamming_error", "model", "Baseline comparison: Hamming error", ylabel="Hamming error")
    if summary_df["bce"].notna().any():
        plot_metric_curves(summary_df[summary_df["bce"].notna()], output_dir / "baseline_bce_vs_step.png", "bce", "model", "Baseline comparison: BCE", ylabel="BCE")
    final_rows = summary_df.sort_values("step").groupby("model").tail(1)
    return {"num_models_compared": int(final_rows.shape[0])}


def run_ood_density_test(
    spec: RunSpec,
    model: torch.nn.Module,
    device: torch.device,
    output_dir: str | Path,
    num_samples: int = 64,
    steps: tuple[int, ...] = ROLLOUT_STEPS,
) -> dict[str, Any]:
    output_dir = prepare_output_dir(output_dir)
    max_step = max(steps)
    detailed_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    conservation_rows: list[dict[str, Any]] = []

    for density_value in OOD_DENSITIES:
        trajectories = generate_trajectories(spec.config, p=density_value, num_samples=num_samples, steps=max_step)
        detailed, summary, pred, target = evaluate_main_model(spec, model, trajectories, device, steps=steps)
        detailed["density_p"] = density_value
        summary["density_p"] = density_value
        detailed_frames.append(detailed)
        summary_frames.append(summary)

        if spec.rule == 184:
            target_counts = temporal_count(target[:, : max_step + 1]).cpu().numpy()
            pred_counts = temporal_count(probs_to_binary(pred[:, : max_step + 1])).cpu().numpy()
            target_density = temporal_density(target[:, : max_step + 1]).cpu().numpy()
            pred_density = temporal_density(probs_to_binary(pred[:, : max_step + 1])).cpu().numpy()
            for step_index in range(target_counts.shape[1]):
                conservation_rows.append(
                    {
                        "density_p": density_value,
                        "step": step_index,
                        "exact_count": float(target_counts[:, step_index].mean()),
                        "pred_count": float(pred_counts[:, step_index].mean()),
                        "exact_density": float(target_density[:, step_index].mean()),
                        "pred_density": float(pred_density[:, step_index].mean()),
                    }
                )

    detailed_df = pd.concat(detailed_frames, ignore_index=True)
    summary_df = pd.concat(summary_frames, ignore_index=True).sort_values(["density_p", "step"]).reset_index(drop=True)
    detailed_df.to_csv(output_dir / "ood_density_detailed.csv", index=False)
    summary_df.to_csv(output_dir / "ood_density_summary.csv", index=False)
    summary_df["density_label"] = summary_df["density_p"].map(lambda value: f"p={value:g}")
    plot_metric_curves(summary_df, output_dir / "ood_accuracy_vs_step.png", "accuracy", "density_label", "OOD density accuracy", ylabel="Accuracy")
    plot_metric_curves(summary_df, output_dir / "ood_hamming_vs_step.png", "hamming_error", "density_label", "OOD density Hamming error", ylabel="Hamming error")

    if conservation_rows:
        conservation_df = pd.DataFrame(conservation_rows)
        conservation_df.to_csv(output_dir / "rule184_conservation_by_density.csv", index=False)
        conservation_long = pd.concat(
            [
                conservation_df.rename(columns={"exact_count": "value"}).assign(series="exact_count")[["density_p", "step", "series", "value"]],
                conservation_df.rename(columns={"pred_count": "value"}).assign(series="pred_count")[["density_p", "step", "series", "value"]],
            ],
            ignore_index=True,
        )
        conservation_long["density_label"] = conservation_long["density_p"].map(lambda value: f"p={value:g}")
        plot_metric_curves(
            conservation_long.rename(columns={"value": "particle_count"}),
            output_dir / "rule184_particle_count_vs_step.png",
            "particle_count",
            "density_label",
            "Rule 184 particle count over time",
            ylabel="Particle count",
            style="series",
        )
        density_long = pd.concat(
            [
                conservation_df.rename(columns={"exact_density": "value"}).assign(series="exact_density")[["density_p", "step", "series", "value"]],
                conservation_df.rename(columns={"pred_density": "value"}).assign(series="pred_density")[["density_p", "step", "series", "value"]],
            ],
            ignore_index=True,
        )
        density_long["density_label"] = density_long["density_p"].map(lambda value: f"p={value:g}")
        plot_metric_curves(
            density_long.rename(columns={"value": "density"}),
            output_dir / "rule184_density_vs_step.png",
            "density",
            "density_label",
            "Rule 184 density over time",
            ylabel="Density",
            style="series",
        )
    return {"num_densities": len(OOD_DENSITIES)}


def run_lattice_size_generalization(
    spec: RunSpec,
    model: torch.nn.Module,
    device: torch.device,
    output_dir: str | Path,
    num_samples: int = 64,
    steps: tuple[int, ...] = ROLLOUT_STEPS,
) -> dict[str, Any]:
    output_dir = prepare_output_dir(output_dir)
    supported, reason = supports_arbitrary_length(spec.config)
    if not supported:
        save_skip(output_dir / "summary.json", reason)
        return {"status": "skipped", "reason": reason}
    if int(spec.config["ca"]["size"]) != 128:
        reason = f"Expected training size 128, found {spec.config['ca']['size']}."
        save_skip(output_dir / "summary.json", reason)
        return {"status": "skipped", "reason": reason}

    config_256 = clone_config_with_length(spec.config, 256)
    trajectories = generate_trajectories(
        config_256,
        p=default_density_for_config(spec.config),
        num_samples=num_samples,
        steps=max(steps),
    )
    detailed, summary, _, _ = evaluate_main_model(spec, model, trajectories, device, steps=steps)
    detailed.to_csv(output_dir / "lattice_generalization_detailed.csv", index=False)
    summary.to_csv(output_dir / "lattice_generalization_summary.csv", index=False)
    plot_metric_curves(summary, output_dir / "lattice_accuracy_vs_step.png", "accuracy", "model", "L=256 accuracy from L=128 checkpoint", ylabel="Accuracy")
    plot_metric_curves(summary, output_dir / "lattice_hamming_vs_step.png", "hamming_error", "model", "L=256 Hamming error from L=128 checkpoint", ylabel="Hamming error")
    return {"status": "ok"}
