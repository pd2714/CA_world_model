"""Train baseline vs CA-biased 1D world models and aggregate metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.analysis.closure_analysis import latent_closure_error
from src.training.eval import evaluate_one_step, evaluate_rollout, save_eval_summary
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.factory import build_dataloaders, build_model, build_next_step_loader
from src.utils.runtime import configure_runtime
from src.utils.seed import set_seed


DEFAULT_BASELINE_CONFIGS = [
    "configs/experiment/rule30_best_long.yaml",
    "configs/experiment/rule54_best_long.yaml",
    "configs/experiment/rule110_best_long.yaml",
    "configs/experiment/rule184_best_long.yaml",
]

DEFAULT_BIASED_CONFIGS = [
    "configs/experiment/rule30_ca_biased_small.yaml",
    "configs/experiment/rule54_ca_biased_small.yaml",
    "configs/experiment/rule110_ca_biased_small.yaml",
    "configs/experiment/rule184_ca_biased_small.yaml",
]


def _closure_value(frame: pd.DataFrame, tau: int) -> float:
    if frame.empty or "tau" not in frame or "closure_mse" not in frame:
        return float("nan")
    match = frame[frame["tau"] == tau]
    if match.empty:
        return float("nan")
    return float(match["closure_mse"].iloc[0])


def run_comparison_experiment(
    config_path: str,
    output_root: Path,
    data_dir: Path,
    epochs: int,
    horizon: int,
    device: torch.device,
    batch_size: int | None,
) -> tuple[Path, dict]:
    config = load_config(config_path)
    set_seed(int(config.get("seed", 0)))
    config["train"]["epochs"] = epochs
    if batch_size is not None:
        config["train"]["batch_size"] = batch_size
    train_loader, val_loader, test_dataset = build_dataloaders(config, data_dir)
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["train"].get("lr", 1e-3)))
    run_dir = output_root / config["experiment_name"]
    trainer = Trainer(model=model, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader, config=config, run_dir=run_dir, device=device)
    best_path = trainer.fit()

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    dataset_prefix = config.get("dataset_name", config["experiment_name"])
    test_path = data_dir / f"{dataset_prefix}_test.npz"
    next_loader = build_next_step_loader(
        test_path,
        batch_size=int(config["train"].get("batch_size", 32)),
        num_workers=int(config["train"].get("num_workers", 0)),
    )
    one_step = evaluate_one_step(model, next_loader, device, config["model"]["name"])
    rollout_df = evaluate_rollout(
        model,
        test_dataset.trajectories,
        device,
        horizon=horizon,
        model_name=config["model"]["name"],
    )
    eval_dir = run_dir / "eval"
    save_eval_summary(one_step, rollout_df, eval_dir)

    probe_dir = run_dir / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    closure_df = pd.DataFrame(
        latent_closure_error(
            model,
            test_dataset.trajectories[:, : min(16, test_dataset.trajectories.shape[1])].to(device),
            max_tau=8,
        )
    )
    closure_df.to_csv(probe_dir / "closure_metrics.csv", index=False)
    return run_dir, config


def summarize_run(run_dir: Path, config: dict, setup: str) -> dict[str, float | int | str]:
    one_step = pd.read_csv(run_dir / "eval" / "one_step_metrics.csv").iloc[0]
    rollout_df = pd.read_csv(run_dir / "eval" / "rollout_metrics.csv")
    closure_df = pd.read_csv(run_dir / "probe" / "closure_metrics.csv")
    final_horizon = int(rollout_df["horizon"].max())
    final_rollout = rollout_df[rollout_df["horizon"] == final_horizon]
    return {
        "setup": setup,
        "experiment": run_dir.name,
        "rule": int(config["ca"]["rule"]),
        "one_step_bce": float(one_step["one_step_bce"]),
        "one_step_acc": float(one_step["one_step_acc"]),
        "rollout_horizon": final_horizon,
        "rollout_hamming_final": float(final_rollout["hamming"].mean()),
        "rollout_bce_final": float(final_rollout["rollout_bce"].mean()),
        "density_error_final": float(final_rollout["density_error"].mean()),
        "shift_aligned_hamming_final": float(final_rollout["shift_aligned_hamming"].mean()),
        "closure_mse_tau1": _closure_value(closure_df, 1),
        "closure_mse_tau4": _closure_value(closure_df, 4),
        "closure_mse_tau8": _closure_value(closure_df, 8),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="outputs/ca_biased_comparison")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--device", default="mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--cpu_threads", type=int, default=2)
    parser.add_argument("--baseline_configs", nargs="*", default=DEFAULT_BASELINE_CONFIGS)
    parser.add_argument("--biased_configs", nargs="*", default=DEFAULT_BIASED_CONFIGS)
    args = parser.parse_args()

    if len(args.baseline_configs) != len(args.biased_configs):
        raise ValueError("Baseline and biased config lists must have the same length.")

    configure_runtime(cpu_threads=args.cpu_threads, interop_threads=1)
    device = torch.device(args.device)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    summary_rows: list[dict[str, float | int | str]] = []
    config_groups = [("baseline", args.baseline_configs), ("ca_biased_small", args.biased_configs)]
    for setup, config_paths in config_groups:
        setup_root = output_root / setup
        setup_root.mkdir(parents=True, exist_ok=True)
        for config_path in config_paths:
            config = load_config(config_path)
            epochs = args.epochs if args.epochs is not None else int(config["train"]["epochs"])
            run_dir, config = run_comparison_experiment(
                config_path=config_path,
                output_root=setup_root,
                data_dir=data_dir,
                epochs=epochs,
                horizon=args.horizon,
                device=device,
                batch_size=args.batch_size,
            )
            summary_rows.append(summarize_run(run_dir, config, setup))

    summary_df = pd.DataFrame(summary_rows).sort_values(["rule", "setup"]).reset_index(drop=True)
    summary_path = output_root / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    baseline_df = summary_df[summary_df["setup"] == "baseline"].drop(columns=["setup", "experiment"]).rename(
        columns=lambda col: col if col == "rule" else f"{col}_baseline"
    )
    biased_df = summary_df[summary_df["setup"] == "ca_biased_small"].drop(columns=["setup", "experiment"]).rename(
        columns=lambda col: col if col == "rule" else f"{col}_biased"
    )
    paired_df = baseline_df.merge(biased_df, on="rule", how="inner")
    for metric in [
        "one_step_bce",
        "one_step_acc",
        "rollout_hamming_final",
        "rollout_bce_final",
        "density_error_final",
        "shift_aligned_hamming_final",
        "closure_mse_tau1",
        "closure_mse_tau4",
        "closure_mse_tau8",
    ]:
        paired_df[f"delta_{metric}"] = paired_df[f"{metric}_biased"] - paired_df[f"{metric}_baseline"]
    paired_path = output_root / "paired_summary.csv"
    paired_df.to_csv(paired_path, index=False)
    print(summary_path)
    print(paired_path)


if __name__ == "__main__":
    main()
