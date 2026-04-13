"""Run a practical end-to-end 1D CA experiment sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.analysis.compare_rollouts import plot_error_vs_horizon
from src.ca.datasets import CADataset
from src.ca.observables import density
from src.training.eval import evaluate_one_step, evaluate_rollout, save_eval_summary
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.factory import build_dataloaders, build_model, build_next_step_loader
from src.utils.runtime import configure_runtime
from src.utils.seed import set_seed


DEFAULT_EXPERIMENTS = [
    "configs/experiment/rule184_dense.yaml",
    "configs/experiment/rule54_dense.yaml",
    "configs/experiment/rule110_dense.yaml",
    "configs/experiment/rule30_dense.yaml",
    "configs/experiment/rule54_pixel.yaml",
    "configs/experiment/rule54_object.yaml",
]


@torch.no_grad()
def run_posthoc_artifacts(
    model: torch.nn.Module,
    config: dict,
    run_dir: Path,
    data_dir: Path,
    device: torch.device,
    horizon: int,
) -> None:
    from src.analysis.closure_analysis import latent_closure_error
    from src.analysis.probe_analysis import run_probe_suite
    from src.analysis.latent_viz import save_projection_bundle
    from src.ca.visualization import plot_trajectory_comparison
    from src.training.rollout import model_rollout

    dataset_prefix = config.get("dataset_name", config["experiment_name"])
    test_path = data_dir / f"{dataset_prefix}_test.npz"
    next_loader = build_next_step_loader(test_path, batch_size=int(config["train"]["batch_size"]))
    rollout_dataset = CADataset.from_npz(test_path)
    one_step = evaluate_one_step(model, next_loader, device, config["model"]["name"])
    rollout_df = evaluate_rollout(
        model,
        rollout_dataset.trajectories,
        device,
        horizon=horizon,
        model_name=config["model"]["name"],
    )
    eval_dir = run_dir / "eval"
    save_eval_summary(one_step, rollout_df, eval_dir)
    rollout_df["model"] = config["experiment_name"]
    plot_error_vs_horizon(rollout_df, eval_dir / "hamming_vs_horizon.png", metric="hamming")
    plot_error_vs_horizon(rollout_df, eval_dir / "density_error_vs_horizon.png", metric="density_error")

    seq = rollout_dataset.trajectories[:1].to(device)
    effective_horizon = min(horizon, seq.shape[1] - 1)
    pred = model_rollout(model, seq[:, 0], steps=effective_horizon)["states"][0, :, 0].cpu().numpy()
    exact = seq[0, : effective_horizon + 1, 0].cpu().numpy()
    compare_dir = run_dir / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    plot_trajectory_comparison(exact, pred, title=config["experiment_name"], path=compare_dir / "compare_sample_0.png")

    if hasattr(model, "encode"):
        x = rollout_dataset.trajectories[:, 0].to(device)
        encoded = model.encode(x)
        latents_tensor = encoded[0] if isinstance(encoded, tuple) else encoded
        latents = latents_tensor.detach().cpu().numpy()
        colors = density(x).cpu().numpy()
        save_projection_bundle(latents=latents, color=colors, out_dir=run_dir / "latent_viz", title_prefix=config["experiment_name"])

        flat_latents = latents.reshape(latents.shape[0], -1)
        features = {
            "latent": flat_latents,
            "density_t": density(rollout_dataset.trajectories[:, 0]).cpu().numpy(),
            "density_t1": density(rollout_dataset.trajectories[:, 1]).cpu().numpy(),
        }
        probe_dir = run_dir / "probe"
        probe_dir.mkdir(parents=True, exist_ok=True)
        probe_df = run_probe_suite(features, probe_dir / "probe_metrics.csv")
        pd.DataFrame(latent_closure_error(model, rollout_dataset.trajectories[:, : min(16, rollout_dataset.trajectories.shape[1])].to(device), max_tau=8)).to_csv(
            probe_dir / "closure_metrics.csv", index=False
        )
        probe_df.to_csv(probe_dir / "probe_metrics.csv", index=False)


def run_experiment(
    config_path: str,
    output_root: Path,
    data_dir: Path,
    epochs: int,
    horizon: int,
    device: torch.device,
    batch_size: int | None,
) -> Path:
    config = load_config(config_path)
    set_seed(int(config.get("seed", 0)))
    config["train"]["epochs"] = epochs
    if batch_size is not None:
        config["train"]["batch_size"] = batch_size
    train_loader, val_loader, _ = build_dataloaders(config, data_dir)
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["train"].get("lr", 1e-3)))
    run_dir = output_root / config["experiment_name"]
    trainer = Trainer(model=model, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader, config=config, run_dir=run_dir, device=device)
    best_path = trainer.fit()
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    run_posthoc_artifacts(model, config, run_dir, data_dir, device, horizon)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="outputs/1d_sweep")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cpu_threads", type=int, default=2)
    parser.add_argument("--experiments", nargs="*", default=DEFAULT_EXPERIMENTS)
    args = parser.parse_args()

    configure_runtime(cpu_threads=args.cpu_threads, interop_threads=1)
    device = torch.device(args.device)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    summary_rows = []
    for config_path in args.experiments:
        run_dir = run_experiment(config_path, output_root, data_dir, args.epochs, args.horizon, device, args.batch_size)
        one_step = pd.read_csv(run_dir / "eval" / "one_step_metrics.csv").iloc[0].to_dict()
        rollout_df = pd.read_csv(run_dir / "eval" / "rollout_metrics.csv")
        summary_rows.append(
            {
                "experiment": run_dir.name,
                **one_step,
                "hamming_h32": float(rollout_df[rollout_df["horizon"] == rollout_df["horizon"].max()]["hamming"].mean()),
                "density_err_h32": float(rollout_df[rollout_df["horizon"] == rollout_df["horizon"].max()]["density_error"].mean()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(output_root / "summary.csv", index=False)
    print(output_root)


if __name__ == "__main__":
    main()
