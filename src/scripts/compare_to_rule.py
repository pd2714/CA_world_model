"""Compare model rollout against exact rule trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.ca.datasets import CADataset
from src.ca.visualization import plot_trajectory_comparison
from src.training.rollout import model_rollout
from src.utils.config import load_config
from src.utils.factory import build_model
from src.utils.runtime import configure_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--cpu_threads", type=int, default=2)
    args = parser.parse_args()

    configure_runtime(cpu_threads=args.cpu_threads, interop_threads=1)
    config = load_config(args.config)
    model = build_model(config).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model_state"])
    dataset_prefix = config.get("dataset_name", config["experiment_name"])
    dataset = CADataset.from_npz(Path(args.data_dir) / f"{dataset_prefix}_test.npz")
    seq = dataset.trajectories[args.sample_idx : args.sample_idx + 1].to(args.device)
    effective_horizon = min(args.horizon, seq.shape[1] - 1)
    exact = seq[0, : effective_horizon + 1, 0].cpu().numpy()
    pred = model_rollout(model, seq[:, 0], steps=effective_horizon)["states"][0, :, 0].cpu().numpy()
    output_dir = Path(args.output_dir or Path(args.checkpoint).resolve().parent / "compare")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"compare_sample_{args.sample_idx}.png"
    plot_trajectory_comparison(exact, pred, title=config["experiment_name"], path=out_path)
    print(out_path)


if __name__ == "__main__":
    main()
