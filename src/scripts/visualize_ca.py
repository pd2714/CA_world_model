"""Visualize exact CA rollouts from config."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ca.visualization import animate_2d_trajectory, plot_1d_trajectory
from src.utils.config import load_config
from src.utils.factory import generate_trajectories
from src.utils.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default="outputs/ca_viz")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_dir(args.output_dir)
    trajectories, _ = generate_trajectories(config, "test")
    trajectory = trajectories[0, :, 0]
    if config["ca"]["dimension"] == "1d":
        path = Path(output_dir) / "exact_rollout.png"
        plot_1d_trajectory(trajectory, title=f"Rule visualization: {config['experiment_name']}", path=path)
        print(path)
    else:
        path = Path(output_dir) / "exact_rollout.gif"
        animate_2d_trajectory(trajectory, path=path)
        print(path)


if __name__ == "__main__":
    main()
