"""Plot saved evaluation tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.compare_rollouts import plot_error_vs_horizon, plot_multi_metric_vs_horizon


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    eval_dir = run_dir / "eval"
    rollout_df = pd.read_csv(eval_dir / "rollout_metrics.csv")
    rollout_df["model"] = run_dir.name
    plot_error_vs_horizon(rollout_df, eval_dir / "hamming_vs_horizon.png", metric="hamming")
    plot_error_vs_horizon(rollout_df, eval_dir / "density_error_vs_horizon.png", metric="density_error")
    if {"shift_aligned_hamming", "hamming"}.issubset(rollout_df.columns):
        plot_multi_metric_vs_horizon(
            rollout_df,
            eval_dir / "raw_vs_shift_aligned_hamming.png",
            metrics=["hamming", "shift_aligned_hamming"],
            labels=["Raw Hamming", "Shift-Aligned Hamming"],
        )


if __name__ == "__main__":
    main()
