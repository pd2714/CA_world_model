"""Generate random exact-vs-predicted rollout comparisons for a checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
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
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    rng = np.random.default_rng(args.seed)
    sample_count = min(args.num_samples, len(dataset))
    sample_indices = sorted(int(idx) for idx in rng.choice(len(dataset), size=sample_count, replace=False).tolist())

    output_dir = Path(args.output_dir or Path(args.checkpoint).resolve().parent / "random_compare_h100")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str | None]] = []
    for sample_idx in sample_indices:
        seq = dataset.trajectories[sample_idx : sample_idx + 1].to(args.device)
        effective_horizon = min(args.horizon, seq.shape[1] - 1)
        pred = model_rollout(model, seq[:, 0], steps=effective_horizon)["states"][0, :, 0].cpu()
        exact = seq[0, : effective_horizon + 1, 0].cpu()
        hammings = (pred != exact).float().view(pred.shape[0], -1).mean(dim=1)
        mismatch_steps = [int(t) for t, value in enumerate(hammings.tolist()) if value > 0.0]

        plot_trajectory_comparison(
            exact.numpy(),
            pred.numpy(),
            title=f"{config['experiment_name']} sample {sample_idx} horizon {effective_horizon}",
            path=output_dir / f"compare_sample_{sample_idx}_h{effective_horizon}.png",
        )

        rows.append(
            {
                "sample_idx": sample_idx,
                "first_mismatch_t": mismatch_steps[0] if mismatch_steps else None,
                "num_mismatch_steps": len(mismatch_steps),
                "last_mismatch_t": mismatch_steps[-1] if mismatch_steps else None,
                "final_hamming": float(hammings[-1].item()),
                "mean_hamming": float(hammings.mean().item()),
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame({"sample_idx": sample_indices}).to_csv(output_dir / "sample_indices.csv", index=False)
    print(output_dir)


if __name__ == "__main__":
    main()
