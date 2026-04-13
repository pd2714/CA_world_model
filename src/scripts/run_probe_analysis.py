"""Run linear probe and closure analysis from a checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.analysis.closure_analysis import latent_closure_error
from src.analysis.probe_analysis import run_probe_suite
from src.ca.datasets import CADataset
from src.ca.observables import domain_wall_density_1d, density
from src.utils.config import load_config
from src.utils.factory import build_model
from src.utils.runtime import configure_runtime


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_tau", type=int, default=8)
    parser.add_argument("--cpu_threads", type=int, default=2)
    args = parser.parse_args()

    configure_runtime(cpu_threads=args.cpu_threads, interop_threads=1)
    config = load_config(args.config)
    model = build_model(config).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model_state"])
    dataset_prefix = config.get("dataset_name", config["experiment_name"])
    dataset = CADataset.from_npz(Path(args.data_dir) / f"{dataset_prefix}_test.npz")
    trajectories = dataset.trajectories.to(args.device)
    if not hasattr(model, "encode"):
        raise ValueError("Probe analysis requires an encoder.")
    z = model.encode(trajectories[:, 0]).detach().cpu().numpy().reshape(len(trajectories), -1)
    probe_df = run_probe_suite(
        {
            "latent": z,
            "density_t": density(trajectories[:, 0]).cpu().numpy(),
            "density_t1": density(trajectories[:, 1]).cpu().numpy(),
            "domain_wall_t": domain_wall_density_1d(trajectories[:, 0]).cpu().numpy()
            if trajectories.ndim == 4
            else np.zeros(len(trajectories)),
            "domain_wall_t1": domain_wall_density_1d(trajectories[:, 1]).cpu().numpy()
            if trajectories.ndim == 4
            else np.zeros(len(trajectories)),
        },
        Path(args.output_dir or Path(args.checkpoint).resolve().parent / "probe") / "probe_metrics.csv",
    )
    closure_horizon = min(args.max_tau + 1, trajectories.shape[1])
    closure = pd.DataFrame(latent_closure_error(model, trajectories[:, :closure_horizon], max_tau=args.max_tau))
    out_dir = Path(args.output_dir or Path(args.checkpoint).resolve().parent / "probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    closure.to_csv(out_dir / "closure_metrics.csv", index=False)
    print(out_dir)
    print(probe_df)


if __name__ == "__main__":
    main()
