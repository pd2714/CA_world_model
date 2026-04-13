"""Visualize latent states from a checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.ca.datasets import CADataset
from src.ca.observables import density
from src.analysis.latent_viz import save_projection_bundle
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
    parser.add_argument("--cpu_threads", type=int, default=2)
    args = parser.parse_args()

    configure_runtime(cpu_threads=args.cpu_threads, interop_threads=1)
    config = load_config(args.config)
    model = build_model(config).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model_state"])
    dataset_prefix = config.get("dataset_name", config["experiment_name"])
    dataset = CADataset.from_npz(Path(args.data_dir) / f"{dataset_prefix}_test.npz")
    x = dataset.trajectories[:, 0].to(args.device)
    if not hasattr(model, "encode"):
        raise ValueError("This model does not expose encode().")
    latent = model.encode(x).detach().cpu().numpy()
    color = density(x).cpu().numpy()
    output_dir = args.output_dir or str(Path(args.checkpoint).resolve().parent / "latent_viz")
    save_projection_bundle(latents=latent, color=np.asarray(color), out_dir=output_dir, title_prefix=config["experiment_name"])
    print(output_dir)


if __name__ == "__main__":
    main()
