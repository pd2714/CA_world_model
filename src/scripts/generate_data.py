"""Generate CA trajectory datasets from config."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.config import load_config
from src.utils.factory import generate_trajectories
from src.utils.io import ensure_dir, save_npz
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default="data")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 0)))
    output_dir = ensure_dir(args.output_dir)
    dataset_prefix = config.get("dataset_name", config["experiment_name"])
    for split in ["train", "val", "test"]:
        trajectories, metadata = generate_trajectories(config, split)
        path = Path(output_dir) / f"{dataset_prefix}_{split}.npz"
        save_npz(path, trajectories=trajectories, metadata=metadata)
        print(f"saved {split}: {path}")


if __name__ == "__main__":
    main()
