"""Train a model from config."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

from src.ca.datasets import CADataset
from src.ca.visualization import plot_trajectory_comparison
from src.training.trainer import Trainer
from src.training.rollout import model_rollout
from src.utils.config import load_config
from src.utils.factory import build_dataloaders, build_model
from src.utils.runtime import configure_runtime
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--init_checkpoint", default=None)
    parser.add_argument("--resume_optimizer", action="store_true")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compare_horizon", type=int, default=64)
    parser.add_argument("--cpu_threads", type=int, default=2)
    args = parser.parse_args()

    configure_runtime(cpu_threads=args.cpu_threads, interop_threads=1)
    config = load_config(args.config)
    config.setdefault("train", {})
    config["train"].setdefault("num_workers", 0)
    if args.init_checkpoint:
        config["init_checkpoint"] = str(Path(args.init_checkpoint).resolve())
        if args.resume_optimizer:
            config["resume_optimizer"] = True
    set_seed(int(config.get("seed", 0)))
    train_loader, val_loader, _ = build_dataloaders(config, args.data_dir)
    model = build_model(config).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["train"].get("lr", 1e-3)))
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=args.device)
        model.load_state_dict(ckpt["model_state"])
        if args.resume_optimizer and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
    run_name = f"{config['experiment_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_root) / run_name
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        run_dir=run_dir,
        device=torch.device(args.device),
    )
    best = trainer.fit()
    ckpt = torch.load(best, map_location=args.device)
    model.load_state_dict(ckpt["model_state"])

    # Always emit a simple exact-vs-predicted comparison for the first test sample.
    dataset_prefix = config.get("dataset_name", config["experiment_name"])
    test_path = Path(args.data_dir) / f"{dataset_prefix}_test.npz"
    if test_path.exists():
        dataset = CADataset.from_npz(test_path)
        seq = dataset.trajectories[:1].to(args.device)
        effective_horizon = min(args.compare_horizon, seq.shape[1] - 1)
        pred = model_rollout(model, seq[:, 0], steps=effective_horizon)["states"][0, :, 0].cpu().numpy()
        exact = seq[0, : effective_horizon + 1, 0].cpu().numpy()
        compare_dir = run_dir / "compare"
        compare_dir.mkdir(parents=True, exist_ok=True)
        plot_trajectory_comparison(exact, pred, title=config["experiment_name"], path=compare_dir / "compare_sample_0.png")
    print(best)


if __name__ == "__main__":
    main()
