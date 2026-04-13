"""Evaluate a trained model checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.training.eval import evaluate_exact_local_rule, evaluate_one_step, evaluate_rollout, save_eval_summary
from src.utils.config import load_config
from src.utils.factory import build_model, build_next_step_loader
from src.ca.datasets import CADataset
from src.utils.runtime import configure_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--feedback_mode", choices=["hard", "soft"], default="hard")
    parser.add_argument("--rollout_mode", choices=["latent", "reencode"], default="latent")
    parser.add_argument("--cpu_threads", type=int, default=2)
    args = parser.parse_args()

    configure_runtime(cpu_threads=args.cpu_threads, interop_threads=1)
    config = load_config(args.config)
    model = build_model(config).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model_state"])
    dataset_prefix = config.get("dataset_name", config["experiment_name"])
    test_path = Path(args.data_dir) / f"{dataset_prefix}_test.npz"
    next_loader = build_next_step_loader(
        test_path,
        batch_size=int(config["train"].get("batch_size", 32)),
        num_workers=int(config["train"].get("num_workers", 0)),
    )
    rollout_dataset = CADataset.from_npz(test_path)
    one_step = evaluate_one_step(model, next_loader, torch.device(args.device), config["model"]["name"])
    rollout = evaluate_rollout(
        model,
        rollout_dataset.trajectories,
        torch.device(args.device),
        horizon=args.horizon,
        model_name=config["model"]["name"],
        feedback_mode=args.feedback_mode,
        rollout_mode=args.rollout_mode,
    )
    local_rule = None
    if config["ca"].get("type") == "elementary_1d" and config["ca"].get("dimension") == "1d":
        local_rule = evaluate_exact_local_rule(
            model,
            torch.device(args.device),
            config["model"]["name"],
            rule=int(config["ca"]["rule"]),
            input_size=int(config["ca"]["size"]),
        )
    output_dir = args.output_dir or str(Path(args.checkpoint).resolve().parent / "eval")
    save_eval_summary(one_step, rollout, output_dir, local_rule_df=local_rule)
    print(output_dir)


if __name__ == "__main__":
    main()
