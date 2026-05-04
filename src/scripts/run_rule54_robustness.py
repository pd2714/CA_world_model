"""Run a compact robustness check for the promoted rule54_ca_light model."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.analysis.closure_analysis import latent_closure_error
from src.ca.datasets import CADataset
from src.training.eval import evaluate_one_step, evaluate_rollout, save_eval_summary
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.factory import build_dataloaders, build_model, build_next_step_loader, generate_trajectories
from src.utils.runtime import configure_runtime
from src.utils.seed import set_seed


def summarize_eval(run_dir: Path, seed: int | None = None, label: str | None = None) -> dict[str, float | int | str]:
    one = pd.read_csv(run_dir / "eval" / "one_step_metrics.csv").iloc[0]
    rollout = pd.read_csv(run_dir / "eval" / "rollout_metrics.csv")
    final = rollout[rollout["horizon"] == rollout["horizon"].max()].mean(numeric_only=True)
    closure = pd.read_csv(run_dir / "probe" / "closure_metrics.csv")
    row: dict[str, float | int | str] = {
        "run": label or run_dir.name,
        "one_step_bce": float(one["one_step_bce"]),
        "one_step_acc": float(one["one_step_acc"]),
        "rollout_horizon": int(rollout["horizon"].max()),
        "rollout_hamming_final": float(final["hamming"]),
        "rollout_bce_final": float(final["rollout_bce"]),
        "density_error_final": float(final["density_error"]),
        "shift_aligned_hamming_final": float(final["shift_aligned_hamming"]),
        "closure_tau1": float(closure[closure["tau"] == 1]["closure_mse"].iloc[0]),
        "closure_tau4": float(closure[closure["tau"] == 4]["closure_mse"].iloc[0]),
        "closure_tau8": float(closure[closure["tau"] == 8]["closure_mse"].iloc[0]),
    }
    if seed is not None:
        row["seed"] = seed
    return row


def train_and_eval_seed(config_path: str, output_root: Path, data_dir: Path, device: torch.device, seed: int) -> Path:
    config = load_config(config_path)
    config["seed"] = seed
    config["experiment_name"] = f"{config['experiment_name']}_seed{seed}"
    set_seed(seed)
    run_dir = output_root / config["experiment_name"]
    best_path = run_dir / "best.ckpt"

    train_loader, val_loader, test_dataset = build_dataloaders(config, data_dir)
    if not best_path.exists():
        model = build_model(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["train"].get("lr", 1e-3)))
        trainer = Trainer(model=model, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader, config=config, run_dir=run_dir, device=device)
        best_path = Path(trainer.fit())

    model = build_model(config).to(device)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    dataset_prefix = config.get("dataset_name", config["experiment_name"])
    test_path = data_dir / f"{dataset_prefix}_test.npz"
    next_loader = build_next_step_loader(
        test_path,
        batch_size=int(config["train"].get("batch_size", 32)),
        num_workers=int(config["train"].get("num_workers", 0)),
    )

    eval_dir = run_dir / "eval"
    if not (eval_dir / "one_step_metrics.csv").exists():
        one_step = evaluate_one_step(model, next_loader, device, config["model"]["name"])
        rollout_df = evaluate_rollout(model, test_dataset.trajectories, device, horizon=24, model_name=config["model"]["name"])
        save_eval_summary(one_step, rollout_df, eval_dir)

    probe_dir = run_dir / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    if not (probe_dir / "closure_metrics.csv").exists():
        closure_df = pd.DataFrame(
            latent_closure_error(
                model,
                test_dataset.trajectories[:, : min(16, test_dataset.trajectories.shape[1])].to(device),
                max_tau=8,
            )
        )
        closure_df.to_csv(probe_dir / "closure_metrics.csv", index=False)
    return run_dir


def save_generated_dataset(config: dict, split: str, out_path: Path) -> CADataset:
    trajectories, metadata = generate_trajectories(config, split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, trajectories=trajectories, metadata=np.array(metadata, dtype=object))
    return CADataset.from_npz(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/rule54_ca_light.yaml")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_root", default="outputs/rule54_ca_light_robustness_20260413")
    parser.add_argument("--device", default="mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--cpu_threads", type=int, default=2)
    parser.add_argument(
        "--max_rollout_trajectories",
        type=int,
        default=None,
        help="Optional cap for long-horizon and IID/OOD rollout evaluation trajectories.",
    )
    args = parser.parse_args()

    configure_runtime(cpu_threads=args.cpu_threads, interop_threads=1)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    device = torch.device(args.device)

    base_config = load_config(args.config)
    base_run = Path("outputs/followup_ca_tuning_20260413/rule54_ca_light")

    def maybe_subset(trajectories: torch.Tensor) -> torch.Tensor:
        if args.max_rollout_trajectories is None:
            return trajectories
        return trajectories[: args.max_rollout_trajectories]

    # 1) Seed sweep: reuse seed 11, add 7 and 29.
    seed_rows = [summarize_eval(base_run, seed=11, label="rule54_ca_light_seed11")]
    seed_root = output_root / "seed_sweep"
    for seed in [7, 29]:
        run_dir = train_and_eval_seed(args.config, seed_root, data_dir, device, seed)
        seed_rows.append(summarize_eval(run_dir, seed=seed))
    seed_df = pd.DataFrame(seed_rows).sort_values("seed").reset_index(drop=True)
    seed_df.to_csv(output_root / "seed_summary.csv", index=False)
    seed_df.select_dtypes(include=["number"]).agg(["mean", "std", "min", "max"]).to_csv(output_root / "seed_aggregate.csv")

    # 2) Longer rollout horizon on the promoted winner using the current OOD test set.
    winner_model = build_model(base_config).to(device)
    winner_ckpt = torch.load(base_run / "best.ckpt", map_location=device)
    winner_model.load_state_dict(winner_ckpt["model_state"])
    winner_model.eval()
    ood_test = CADataset.from_npz(data_dir / "rule54_dense_test.npz")
    ood_rollout_trajs = maybe_subset(ood_test.trajectories)
    horizon_rows = []
    for horizon in [24, 48, 64]:
        rollout = evaluate_rollout(
            winner_model,
            ood_rollout_trajs,
            device,
            horizon=horizon,
            model_name=base_config["model"]["name"],
        )
        final = rollout[rollout["horizon"] == rollout["horizon"].max()].mean(numeric_only=True)
        horizon_rows.append(
            {
                "horizon": horizon,
                "num_trajectories": int(ood_rollout_trajs.shape[0]),
                "rollout_hamming_final": float(final["hamming"]),
                "rollout_bce_final": float(final["rollout_bce"]),
                "density_error_final": float(final["density_error"]),
                "shift_aligned_hamming_final": float(final["shift_aligned_hamming"]),
            }
        )
    pd.DataFrame(horizon_rows).to_csv(output_root / "long_horizon_summary.csv", index=False)

    # 3) Explicit IID vs OOD evaluation. The on-disk test set is already OOD (p=0.3).
    eval_root = output_root / "distribution_eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    iid_config = deepcopy(base_config)
    iid_config["dataset"]["ood"] = {}
    iid_path = eval_root / "rule54_dense_iid_test.npz"
    iid_dataset = save_generated_dataset(iid_config, "test", iid_path)
    iid_loader = build_next_step_loader(iid_path, batch_size=int(base_config["train"].get("batch_size", 32)))
    iid_one = evaluate_one_step(winner_model, iid_loader, device, base_config["model"]["name"])
    iid_roll = evaluate_rollout(
        winner_model,
        maybe_subset(iid_dataset.trajectories),
        device,
        horizon=24,
        model_name=base_config["model"]["name"],
    )
    save_eval_summary(iid_one, iid_roll, eval_root / "iid_eval")

    ood_loader = build_next_step_loader(data_dir / "rule54_dense_test.npz", batch_size=int(base_config["train"].get("batch_size", 32)))
    ood_one = evaluate_one_step(winner_model, ood_loader, device, base_config["model"]["name"])
    ood_roll = evaluate_rollout(
        winner_model,
        ood_rollout_trajs,
        device,
        horizon=24,
        model_name=base_config["model"]["name"],
    )
    save_eval_summary(ood_one, ood_roll, eval_root / "ood_eval")

    dist_rows = []
    for name, one, roll in [("iid_p0.5", iid_one, iid_roll), ("ood_p0.3", ood_one, ood_roll)]:
        final = roll[roll["horizon"] == roll["horizon"].max()].mean(numeric_only=True)
        dist_rows.append(
            {
                "distribution": name,
                "one_step_bce": float(one["one_step_bce"]),
                "one_step_acc": float(one["one_step_acc"]),
                "num_rollout_trajectories": int(roll["sample"].max() + 1) if len(roll) else 0,
                "rollout_hamming_final": float(final["hamming"]),
                "rollout_bce_final": float(final["rollout_bce"]),
                "density_error_final": float(final["density_error"]),
                "shift_aligned_hamming_final": float(final["shift_aligned_hamming"]),
            }
        )
    pd.DataFrame(dist_rows).to_csv(output_root / "iid_vs_ood_summary.csv", index=False)

    # 4) Smaller-capacity ablation using the completed ca_biased_small run.
    ablation_rows = []
    baseline = {
        "model": "rule54_best_long_previous",
        "one_step_bce": 2.981628740958274e-07,
        "rollout_hamming_final": 0.000244140625,
        "rollout_bce_final": 0.0006952292278512382,
        "density_error_final": 0.000244140625,
        "closure_tau8": 0.0599991530179977,
    }
    winner = summarize_eval(base_run, label="rule54_ca_light")
    small = summarize_eval(Path("outputs/ca_biased_only_20260413/rule54_ca_biased_small"), label="rule54_ca_biased_small")
    ablation_rows.append(baseline)
    ablation_rows.append(
        {
            "model": "rule54_ca_light",
            "one_step_bce": winner["one_step_bce"],
            "rollout_hamming_final": winner["rollout_hamming_final"],
            "rollout_bce_final": winner["rollout_bce_final"],
            "density_error_final": winner["density_error_final"],
            "closure_tau8": winner["closure_tau8"],
        }
    )
    ablation_rows.append(
        {
            "model": "rule54_ca_biased_small",
            "one_step_bce": small["one_step_bce"],
            "rollout_hamming_final": small["rollout_hamming_final"],
            "rollout_bce_final": small["rollout_bce_final"],
            "density_error_final": small["density_error_final"],
            "closure_tau8": small["closure_tau8"],
        }
    )
    pd.DataFrame(ablation_rows).to_csv(output_root / "capacity_ablation_summary.csv", index=False)

    print(output_root)


if __name__ == "__main__":
    main()
