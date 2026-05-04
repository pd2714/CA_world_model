"""Factories for automata, datasets, and models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.ca.datasets import CADataset, NextStepDataset, SequenceWindowDataset
from src.ca.elementary_1d import ElementaryCA
from src.ca.life_like_2d import LifeLikeCA
from src.ca.trajectory import rollout_batch, sample_initial_state
from src.models.dense_world_model import DenseWorldModel
from src.models.legacy_dense_world_model import LegacyDenseWorldModel
from src.models.object_world_model import ObjectWorldModel
from src.models.pixel_predictor import PixelPredictor
from src.utils.runtime import dataloader_num_workers


def build_automaton(config: dict[str, Any]):
    ca_cfg = config["ca"]
    if ca_cfg["type"] == "elementary_1d":
        return ElementaryCA(rule=int(ca_cfg["rule"]))
    if ca_cfg["type"] == "life_like_2d":
        return LifeLikeCA(rule=str(ca_cfg.get("rule_string", "B3/S23")))
    raise ValueError(f"Unsupported CA type: {ca_cfg['type']}")


def generate_trajectories(config: dict[str, Any], split: str) -> tuple[np.ndarray, dict[str, Any]]:
    split_offsets = {"train": 0, "val": 1, "test": 2}
    rng = np.random.default_rng(int(config.get("seed", 0)) + split_offsets.get(split, 0))
    automaton = build_automaton(config)
    data_cfg = config["dataset"]
    split_cfg = data_cfg["splits"][split]
    init_cfg = dict(data_cfg["initial_condition"])
    if split in data_cfg.get("ood", {}):
        init_cfg.update(data_cfg["ood"][split])
    sampler = lambda: sample_initial_state({**init_cfg, "dimension": config["ca"]["dimension"]}, rng)
    trajectories = rollout_batch(
        automaton=automaton,
        batch_size=int(split_cfg["num_samples"]),
        steps=int(data_cfg["trajectory_length"]) - 1,
        sampler=sampler,
    )
    if config["ca"]["dimension"] == "1d":
        trajectories = trajectories[:, :, None, :]
    else:
        trajectories = trajectories[:, :, None, :, :]
    metadata = {
        "rule": config["ca"],
        "trajectory_length": int(data_cfg["trajectory_length"]),
        "initial_condition": init_cfg,
        "split": split,
    }
    return trajectories.astype(np.float32), metadata


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    model_cfg = config["model"]
    name = model_cfg["name"]
    if name == "pixel_predictor":
        return PixelPredictor(
            dimension=config["ca"]["dimension"],
            hidden_channels=int(model_cfg.get("hidden_channels", 64)),
            depth=int(model_cfg.get("depth", 4)),
            kernel_size=int(model_cfg.get("kernel_size", 3)),
        )
    if name == "dense_world_model":
        if bool(model_cfg.get("legacy_rule184_compat", False)):
            return LegacyDenseWorldModel(
                channels=int(model_cfg.get("latent_channels", 32)),
                depth=int(model_cfg.get("depth", 3)),
                step_size=float(model_cfg.get("dynamics_step_size", 1.0)),
                use_post_norm=bool(model_cfg.get("dynamics_use_post_norm", False)),
                clamp_delta=float(model_cfg.get("dynamics_clamp_delta", 0.0)),
            )
        dynamics_norm_type = str(
            model_cfg.get(
                "dynamics_norm_type",
                "layer" if bool(model_cfg.get("dynamics_use_post_norm", False)) else "none",
            )
        )
        default_latent_type = "bottleneck" if config["ca"]["dimension"] == "1d" else "spatial"
        return DenseWorldModel(
            dimension=config["ca"]["dimension"],
            latent_type=str(model_cfg.get("latent_type", default_latent_type)),
            latent_channels=int(model_cfg.get("latent_channels", 1)),
            latent_dim=int(model_cfg.get("latent_dim", 8)),
            latent_length=int(model_cfg.get("latent_length", 8)),
            input_size=config["ca"]["size"],
            depth=int(model_cfg.get("depth", 3)),
            dynamics_depth=int(model_cfg.get("dynamics_depth", model_cfg.get("depth", 3))),
            dynamics_kernel_size=int(model_cfg.get("dynamics_kernel_size", 3)),
            dynamics_alpha=float(model_cfg.get("dynamics_alpha", model_cfg.get("dynamics_step_size", 1.0))),
            dynamics_norm_type=dynamics_norm_type,
            dynamics_init_scale=float(model_cfg.get("dynamics_init_scale", 1.0)),
            dynamics_step_size=float(model_cfg.get("dynamics_step_size", 1.0)),
            dynamics_use_post_norm=bool(model_cfg.get("dynamics_use_post_norm", False)),
            dynamics_clamp_delta=float(model_cfg.get("dynamics_clamp_delta", 0.0)),
            bottleneck_hidden_channels=int(model_cfg.get("bottleneck_hidden_channels", 32)),
            bottleneck_hidden_dim=int(model_cfg.get("bottleneck_hidden_dim", 256)),
            bottleneck_position_dim=int(model_cfg.get("bottleneck_position_dim", 32)),
            bottleneck_dynamics_hidden_dim=int(model_cfg.get("bottleneck_dynamics_hidden_dim", 128)),
        )
    if name == "object_world_model":
        if config["ca"]["dimension"] != "1d":
            raise ValueError("Object world model is currently only implemented for 1D.")
        return ObjectWorldModel(
            length=int(config["ca"]["size"]),
            num_slots=int(model_cfg.get("num_slots", 6)),
            slot_dim=int(model_cfg.get("slot_dim", 32)),
            feature_channels=int(model_cfg.get("feature_channels", 32)),
        )
    raise ValueError(f"Unsupported model name: {name}")


def build_dataloaders(config: dict[str, Any], data_dir: str | Path) -> tuple[DataLoader, DataLoader, CADataset]:
    dataset_prefix = config.get("dataset_name", config["experiment_name"])
    train_path = Path(data_dir) / f"{dataset_prefix}_train.npz"
    val_path = Path(data_dir) / f"{dataset_prefix}_val.npz"
    test_path = Path(data_dir) / f"{dataset_prefix}_test.npz"
    batch_size = int(config["train"].get("batch_size", 32))
    window = int(config["train"].get("sequence_window", 4))
    num_workers = dataloader_num_workers(config)
    train_dataset = SequenceWindowDataset.from_npz(train_path, window=window)
    val_dataset = SequenceWindowDataset.from_npz(val_path, window=window)
    test_dataset = CADataset.from_npz(test_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_dataset


def build_next_step_loader(path: str | Path, batch_size: int, num_workers: int = 0) -> DataLoader:
    dataset = NextStepDataset.from_npz(path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
