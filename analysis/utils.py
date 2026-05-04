"""Shared helpers for the analysis suite."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.ca.datasets import CADataset
from src.ca.trajectory import rollout_batch, sample_initial_state
from src.training.eval import rollout_with_optional_logits
from src.training.rollout import model_rollout
from src.utils.config import load_config
from src.utils.factory import build_automaton, build_model
from src.utils.io import ensure_dir, load_json, save_json

ROLLOUT_STEPS = (1, 2, 4, 8, 16, 30, 60)
OOD_DENSITIES = (0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9)
LOCALITY_RADII = (0, 1, 2, 3)
DEFAULT_SEED = 7


@dataclass
class RunSpec:
    config_path: Path
    checkpoint_path: Path
    config: dict[str, Any]
    output_dir: Path

    @property
    def rule(self) -> int | None:
        rule = self.config.get("ca", {}).get("rule")
        return int(rule) if rule is not None else None

    @property
    def rule_label(self) -> str:
        if self.rule is None:
            return "unknown_rule"
        return f"rule{self.rule}"

    @property
    def model_name(self) -> str:
        return str(self.config["model"]["name"])

    @property
    def run_name(self) -> str:
        return self.checkpoint_path.parent.name

    @property
    def dataset_prefix(self) -> str:
        return str(self.config.get("dataset_name", self.config["experiment_name"]))

    @property
    def dimension(self) -> str:
        return str(self.config["ca"]["dimension"])


@dataclass
class PositionwiseLatentData:
    features: np.ndarray
    time_idx: np.ndarray
    center_cell: np.ndarray
    left_neighbor: np.ndarray
    right_neighbor: np.ndarray
    next_cell: np.ndarray
    neighborhood_class: np.ndarray
    local_density: np.ndarray
    local_current: np.ndarray | None


def log(message: str) -> None:
    print(f"[analysis] {message}", flush=True)


def save_skip(path: str | Path, reason: str) -> None:
    save_json({"status": "skipped", "reason": reason}, path)


def load_any_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return load_json(path)
    return load_config(path)


def resolve_config_for_checkpoint(checkpoint_path: Path) -> Path | None:
    candidate = checkpoint_path.parent / "config.json"
    if candidate.exists():
        return candidate
    return None


def checkpoint_run_name(checkpoint_path: Path) -> str:
    return checkpoint_path.parent.name if checkpoint_path.suffix == ".ckpt" else checkpoint_path.stem


def build_run_spec(config_path: str | Path, checkpoint_path: str | Path, output_root: str | Path) -> RunSpec:
    raw_checkpoint_path = Path(checkpoint_path)
    raw_config_path = Path(config_path)
    checkpoint_path = raw_checkpoint_path.resolve()
    config_path = raw_config_path.resolve()
    config = load_any_config(config_path)
    run_name = (
        "analysis_v1"
        if "best_models" in raw_checkpoint_path.parts or "best_models" in raw_config_path.parts
        else checkpoint_run_name(checkpoint_path)
    )
    output_dir = (
        Path(output_root).resolve()
        / f"rule{config['ca'].get('rule', 'unknown')}"
        / str(config["model"]["name"])
        / run_name
    )
    return RunSpec(config_path=config_path, checkpoint_path=checkpoint_path, config=config, output_dir=output_dir)


def discover_run_specs(
    config_paths: list[str] | None,
    checkpoint_paths: list[str] | None,
    search_roots: list[str] | None,
    output_root: str | Path,
    rules: set[int] | None = None,
    checkpoint_name: str = "best.ckpt",
) -> list[RunSpec]:
    configs = [Path(path).resolve() for path in (config_paths or [])]
    checkpoints = [Path(path).resolve() for path in (checkpoint_paths or [])]
    specs: list[RunSpec] = []

    if checkpoints:
        if configs and len(configs) not in {1, len(checkpoints)}:
            raise ValueError("Provide either one config for all checkpoints or one config per checkpoint.")
        explicit_pairs: list[tuple[Path, Path]] = []
        for index, checkpoint_path in enumerate(checkpoints):
            if not checkpoint_path.exists():
                log(f"Skipping missing checkpoint: {checkpoint_path}")
                continue
            if configs:
                config_path = configs[0] if len(configs) == 1 else configs[index]
            else:
                config_path = resolve_config_for_checkpoint(checkpoint_path)
                if config_path is None:
                    log(f"Skipping checkpoint without sibling config.json: {checkpoint_path}")
                    continue
            explicit_pairs.append((config_path, checkpoint_path))
        for config_path, checkpoint_path in explicit_pairs:
            if not config_path.exists():
                log(f"Skipping missing config: {config_path}")
                continue
            specs.append(build_run_spec(config_path, checkpoint_path, output_root))
    elif configs:
        for config_path in configs:
            checkpoint_path = (config_path.parent / checkpoint_name) if config_path.suffix.lower() == ".json" else None
            if checkpoint_path is None or not checkpoint_path.exists():
                log(f"Skipping config without nearby {checkpoint_name}: {config_path}")
                continue
            specs.append(build_run_spec(config_path, checkpoint_path, output_root))
    else:
        for search_root in [Path(path).resolve() for path in (search_roots or ["outputs"])]:
            if not search_root.exists():
                log(f"Skipping missing search root: {search_root}")
                continue
            for checkpoint_path in sorted(search_root.rglob(checkpoint_name)):
                config_path = resolve_config_for_checkpoint(checkpoint_path)
                if config_path is None:
                    continue
                specs.append(build_run_spec(config_path, checkpoint_path, output_root))

    deduped: list[RunSpec] = []
    seen: set[tuple[Path, Path]] = set()
    for spec in specs:
        key = (spec.config_path, spec.checkpoint_path)
        if key in seen:
            continue
        seen.add(key)
        if rules is not None and spec.rule not in rules:
            continue
        deduped.append(spec)
    return deduped


def dataset_path(config: dict[str, Any], data_dir: str | Path, split: str) -> Path:
    return Path(data_dir) / f"{config.get('dataset_name', config['experiment_name'])}_{split}.npz"


def load_trajectories(config: dict[str, Any], data_dir: str | Path, split: str) -> torch.Tensor:
    path = dataset_path(config, data_dir, split)
    return CADataset.from_npz(path).trajectories.float()


def load_model_from_checkpoint(spec: RunSpec, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(spec.checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state"]
    model = build_model(spec.config).to(device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        legacy_rule184_keys = {
            "dynamics.net.0.weight",
            "dynamics.net.2.weight",
            "dynamics.net.4.weight",
            "dynamics.post_norm.weight",
        }
        if legacy_rule184_keys.issubset(state_dict.keys()):
            legacy_config = deepcopy(spec.config)
            legacy_config.setdefault("model", {})["legacy_rule184_compat"] = True
            model = build_model(legacy_config).to(device)
            model.load_state_dict(state_dict)
            model.eval()
            return model
        remapped = dict(state_dict)
        if "dynamics.net.0.weight" in remapped and "dynamics.net.1.weight" in model.state_dict():
            remapped["dynamics.net.1.weight"] = remapped.pop("dynamics.net.0.weight")
            remapped["dynamics.net.1.bias"] = remapped.pop("dynamics.net.0.bias")
        if "dynamics.net.2.weight" in remapped and "dynamics.final.weight" in model.state_dict():
            remapped["dynamics.final.weight"] = remapped.pop("dynamics.net.2.weight")
            remapped["dynamics.final.bias"] = remapped.pop("dynamics.net.2.bias")
        model.load_state_dict(remapped)
    model.eval()
    return model


def prepare_output_dir(path: str | Path) -> Path:
    return ensure_dir(path)


def save_manifest(records: list[dict[str, Any]], path: str | Path) -> None:
    if not records:
        return
    prepare_output_dir(Path(path).parent)
    pd.DataFrame(records).to_csv(path, index=False)


def sanitize_numpy(array: np.ndarray, clip: float = 1.0e4) -> np.ndarray:
    sanitized = np.asarray(array, dtype=np.float64)
    sanitized = np.nan_to_num(sanitized, nan=0.0, posinf=clip, neginf=-clip)
    return np.clip(sanitized, -clip, clip)


def temporal_count(states: torch.Tensor) -> torch.Tensor:
    return states.float().sum(dim=tuple(range(2, states.ndim)))


def temporal_density(states: torch.Tensor) -> torch.Tensor:
    return states.float().mean(dim=tuple(range(2, states.ndim)))


def default_density_for_config(config: dict[str, Any]) -> float:
    init_cfg = config.get("dataset", {}).get("initial_condition", {})
    if "p" in init_cfg:
        return float(init_cfg["p"])
    if "p_list" in init_cfg:
        values = [float(value) for value in init_cfg["p_list"]]
        return float(np.mean(values)) if values else 0.5
    if "p_range" in init_cfg:
        low, high = init_cfg["p_range"]
        return 0.5 * (float(low) + float(high))
    return 0.5


def clone_config_with_length(config: dict[str, Any], length: int) -> dict[str, Any]:
    cloned = deepcopy(config)
    cloned["ca"]["size"] = length
    init_cfg = cloned.get("dataset", {}).get("initial_condition", {})
    if init_cfg.get("dimension") == "1d" or isinstance(init_cfg.get("size"), int):
        init_cfg["size"] = length
    return cloned


def generate_trajectories(
    config: dict[str, Any],
    p: float,
    num_samples: int,
    steps: int,
    seed: int = DEFAULT_SEED,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    automaton = build_automaton(config)
    init_cfg = dict(config["dataset"]["initial_condition"])
    init_cfg["dimension"] = config["ca"]["dimension"]
    init_cfg["kind"] = "bernoulli"
    init_cfg["p"] = float(p)
    trajectories = rollout_batch(
        automaton=automaton,
        batch_size=int(num_samples),
        steps=int(steps),
        sampler=lambda: sample_initial_state(init_cfg, rng),
    )
    if config["ca"]["dimension"] == "1d":
        trajectories = trajectories[:, :, None, :]
    else:
        trajectories = trajectories[:, :, None, :, :]
    return torch.tensor(trajectories).float()


@torch.no_grad()
def rollout_main_model(
    model: torch.nn.Module,
    x0: torch.Tensor,
    steps: int,
    model_name: str,
    feedback_mode: str = "hard",
    rollout_mode: str = "latent",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if model_name == "dense_world_model" and not hasattr(model, "rollout"):
        rollout = model_rollout(model, x0, steps=steps, feedback_mode=feedback_mode, rollout_mode=rollout_mode)
        return rollout["states"], rollout.get("logits")
    return rollout_with_optional_logits(
        model,
        x0,
        steps=steps,
        model_name=model_name,
        feedback_mode=feedback_mode,
        rollout_mode=rollout_mode,
    )


def supports_arbitrary_length(config: dict[str, Any]) -> tuple[bool, str]:
    if config["ca"]["dimension"] != "1d":
        return False, "Only 1D lattice-size generalization is implemented."
    model_name = str(config["model"]["name"])
    if model_name == "pixel_predictor":
        return True, "convolutional predictor"
    if model_name == "dense_world_model" and str(config["model"].get("latent_type", "spatial")) == "spatial":
        return True, "spatial convolutional latent model"
    return False, f"Model {model_name} uses a fixed-size or non-spatial representation."


def supports_spatial_latent_analysis(model: torch.nn.Module, config: dict[str, Any]) -> tuple[bool, str]:
    if not hasattr(model, "encode"):
        return False, "Model does not expose encode()."
    if config["ca"]["dimension"] != "1d":
        return False, "Latent analysis is only implemented for 1D runs."
    model_name = str(config["model"]["name"])
    if model_name != "dense_world_model":
        return False, f"Model {model_name} does not expose a positionwise dense latent tensor."
    if str(config["model"].get("latent_type", "spatial")) != "spatial":
        return False, "Latent analysis requires a spatial latent representation."
    return True, "supported"


@torch.no_grad()
def encode_spatial_latent_sequence(
    model: torch.nn.Module,
    trajectories: torch.Tensor,
) -> tuple[torch.Tensor | None, str | None]:
    if not hasattr(model, "encode"):
        return None, "Model does not expose encode()."
    batch, steps = trajectories.shape[:2]
    flat = trajectories.reshape(batch * steps, *trajectories.shape[2:])
    latent = model.encode(flat)
    if not isinstance(latent, torch.Tensor):
        return None, "encode() did not return a tensor."
    if latent.ndim != 3:
        return None, f"Expected latent shape [batch, channels, length], got rank {latent.ndim}."
    channels, length = latent.shape[1], latent.shape[2]
    return latent.reshape(batch, steps, channels, length), None


@torch.no_grad()
def collect_positionwise_latent_data(
    model: torch.nn.Module,
    trajectories: torch.Tensor,
    rule: int | None,
    max_points: int,
    seed: int = DEFAULT_SEED,
) -> tuple[PositionwiseLatentData | None, str | None]:
    if trajectories.shape[1] < 2:
        return None, "Need at least two timesteps for latent probes."
    latent_seq, reason = encode_spatial_latent_sequence(model, trajectories[:, :-1])
    if latent_seq is None:
        return None, reason

    current = trajectories[:, :-1, 0]
    next_state = trajectories[:, 1:, 0]
    left = torch.roll(current, shifts=1, dims=-1)
    right = torch.roll(current, shifts=-1, dims=-1)
    neighborhood = (left.long() << 2) | (current.long() << 1) | right.long()
    local_density = (left.float() + current.float() + right.float()) / 3.0
    local_current = (current * (1.0 - right)).float() if rule == 184 else None

    features = sanitize_numpy(latent_seq.permute(0, 1, 3, 2).reshape(-1, latent_seq.shape[2]).cpu().numpy())
    batch_size, num_steps, width = current.shape
    time_idx = (
        torch.arange(num_steps, device=current.device)
        .view(1, num_steps, 1)
        .expand(batch_size, num_steps, width)
    )
    arrays: dict[str, np.ndarray] = {
        "time_idx": time_idx.reshape(-1).cpu().numpy().astype(np.int64),
        "center_cell": current.reshape(-1).cpu().numpy().astype(np.int64),
        "left_neighbor": left.reshape(-1).cpu().numpy().astype(np.int64),
        "right_neighbor": right.reshape(-1).cpu().numpy().astype(np.int64),
        "next_cell": next_state.reshape(-1).cpu().numpy().astype(np.int64),
        "neighborhood_class": neighborhood.reshape(-1).cpu().numpy().astype(np.int64),
        "local_density": local_density.reshape(-1).cpu().numpy().astype(np.float32),
    }
    if local_current is not None:
        arrays["local_current"] = local_current.reshape(-1).cpu().numpy().astype(np.float32)

    total_points = features.shape[0]
    if max_points < total_points:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(total_points, size=max_points, replace=False))
        features = features[indices]
        arrays = {key: value[indices] for key, value in arrays.items()}

    return (
        PositionwiseLatentData(
            features=features,
            time_idx=arrays["time_idx"],
            center_cell=arrays["center_cell"],
            left_neighbor=arrays["left_neighbor"],
            right_neighbor=arrays["right_neighbor"],
            next_cell=arrays["next_cell"],
            neighborhood_class=arrays["neighborhood_class"],
            local_density=arrays["local_density"],
            local_current=arrays.get("local_current"),
        ),
        None,
    )
