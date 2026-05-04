"""Linear probes, locality probes, and Rule 184 conservation analysis."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch

from analysis.plotting import plot_count_trajectories, plot_probe_radius_curve, plot_q_vs_count
from analysis.utils import (
    DEFAULT_SEED,
    LOCALITY_RADII,
    ROLLOUT_STEPS,
    RunSpec,
    collect_positionwise_latent_data,
    encode_spatial_latent_sequence,
    prepare_output_dir,
    rollout_main_model,
    save_skip,
    sanitize_numpy,
    supports_spatial_latent_analysis,
    temporal_count,
)
from src.utils.metrics import probs_to_binary

try:
    from sklearn.linear_model import Ridge, RidgeClassifier
except ImportError:  # pragma: no cover
    Ridge = None
    RidgeClassifier = None


def _split_arrays(features: np.ndarray, targets: np.ndarray, seed: int = DEFAULT_SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = sanitize_numpy(features)
    targets = sanitize_numpy(targets)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(features.shape[0])
    split = max(1, int(0.8 * len(indices)))
    train_idx = indices[:split]
    test_idx = indices[split:]
    if len(test_idx) == 0:
        test_idx = train_idx
    return features[train_idx], features[test_idx], targets[train_idx], targets[test_idx]


def _standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std


def _fit_classifier(features: np.ndarray, targets: np.ndarray) -> tuple[float, int, int]:
    train_x, test_x, train_y, test_y = _split_arrays(features, targets)
    train_x, test_x = _standardize(train_x, test_x)
    if RidgeClassifier is None:
        preds = np.full_like(test_y, int(np.bincount(train_y.astype(np.int64)).argmax()))
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            model = RidgeClassifier(alpha=1.0)
            model.fit(train_x, train_y)
            preds = model.predict(test_x)
    accuracy = float((preds == test_y).mean())
    return accuracy, int(train_y.shape[0]), int(test_y.shape[0])


def _fit_regressor(features: np.ndarray, targets: np.ndarray) -> tuple[float, float, int, int]:
    train_x, test_x, train_y, test_y = _split_arrays(features, targets)
    train_x, test_x = _standardize(train_x, test_x)
    if Ridge is None:
        coef, _, _, _ = np.linalg.lstsq(train_x, train_y, rcond=None)
        preds = test_x @ coef
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            model = Ridge(alpha=1.0)
            model.fit(train_x, train_y)
            preds = model.predict(test_x)
    mse = float(np.mean((preds - test_y) ** 2))
    denom = float(np.sum((test_y - np.mean(test_y, axis=0, keepdims=True)) ** 2))
    r2 = float(1.0 - np.sum((preds - test_y) ** 2) / max(denom, 1e-8))
    return mse, r2, int(train_y.shape[0]), int(test_y.shape[0])


def run_linear_probes(
    spec: RunSpec,
    model: torch.nn.Module,
    trajectories: torch.Tensor,
    output_dir: str | Path,
    max_points: int = 12000,
) -> dict[str, str]:
    output_dir = prepare_output_dir(output_dir)
    supported, reason = supports_spatial_latent_analysis(model, spec.config)
    if not supported:
        save_skip(output_dir / "summary.json", reason)
        return {"status": "skipped", "reason": reason}

    latent_data, latent_reason = collect_positionwise_latent_data(
        model=model,
        trajectories=trajectories,
        rule=spec.rule,
        max_points=max_points,
    )
    if latent_data is None:
        save_skip(output_dir / "summary.json", latent_reason or "Failed to collect latent data.")
        return {"status": "skipped", "reason": latent_reason or "latent feature collection failed"}

    rows: list[dict[str, float | int | str]] = []
    for target_name, target_values in [
        ("center_cell", latent_data.center_cell),
        ("left_neighbor", latent_data.left_neighbor),
        ("right_neighbor", latent_data.right_neighbor),
        ("next_cell", latent_data.next_cell),
        ("neighborhood_class", latent_data.neighborhood_class),
    ]:
        accuracy, train_n, test_n = _fit_classifier(latent_data.features, np.asarray(target_values))
        rows.append(
            {
                "target": target_name,
                "task": "classification",
                "accuracy": accuracy,
                "mse": np.nan,
                "r2": np.nan,
                "train_n": train_n,
                "test_n": test_n,
            }
        )

    mse, r2, train_n, test_n = _fit_regressor(latent_data.features, np.asarray(latent_data.local_density))
    rows.append(
        {
            "target": "local_density",
            "task": "regression",
            "accuracy": np.nan,
            "mse": mse,
            "r2": r2,
            "train_n": train_n,
            "test_n": test_n,
        }
    )
    if spec.rule == 184 and latent_data.local_current is not None:
        accuracy, train_n, test_n = _fit_classifier(latent_data.features, np.asarray(latent_data.local_current).astype(np.int64))
        rows.append(
            {
                "target": "local_current",
                "task": "classification",
                "accuracy": accuracy,
                "mse": np.nan,
                "r2": np.nan,
                "train_n": train_n,
                "test_n": test_n,
            }
        )

    probe_df = pd.DataFrame(rows)
    probe_df.to_csv(output_dir / "linear_probe_results.csv", index=False)
    return {"status": "ok"}


def run_locality_probe(
    spec: RunSpec,
    model: torch.nn.Module,
    trajectories: torch.Tensor,
    output_dir: str | Path,
    max_points: int = 30000,
) -> dict[str, str]:
    output_dir = prepare_output_dir(output_dir)
    supported, reason = supports_spatial_latent_analysis(model, spec.config)
    if not supported:
        save_skip(output_dir / "summary.json", reason)
        return {"status": "skipped", "reason": reason}

    latent_seq, latent_reason = encode_spatial_latent_sequence(model, trajectories)
    if latent_seq is None:
        save_skip(output_dir / "summary.json", latent_reason or "Failed to encode latent sequence.")
        return {"status": "skipped", "reason": latent_reason or "latent encoding failed"}
    if latent_seq.shape[1] < 2:
        save_skip(output_dir / "summary.json", "Need at least two timesteps for locality probe.")
        return {"status": "skipped", "reason": "Need at least two timesteps for locality probe."}

    x_latent = sanitize_numpy(latent_seq[:, :-1].detach().cpu().numpy())
    y_latent = sanitize_numpy(latent_seq[:, 1:].detach().cpu().numpy())
    rows: list[dict[str, float | int]] = []

    for radius in LOCALITY_RADII:
        neighborhoods = [np.roll(x_latent, shift=offset, axis=-1) for offset in range(-radius, radius + 1)]
        features = np.concatenate(neighborhoods, axis=2).transpose(0, 1, 3, 2).reshape(-1, x_latent.shape[2] * (2 * radius + 1))
        targets = y_latent.transpose(0, 1, 3, 2).reshape(-1, y_latent.shape[2])
        if max_points < features.shape[0]:
            rng = np.random.default_rng(DEFAULT_SEED)
            indices = np.sort(rng.choice(features.shape[0], size=max_points, replace=False))
            features = features[indices]
            targets = targets[indices]
        mse, r2, train_n, test_n = _fit_regressor(features, targets)
        rows.append({"radius": radius, "mse": mse, "r2": r2, "train_n": train_n, "test_n": test_n})

    locality_df = pd.DataFrame(rows)
    locality_df.to_csv(output_dir / "locality_probe_results.csv", index=False)
    plot_probe_radius_curve(locality_df, output_dir / "locality_probe_radius_curve.png", "Latent update locality probe")
    return {"status": "ok"}


def run_rule184_conservation_test(
    spec: RunSpec,
    model: torch.nn.Module,
    trajectories: torch.Tensor,
    device: torch.device,
    output_dir: str | Path,
    max_steps: int = max(ROLLOUT_STEPS),
) -> dict[str, str]:
    output_dir = prepare_output_dir(output_dir)
    if spec.rule != 184:
        save_skip(output_dir / "summary.json", "Rule 184 conservation test only applies to Rule 184.")
        return {"status": "skipped", "reason": "Rule 184 only"}
    supported, reason = supports_spatial_latent_analysis(model, spec.config)
    if not supported:
        save_skip(output_dir / "summary.json", reason)
        return {"status": "skipped", "reason": reason}

    horizon = min(max_steps, trajectories.shape[1] - 1)
    sequence = trajectories[:, : horizon + 1].to(device)
    latent_seq, latent_reason = encode_spatial_latent_sequence(model, sequence)
    if latent_seq is None:
        save_skip(output_dir / "summary.json", latent_reason or "Failed to encode latent sequence.")
        return {"status": "skipped", "reason": latent_reason or "latent encoding failed"}

    pred_states, _ = rollout_main_model(
        model,
        sequence[:, 0],
        steps=horizon,
        model_name=spec.model_name,
        feedback_mode="hard",
        rollout_mode="latent",
    )
    pred_states = pred_states[:, : horizon + 1]

    if not hasattr(model, "step_latent"):
        save_skip(output_dir / "summary.json", "Model does not expose step_latent().")
        return {"status": "skipped", "reason": "Missing step_latent()"}

    z_rollout = [latent_seq[:, 0]]
    z_current = latent_seq[:, 0]
    for _ in range(horizon):
        z_current = model.step_latent(z_current)
        z_rollout.append(z_current)
    z_rollout_tensor = torch.stack(z_rollout, dim=1)

    true_counts = temporal_count(sequence).detach().cpu().numpy()
    pred_counts = temporal_count(probs_to_binary(pred_states)).detach().cpu().numpy()
    latent_sum_true = sanitize_numpy(latent_seq.detach().cpu().numpy().sum(axis=-1).reshape(-1, latent_seq.shape[2]))
    latent_sum_rollout = sanitize_numpy(z_rollout_tensor.detach().cpu().numpy().sum(axis=-1).reshape(-1, z_rollout_tensor.shape[2]))
    target_counts = true_counts.reshape(-1)

    mse, r2, _, _ = _fit_regressor(latent_sum_true, target_counts)
    if Ridge is None:
        coef, _, _, _ = np.linalg.lstsq(latent_sum_true, target_counts, rcond=None)

        def predict_q(x: np.ndarray) -> np.ndarray:
            return x @ coef
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            regressor = Ridge(alpha=1.0)
            regressor.fit(latent_sum_true, target_counts)

        def predict_q(x: np.ndarray) -> np.ndarray:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return regressor.predict(x)

    q_true = predict_q(latent_sum_true).reshape(true_counts.shape)
    q_rollout = predict_q(latent_sum_rollout).reshape(true_counts.shape)

    summary = pd.DataFrame(
        [
            {
                "q_fit_mse": mse,
                "q_fit_r2": r2,
                "true_count_drift_mean_abs": float(np.mean(np.abs(true_counts - true_counts[:, :1]))),
                "pred_count_drift_mean_abs": float(np.mean(np.abs(pred_counts - pred_counts[:, :1]))),
                "q_true_drift_mean_abs": float(np.mean(np.abs(q_true - q_true[:, :1]))),
                "q_rollout_drift_mean_abs": float(np.mean(np.abs(q_rollout - q_rollout[:, :1]))),
                "pred_count_mae": float(np.mean(np.abs(pred_counts - true_counts))),
            }
        ]
    )
    summary.to_csv(output_dir / "rule184_conservation_summary.csv", index=False)
    plot_count_trajectories(true_counts, pred_counts, output_dir / "rule184_particle_count_rollout.png", "Rule 184 particle count conservation")
    plot_q_vs_count(true_counts.mean(axis=0), q_true.mean(axis=0), q_rollout.mean(axis=0), output_dir / "rule184_latent_q_vs_count.png", "Rule 184 conserved latent quantity")
    pd.DataFrame(
        {
            "step": np.arange(true_counts.shape[1]),
            "true_count_mean": true_counts.mean(axis=0),
            "pred_count_mean": pred_counts.mean(axis=0),
            "q_true_mean": q_true.mean(axis=0),
            "q_rollout_mean": q_rollout.mean(axis=0),
        }
    ).to_csv(output_dir / "rule184_conservation_timeseries.csv", index=False)
    return {"status": "ok"}
