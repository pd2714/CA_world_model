"""Unsupervised latent-physics discovery for 1D CA world models."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.plotting import set_analysis_style
from analysis.utils import (
    DEFAULT_SEED,
    RunSpec,
    discover_run_specs,
    encode_spatial_latent_sequence,
    load_model_from_checkpoint,
    load_trajectories,
    log,
    prepare_output_dir,
    sanitize_numpy,
    save_manifest,
    save_skip,
    supports_spatial_latent_analysis,
)
from src.utils.io import save_json
from src.utils.plotting import save_figure
from src.utils.runtime import configure_runtime
from src.utils.seed import set_seed

try:
    from sklearn.cluster import KMeans
    from sklearn.linear_model import Ridge, RidgeClassifier
    from sklearn.metrics import normalized_mutual_info_score
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover - analysis dependency guard
    raise RuntimeError("latent_discovery.py requires scikit-learn to be installed.") from exc


CLUSTER_FIT_CAP = 60000
GMM_FIT_CAP = 20000
VALIDATION_LOCAL_CAP = 60000
EPS = 1.0e-6


@dataclass
class SymbolicDiscoveryResult:
    row: dict[str, Any]
    symbols: np.ndarray


@dataclass
class ConservedQuantityResult:
    row: dict[str, Any]
    q_local: np.ndarray
    Q: np.ndarray


@dataclass
class SlowVariableResult:
    row: dict[str, Any]
    g: np.ndarray


@dataclass
class ContinuityDiscoveryResult:
    row: dict[str, Any]
    rho: np.ndarray
    J: np.ndarray


@dataclass
class LabelBundle:
    states: np.ndarray
    next_states: np.ndarray
    neighborhood: np.ndarray
    rule184_current: np.ndarray
    particle_number: np.ndarray


def _local_latent_numpy(latent_seq: torch.Tensor) -> np.ndarray:
    return sanitize_numpy(latent_seq.detach().cpu().permute(0, 1, 3, 2).numpy(), clip=100.0)


def _local_latent_torch(latent_seq: torch.Tensor) -> torch.Tensor:
    return latent_seq.permute(0, 1, 3, 2).contiguous()


def _select_indices(total: int, max_points: int, seed: int) -> np.ndarray:
    if total <= max_points:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=max_points, replace=False))


def _build_local_windows_numpy(local_latent: np.ndarray, radius: int) -> np.ndarray:
    parts = [np.roll(local_latent, shift=-offset, axis=2) for offset in range(-radius, radius + 1)]
    return np.concatenate(parts, axis=-1)


def _build_local_windows_torch(local_latent: torch.Tensor, radius: int) -> torch.Tensor:
    parts = [torch.roll(local_latent, shifts=-offset, dims=2) for offset in range(-radius, radius + 1)]
    return torch.cat(parts, dim=-1)


def _build_context_ids(symbols: np.ndarray, radius: int, base: int) -> np.ndarray:
    context_ids = np.zeros_like(symbols, dtype=np.int64)
    factor = 1
    for offset in range(-radius, radius + 1):
        context_ids += factor * np.roll(symbols, shift=-offset, axis=-1).astype(np.int64)
        factor *= int(base)
    return context_ids


def _entropy_from_counts(counts: np.ndarray) -> float:
    counts = counts.astype(np.float64)
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0
    probs = counts[counts > 0.0] / total
    return float(-np.sum(probs * np.log2(probs)))


def _normalized_entropy(counts: np.ndarray, cardinality: int) -> float:
    denom = np.log2(max(2, int(cardinality)))
    return float(_entropy_from_counts(counts) / denom)


def _held_out_empirical_transition_metrics(
    symbols: np.ndarray,
    num_symbols: int,
    transition_radius: int,
    seed: int,
) -> dict[str, float]:
    context_ids = _build_context_ids(symbols[:, :-1], radius=transition_radius, base=num_symbols).reshape(-1)
    targets = symbols[:, 1:].reshape(-1).astype(np.int64)
    rng = np.random.default_rng(seed)
    order = rng.permutation(context_ids.shape[0])
    split = max(1, int(0.8 * len(order)))
    train_idx = order[:split]
    test_idx = order[split:]
    if test_idx.size == 0:
        test_idx = train_idx

    train_ids = context_ids[train_idx]
    train_targets = targets[train_idx]
    test_ids = context_ids[test_idx]
    test_targets = targets[test_idx]

    unique_ids, inverse = np.unique(train_ids, return_inverse=True)
    counts = np.zeros((unique_ids.shape[0], num_symbols), dtype=np.int64)
    np.add.at(counts, (inverse, train_targets), 1)
    majority = counts.argmax(axis=1)
    global_majority = int(np.bincount(train_targets, minlength=num_symbols).argmax())

    lookup = np.searchsorted(unique_ids, test_ids)
    seen_mask = lookup < unique_ids.shape[0]
    valid_lookup = lookup[seen_mask]
    seen_mask[seen_mask] = unique_ids[valid_lookup] == test_ids[seen_mask]
    preds = np.full(test_targets.shape[0], global_majority, dtype=np.int64)
    preds[seen_mask] = majority[lookup[seen_mask]]

    context_totals = counts.sum(axis=1, keepdims=True)
    probs = counts / np.clip(context_totals, 1, None)
    positive = probs > 0.0
    log_probs = np.zeros_like(probs, dtype=np.float64)
    log_probs[positive] = np.log2(probs[positive])
    conditional_entropies = -np.sum(probs * log_probs, axis=1)
    weights = context_totals[:, 0].astype(np.float64)
    weighted_entropy = float(np.sum(weights * conditional_entropies) / max(float(weights.sum()), 1.0))
    determinism = float(np.sum(counts.max(axis=1)) / max(float(counts.sum()), 1.0))

    target_counts = np.bincount(targets, minlength=num_symbols)
    return {
        "prediction_accuracy": float((preds == test_targets).mean()),
        "baseline_accuracy": float(np.mean(test_targets == global_majority)),
        "transition_determinism": determinism,
        "transition_entropy_bits": weighted_entropy,
        "transition_entropy_normalized": float(weighted_entropy / np.log2(max(2, num_symbols))),
        "novel_context_frac": float(1.0 - seen_mask.mean()),
        "num_contexts_train": int(unique_ids.shape[0]),
        "target_entropy_bits": _entropy_from_counts(target_counts),
        "target_entropy_normalized": _normalized_entropy(target_counts, num_symbols),
    }


def _select_minimal_useful_symbolic(df: pd.DataFrame) -> int:
    if df.empty:
        raise ValueError("Cannot select a symbolic candidate from an empty dataframe.")
    best_accuracy = float(df["prediction_accuracy"].max())
    gain_target = float(df["baseline_accuracy"].max()) + 0.02
    threshold = max(best_accuracy - 0.02, gain_target)
    candidates = df[df["prediction_accuracy"] >= threshold].copy()
    if candidates.empty:
        candidates = df[df["prediction_accuracy"] == best_accuracy].copy()
    candidates = candidates.sort_values(
        ["num_symbols", "feature_radius", "transition_radius", "prediction_accuracy", "cluster_method"],
        ascending=[True, True, True, False, True],
    )
    return int(candidates.index[0])


def _plot_symbolic_spacetime(symbols: np.ndarray, num_symbols: int, path: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    image = ax.imshow(symbols, aspect="auto", cmap="tab20", vmin=0, vmax=max(1, num_symbols - 1))
    ax.set_xlabel("Position")
    ax.set_ylabel("Time")
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Symbol")
    save_figure(fig, path)


def _plot_timeseries(values: np.ndarray, path: str | Path, title: str, ylabel: str) -> None:
    steps = np.arange(values.shape[1])
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for idx in range(min(8, values.shape[0])):
        ax.plot(steps, values[idx], color="tab:blue", alpha=0.18, linewidth=1.0)
    ax.plot(steps, values.mean(axis=0), color="tab:blue", linewidth=2.4, label="Mean")
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    save_figure(fig, path)


def _plot_multivariate_timeseries(values: np.ndarray, path: str | Path, title: str) -> None:
    steps = np.arange(values.shape[1])
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for dim in range(values.shape[2]):
        for idx in range(min(4, values.shape[0])):
            ax.plot(steps, values[idx, :, dim], alpha=0.12, linewidth=1.0)
        ax.plot(steps, values.mean(axis=0)[:, dim], linewidth=2.0, label=f"g{dim}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(frameon=False)
    save_figure(fig, path)


def _plot_scalar_spacetime(field: np.ndarray, path: str | Path, title: str, cmap: str = "coolwarm") -> None:
    vmax = float(np.max(np.abs(field))) if field.size else 1.0
    vmax = max(vmax, 1.0e-6)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    image = ax.imshow(field, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Position")
    ax.set_ylabel("Time")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    save_figure(fig, path)


def _build_labels(trajectories: torch.Tensor) -> LabelBundle:
    states = trajectories[:, :, 0].detach().cpu().numpy().astype(np.int64)
    left = np.roll(states, shift=1, axis=-1)
    right = np.roll(states, shift=-1, axis=-1)
    neighborhood = (left << 2) | (states << 1) | right
    next_states = states[:, 1:]
    current_formula = (states * (1 - right)).astype(np.int64)
    particle_number = states.sum(axis=-1).astype(np.float64)
    return LabelBundle(
        states=states,
        next_states=next_states,
        neighborhood=neighborhood.astype(np.int64),
        rule184_current=current_formula,
        particle_number=particle_number,
    )


class _LocalScalarHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        if hidden_dim is None:
            self.net = nn.Linear(input_dim, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _GlobalHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        if hidden_dim is None:
            self.net = nn.Linear(input_dim, output_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _normalize_field(x: torch.Tensor, dims: tuple[int, ...] | None = None) -> torch.Tensor:
    if dims is None:
        mean = x.mean()
        std = x.std(unbiased=False)
    else:
        mean = x.mean(dim=dims, keepdim=True)
        std = x.std(dim=dims, unbiased=False, keepdim=True)
    return (x - mean) / (std + EPS)


def _train_module(
    module: nn.Module,
    loss_fn: Callable[[], torch.Tensor],
    epochs: int,
    lr: float,
    weight_decay: float,
) -> float:
    optimizer = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss = float("inf")
    best_state = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
    patience = max(20, epochs // 5)
    steps_without_improvement = 0
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().item())
        if loss_value + 1.0e-6 < best_loss:
            best_loss = loss_value
            best_state = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1
            if steps_without_improvement >= patience:
                break
    module.load_state_dict(best_state)
    return best_loss


def run_symbolic_discovery(
    latent_seq: torch.Tensor,
    output_dir: str | Path,
    cluster_ks: list[int],
    radii: list[int],
    seed: int,
) -> tuple[pd.DataFrame, SymbolicDiscoveryResult]:
    output_dir = prepare_output_dir(output_dir)
    local_latent = _local_latent_numpy(latent_seq)
    rows: list[dict[str, Any]] = []
    assignments: dict[tuple[str, int, int], np.ndarray] = {}

    for cluster_method in ("kmeans", "gmm"):
        for feature_radius in radii:
            features = _build_local_windows_numpy(local_latent, feature_radius).reshape(-1, local_latent.shape[-1] * (2 * feature_radius + 1))
            fit_cap = GMM_FIT_CAP if cluster_method == "gmm" else CLUSTER_FIT_CAP
            fit_idx = _select_indices(features.shape[0], fit_cap, seed + 17 * feature_radius + (101 if cluster_method == "gmm" else 0))
            scaler = StandardScaler()
            fit_features = np.clip(scaler.fit_transform(features[fit_idx]), -8.0, 8.0)
            all_features = np.clip(scaler.transform(features), -8.0, 8.0)

            for num_symbols in cluster_ks:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    if cluster_method == "kmeans":
                        clusterer = KMeans(n_clusters=num_symbols, n_init=10, random_state=seed)
                        clusterer.fit(fit_features)
                        symbol_flat = clusterer.predict(all_features)
                    else:
                        clusterer = GaussianMixture(
                            n_components=num_symbols,
                            covariance_type="diag",
                            reg_covar=1.0e-6,
                            random_state=seed,
                        )
                        clusterer.fit(fit_features)
                        symbol_flat = clusterer.predict(all_features)

                symbols = symbol_flat.reshape(local_latent.shape[:3]).astype(np.int64)
                assignments[(cluster_method, feature_radius, num_symbols)] = symbols
                symbol_counts = np.bincount(symbol_flat, minlength=num_symbols)
                _plot_symbolic_spacetime(
                    symbols[0],
                    num_symbols=num_symbols,
                    path=output_dir / f"{cluster_method}_K{num_symbols}_feature_r{feature_radius}_sample0.png",
                    title=f"{cluster_method.upper()} symbols, K={num_symbols}, feature r={feature_radius}",
                )

                for transition_radius in radii:
                    metrics = _held_out_empirical_transition_metrics(
                        symbols=symbols,
                        num_symbols=num_symbols,
                        transition_radius=transition_radius,
                        seed=seed + 31 * transition_radius,
                    )
                    rows.append(
                        {
                            "cluster_method": cluster_method,
                            "feature_radius": int(feature_radius),
                            "transition_radius": int(transition_radius),
                            "num_symbols": int(num_symbols),
                            "symbol_entropy_bits": _entropy_from_counts(symbol_counts),
                            "symbol_entropy_normalized": _normalized_entropy(symbol_counts, num_symbols),
                            **metrics,
                        }
                    )

    summary_df = pd.DataFrame(rows).sort_values(
        ["prediction_accuracy", "transition_determinism", "num_symbols", "feature_radius", "transition_radius"],
        ascending=[False, False, True, True, True],
    )
    best_idx = _select_minimal_useful_symbolic(summary_df)
    summary_df["is_minimal_useful"] = False
    summary_df.loc[best_idx, "is_minimal_useful"] = True
    summary_df.to_csv(output_dir / "symbolic_transition_summary.csv", index=False)

    best_row = summary_df.loc[best_idx].to_dict()
    best_symbols = assignments[(str(best_row["cluster_method"]), int(best_row["feature_radius"]), int(best_row["num_symbols"]))]
    save_json(best_row, output_dir / "best_symbolic_candidate.json")
    return summary_df, SymbolicDiscoveryResult(row=best_row, symbols=best_symbols)


def run_conserved_quantity_discovery(
    latent_seq: torch.Tensor,
    output_dir: str | Path,
    families: list[str],
) -> tuple[pd.DataFrame, ConservedQuantityResult]:
    output_dir = prepare_output_dir(output_dir)
    local_latent = _local_latent_torch(latent_seq)
    rows: list[dict[str, Any]] = []
    results: dict[str, ConservedQuantityResult] = {}

    for family in families:
        hidden_dim = None if family == "linear" else 64
        model = _LocalScalarHead(local_latent.shape[-1], hidden_dim=hidden_dim).to(latent_seq.device)

        def loss_fn() -> torch.Tensor:
            q_raw = model(local_latent)
            q = _normalize_field(q_raw)
            Q = q.sum(dim=-1)
            drift = (Q[:, 1:] - Q[:, :-1]).pow(2).mean()
            Q_var = Q.var(unbiased=False)
            return drift / (Q_var + EPS) + 0.02 / (Q_var + EPS)

        _train_module(
            module=model,
            loss_fn=loss_fn,
            epochs=140 if family == "linear" else 180,
            lr=1.0e-2 if family == "linear" else 5.0e-3,
            weight_decay=1.0e-4,
        )

        with torch.no_grad():
            q_raw = model(local_latent)
            q = _normalize_field(q_raw)
            Q = q.sum(dim=-1)
            drift = Q[:, 1:] - Q[:, :-1]
            Q_var = float(Q.var(unbiased=False).item())
            score = float((drift.pow(2).mean() / (Q.var(unbiased=False) + EPS)).item())
            row = {
                "family": family,
                "conservation_score": score,
                "Q_variance": Q_var,
                "local_q_variance": float(q.var(unbiased=False).item()),
                "mean_abs_delta_Q": float(drift.abs().mean().item()),
            }
            q_np = q.detach().cpu().numpy()
            Q_np = Q.detach().cpu().numpy()

        pd.DataFrame(
            {
                "time": np.arange(Q_np.shape[1]),
                "Q_mean": Q_np.mean(axis=0),
                "Q_std": Q_np.std(axis=0),
            }
        ).to_csv(output_dir / f"{family}_Q_timeseries.csv", index=False)
        _plot_timeseries(Q_np, output_dir / f"{family}_Q_timeseries.png", f"Discovered conserved quantity ({family})", "Q(t)")
        _plot_scalar_spacetime(q_np[0], output_dir / f"{family}_q_spacetime_sample0.png", f"Local q(z) sample 0 ({family})")

        rows.append(row)
        results[family] = ConservedQuantityResult(row=row, q_local=q_np, Q=Q_np)

    summary_df = pd.DataFrame(rows).sort_values(["conservation_score", "family"], ascending=[True, True])
    summary_df.to_csv(output_dir / "conservation_summary.csv", index=False)
    best_family = str(summary_df.iloc[0]["family"])
    save_json(results[best_family].row, output_dir / "best_conserved_quantity.json")
    return summary_df, results[best_family]


def run_slow_variable_discovery(
    latent_seq: torch.Tensor,
    output_dir: str | Path,
    families: list[str],
    output_dim: int = 2,
) -> tuple[pd.DataFrame, SlowVariableResult]:
    output_dir = prepare_output_dir(output_dir)
    global_latent = latent_seq.reshape(latent_seq.shape[0], latent_seq.shape[1], -1)
    rows: list[dict[str, Any]] = []
    results: dict[str, SlowVariableResult] = {}

    for family in families:
        hidden_dim = None if family == "linear" else 128
        model = _GlobalHead(global_latent.shape[-1], output_dim=output_dim, hidden_dim=hidden_dim).to(latent_seq.device)

        def loss_fn() -> torch.Tensor:
            g_raw = model(global_latent)
            g = _normalize_field(g_raw, dims=(0, 1))
            delta = g[:, 1:] - g[:, :-1]
            flat = g.reshape(-1, g.shape[-1])
            cov = flat.T @ flat / max(flat.shape[0], 1)
            eye = torch.eye(cov.shape[0], device=cov.device)
            offdiag = cov - eye
            offdiag = offdiag * (1.0 - eye)
            raw_var = g_raw.var(unbiased=False)
            return delta.pow(2).mean() + 0.1 * offdiag.pow(2).mean() + 0.02 / (raw_var + EPS)

        _train_module(
            module=model,
            loss_fn=loss_fn,
            epochs=160 if family == "linear" else 220,
            lr=8.0e-3 if family == "linear" else 4.0e-3,
            weight_decay=1.0e-4,
        )

        with torch.no_grad():
            g_raw = model(global_latent)
            g = _normalize_field(g_raw, dims=(0, 1))
            delta = g[:, 1:] - g[:, :-1]
            score = float(delta.pow(2).mean().item())
            row = {
                "family": family,
                "slowness_score": score,
                "raw_variance": float(g_raw.var(unbiased=False).item()),
                "mean_abs_delta": float(delta.abs().mean().item()),
            }
            g_np = g.detach().cpu().numpy()

        pd.DataFrame(
            {
                "time": np.tile(np.arange(g_np.shape[1]), g_np.shape[2]),
                "component": np.repeat(np.arange(g_np.shape[2]), g_np.shape[1]),
                "mean_value": np.concatenate([g_np.mean(axis=0)[:, dim] for dim in range(g_np.shape[2])]),
            }
        ).to_csv(output_dir / f"{family}_slow_variables_timeseries.csv", index=False)
        _plot_multivariate_timeseries(g_np, output_dir / f"{family}_slow_variables_timeseries.png", f"Discovered slow variables ({family})")

        rows.append(row)
        results[family] = SlowVariableResult(row=row, g=g_np)

    summary_df = pd.DataFrame(rows).sort_values(["slowness_score", "family"], ascending=[True, True])
    summary_df.to_csv(output_dir / "slow_variable_summary.csv", index=False)
    best_family = str(summary_df.iloc[0]["family"])
    save_json(results[best_family].row, output_dir / "best_slow_variables.json")
    return summary_df, results[best_family]


def run_continuity_discovery(
    latent_seq: torch.Tensor,
    output_dir: str | Path,
    radii: list[int],
    families: list[str],
) -> tuple[pd.DataFrame, ContinuityDiscoveryResult]:
    output_dir = prepare_output_dir(output_dir)
    local_latent = _local_latent_torch(latent_seq)
    rows: list[dict[str, Any]] = []
    results: dict[tuple[str, int], ContinuityDiscoveryResult] = {}

    for family in families:
        hidden_dim = None if family == "linear" else 64
        for radius in radii:
            rho_head = _LocalScalarHead(local_latent.shape[-1], hidden_dim=hidden_dim).to(latent_seq.device)
            j_head = _LocalScalarHead(local_latent.shape[-1] * (2 * radius + 1), hidden_dim=hidden_dim).to(latent_seq.device)
            current_window = _build_local_windows_torch(local_latent, radius)

            def loss_fn() -> torch.Tensor:
                rho_raw = rho_head(local_latent)
                J_raw = j_head(current_window)
                rho = _normalize_field(rho_raw)
                J = _normalize_field(J_raw)
                residual = (rho[:, 1:] - rho[:, :-1]) - (torch.roll(J[:, :-1], shifts=1, dims=-1) - J[:, :-1])
                rho_var = rho_raw.var(unbiased=False)
                J_var = J_raw.var(unbiased=False)
                return residual.pow(2).mean() + 0.02 / (rho_var + EPS) + 0.02 / (J_var + EPS)

            _train_module(
                module=nn.ModuleDict({"rho": rho_head, "J": j_head}),
                loss_fn=loss_fn,
                epochs=140 if family == "linear" else 180,
                lr=8.0e-3 if family == "linear" else 4.0e-3,
                weight_decay=1.0e-4,
            )

            with torch.no_grad():
                rho_raw = rho_head(local_latent)
                J_raw = j_head(current_window)
                rho = _normalize_field(rho_raw)
                J = _normalize_field(J_raw)
                residual = (rho[:, 1:] - rho[:, :-1]) - (torch.roll(J[:, :-1], shifts=1, dims=-1) - J[:, :-1])
                row = {
                    "family": family,
                    "radius": int(radius),
                    "continuity_residual_mse": float(residual.pow(2).mean().item()),
                    "rho_variance": float(rho.var(unbiased=False).item()),
                    "J_variance": float(J.var(unbiased=False).item()),
                    "mean_abs_residual": float(residual.abs().mean().item()),
                }
                rho_np = rho.detach().cpu().numpy()
                J_np = J.detach().cpu().numpy()

            rows.append(row)
            results[(family, radius)] = ContinuityDiscoveryResult(row=row, rho=rho_np, J=J_np)

    summary_df = pd.DataFrame(rows).sort_values(
        ["continuity_residual_mse", "radius", "family"],
        ascending=[True, True, True],
    )
    threshold = float(summary_df["continuity_residual_mse"].min()) + 0.02
    candidates = summary_df[summary_df["continuity_residual_mse"] <= threshold].sort_values(
        ["radius", "continuity_residual_mse", "family"],
        ascending=[True, True, True],
    )
    minimal_useful_idx = int(candidates.iloc[0].name)
    summary_df["is_minimal_useful"] = False
    summary_df.loc[minimal_useful_idx, "is_minimal_useful"] = True
    summary_df.to_csv(output_dir / "continuity_summary.csv", index=False)

    best_row = summary_df.loc[minimal_useful_idx].to_dict()
    best_result = results[(str(best_row["family"]), int(best_row["radius"]))]
    _plot_scalar_spacetime(best_result.rho[0], output_dir / "best_rho_spacetime_sample0.png", "Discovered rho(z) sample 0")
    _plot_scalar_spacetime(best_result.J[0], output_dir / "best_current_spacetime_sample0.png", "Discovered J(z_window) sample 0")
    save_json(best_row, output_dir / "best_continuity_candidate.json")
    return summary_df, best_result


def _split_arrays(
    features: np.ndarray,
    targets: np.ndarray,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(features.shape[0])
    split = max(1, int(0.8 * len(indices)))
    train_idx = indices[:split]
    test_idx = indices[split:]
    if test_idx.size == 0:
        test_idx = train_idx
    return features[train_idx], features[test_idx], targets[train_idx], targets[test_idx]


def _subsample_validation(features: np.ndarray, targets: np.ndarray, cap: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if features.shape[0] <= cap:
        return features, targets
    indices = _select_indices(features.shape[0], cap, seed)
    return features[indices], targets[indices]


def _standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1.0e-6] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std


def _classifier_accuracy(features: np.ndarray, targets: np.ndarray, seed: int) -> float:
    train_x, test_x, train_y, test_y = _split_arrays(features, targets, seed=seed)
    train_x, test_x = _standardize(train_x, test_x)
    if np.unique(train_y).size <= 1:
        preds = np.full_like(test_y, train_y[0])
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            model = RidgeClassifier(alpha=1.0)
            model.fit(train_x, train_y)
            preds = model.predict(test_x)
    return float((preds == test_y).mean())


def _regression_metrics(features: np.ndarray, targets: np.ndarray, seed: int) -> tuple[float, float]:
    train_x, test_x, train_y, test_y = _split_arrays(features, targets, seed=seed)
    train_x, test_x = _standardize(train_x, test_x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model = Ridge(alpha=1.0)
        model.fit(train_x, train_y)
        preds = model.predict(test_x)
    ss_res = float(np.sum((preds - test_y) ** 2))
    ss_tot = float(np.sum((test_y - test_y.mean()) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1.0e-8))
    corr = _pearson_corr(preds, test_y)
    return r2, corr


def _majority_map_accuracy(symbols: np.ndarray, labels: np.ndarray) -> float:
    accuracy = 0.0
    total = max(symbols.shape[0], 1)
    for symbol in np.unique(symbols):
        mask = symbols == symbol
        values, counts = np.unique(labels[mask], return_counts=True)
        accuracy += float(counts.max())
    return float(accuracy / total)


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom <= 1.0e-12:
        return 0.0
    return float(np.sum(x * y) / denom)


def run_posthoc_validation(
    labels: LabelBundle,
    symbolic: SymbolicDiscoveryResult,
    conserved: ConservedQuantityResult,
    slow: SlowVariableResult,
    continuity: ContinuityDiscoveryResult,
    output_dir: str | Path,
    seed: int,
) -> pd.DataFrame:
    output_dir = prepare_output_dir(output_dir)
    rows: list[dict[str, Any]] = []

    local_x = labels.states.reshape(-1)
    local_neighborhood = labels.neighborhood.reshape(-1)
    local_current = labels.rule184_current.reshape(-1)
    next_x = labels.next_states.reshape(-1)
    particle_number = labels.particle_number.reshape(-1)

    symbol_flat = symbolic.symbols.reshape(-1)
    symbol_now = symbolic.symbols[:, :-1].reshape(-1)
    symbol_counts = np.eye(int(symbolic.row["num_symbols"]), dtype=np.float32)[symbol_flat].reshape(-1, int(symbolic.row["num_symbols"]))
    symbol_counts_global = np.eye(int(symbolic.row["num_symbols"]), dtype=np.float32)[symbolic.symbols].sum(axis=2).reshape(-1, int(symbolic.row["num_symbols"]))

    local_cap_seed = seed + 701
    for label_name, label_values, symbol_values in [
        ("x_i", local_x, symbol_flat),
        ("local_neighborhood_class", local_neighborhood, symbol_flat),
        ("rule184_current_formula", local_current, symbol_flat),
    ]:
        sym_sub, lab_sub = _subsample_validation(symbol_values[:, None], label_values, VALIDATION_LOCAL_CAP, local_cap_seed)
        rows.append(
            {
                "discovery": "symbolic",
                "candidate": "best",
                "label": label_name,
                "metric": "majority_accuracy",
                "value": _majority_map_accuracy(sym_sub[:, 0].astype(np.int64), lab_sub.astype(np.int64)),
            }
        )
        rows.append(
            {
                "discovery": "symbolic",
                "candidate": "best",
                "label": label_name,
                "metric": "normalized_mutual_info",
                "value": float(normalized_mutual_info_score(lab_sub.astype(np.int64), sym_sub[:, 0].astype(np.int64))),
            }
        )

    sym_sub, next_sub = _subsample_validation(symbol_now[:, None], next_x, VALIDATION_LOCAL_CAP, seed + 702)
    rows.append(
        {
            "discovery": "symbolic",
            "candidate": "best",
            "label": "next_cell",
            "metric": "majority_accuracy",
            "value": _majority_map_accuracy(sym_sub[:, 0].astype(np.int64), next_sub.astype(np.int64)),
        }
    )
    rows.append(
        {
            "discovery": "symbolic",
            "candidate": "best",
            "label": "next_cell",
            "metric": "normalized_mutual_info",
            "value": float(normalized_mutual_info_score(next_sub.astype(np.int64), sym_sub[:, 0].astype(np.int64))),
        }
    )

    r2, corr = _regression_metrics(symbol_counts_global, particle_number, seed + 703)
    rows.extend(
        [
            {"discovery": "symbolic", "candidate": "best", "label": "particle_number", "metric": "r2", "value": r2},
            {"discovery": "symbolic", "candidate": "best", "label": "particle_number", "metric": "pearson", "value": corr},
        ]
    )

    for discovery_name, field, next_field in [
        ("conserved_q_local", conserved.q_local, conserved.q_local[:, :-1]),
        ("continuity_rho", continuity.rho, continuity.rho[:, :-1]),
    ]:
        field_flat = field.reshape(-1, 1)
        field_now = next_field.reshape(-1, 1)
        for label_name, label_values in [
            ("x_i", local_x),
            ("local_neighborhood_class", local_neighborhood),
            ("rule184_current_formula", local_current),
        ]:
            feat_sub, lab_sub = _subsample_validation(field_flat, label_values, VALIDATION_LOCAL_CAP, seed + 710)
            rows.append(
                {
                    "discovery": discovery_name,
                    "candidate": "best",
                    "label": label_name,
                    "metric": "ridge_accuracy",
                    "value": _classifier_accuracy(feat_sub, lab_sub.astype(np.int64), seed + 11),
                }
            )
            if label_name != "local_neighborhood_class":
                rows.append(
                    {
                        "discovery": discovery_name,
                        "candidate": "best",
                        "label": label_name,
                        "metric": "pearson",
                        "value": _pearson_corr(feat_sub[:, 0], lab_sub),
                    }
                )
        feat_sub, lab_sub = _subsample_validation(field_now, next_x, VALIDATION_LOCAL_CAP, seed + 711)
        rows.append(
            {
                "discovery": discovery_name,
                "candidate": "best",
                "label": "next_cell",
                "metric": "ridge_accuracy",
                "value": _classifier_accuracy(feat_sub, lab_sub.astype(np.int64), seed + 12),
            }
        )
        rows.append(
            {
                "discovery": discovery_name,
                "candidate": "best",
                "label": "particle_number",
                "metric": "pearson",
                "value": _pearson_corr(field.sum(axis=-1), labels.particle_number),
            }
        )

    J_flat = continuity.J.reshape(-1, 1)
    feat_sub, lab_sub = _subsample_validation(J_flat, local_current, VALIDATION_LOCAL_CAP, seed + 712)
    rows.extend(
        [
            {
                "discovery": "continuity_current",
                "candidate": "best",
                "label": "rule184_current_formula",
                "metric": "ridge_accuracy",
                "value": _classifier_accuracy(feat_sub, lab_sub.astype(np.int64), seed + 13),
            },
            {
                "discovery": "continuity_current",
                "candidate": "best",
                "label": "rule184_current_formula",
                "metric": "pearson",
                "value": _pearson_corr(feat_sub[:, 0], lab_sub),
            },
        ]
    )

    r2, corr = _regression_metrics(conserved.Q.reshape(-1, 1), particle_number, seed + 720)
    rows.extend(
        [
            {"discovery": "conserved_Q", "candidate": "best", "label": "particle_number", "metric": "r2", "value": r2},
            {"discovery": "conserved_Q", "candidate": "best", "label": "particle_number", "metric": "pearson", "value": corr},
        ]
    )
    r2, corr = _regression_metrics(slow.g.reshape(-1, slow.g.shape[-1]), particle_number, seed + 721)
    rows.extend(
        [
            {"discovery": "slow_variables", "candidate": "best", "label": "particle_number", "metric": "r2", "value": r2},
            {"discovery": "slow_variables", "candidate": "best", "label": "particle_number", "metric": "pearson", "value": corr},
        ]
    )

    validation_df = pd.DataFrame(rows)
    validation_df.to_csv(output_dir / "posthoc_validation.csv", index=False)
    return validation_df


def _discovery_output_dir(spec: RunSpec, output_root: str | Path) -> Path:
    run_name = spec.checkpoint_path.parent.name
    return (
        Path(output_root).resolve()
        / spec.rule_label
        / spec.model_name
        / "latent_discovery"
        / run_name
    )


def run_discovery_for_spec(
    spec: RunSpec,
    model: torch.nn.Module,
    trajectories: torch.Tensor,
    output_dir: str | Path,
    cluster_ks: list[int],
    radii: list[int],
    families: list[str],
    seed: int,
) -> dict[str, str]:
    output_dir = prepare_output_dir(output_dir)
    supported, reason = supports_spatial_latent_analysis(model, spec.config)
    if not supported:
        save_skip(output_dir / "summary.json", reason)
        return {"status": "skipped", "reason": reason}
    if trajectories.shape[1] < 2:
        reason = "Need at least two timesteps for latent discovery."
        save_skip(output_dir / "summary.json", reason)
        return {"status": "skipped", "reason": reason}

    latent_seq, latent_reason = encode_spatial_latent_sequence(model, trajectories)
    if latent_seq is None:
        save_skip(output_dir / "summary.json", latent_reason or "Failed to encode latent sequence.")
        return {"status": "skipped", "reason": latent_reason or "latent encoding failed"}

    labels = _build_labels(trajectories)
    symbolic_df, symbolic_best = run_symbolic_discovery(latent_seq, output_dir / "symbolic", cluster_ks=cluster_ks, radii=radii, seed=seed)
    conservation_df, conserved_best = run_conserved_quantity_discovery(latent_seq, output_dir / "conserved_quantities", families=families)
    slow_df, slow_best = run_slow_variable_discovery(latent_seq, output_dir / "slow_variables", families=families)
    continuity_df, continuity_best = run_continuity_discovery(latent_seq, output_dir / "continuity_equation", radii=radii, families=families)
    validation_df = run_posthoc_validation(
        labels=labels,
        symbolic=symbolic_best,
        conserved=conserved_best,
        slow=slow_best,
        continuity=continuity_best,
        output_dir=output_dir / "posthoc_validation",
        seed=seed,
    )

    save_json(
        {
            "status": "ok",
            "symbolic_best": symbolic_best.row,
            "conserved_best": conserved_best.row,
            "slow_best": slow_best.row,
            "continuity_best": continuity_best.row,
            "num_symbolic_candidates": int(symbolic_df.shape[0]),
            "num_conservation_fits": int(conservation_df.shape[0]),
            "num_slow_fits": int(slow_df.shape[0]),
            "num_continuity_fits": int(continuity_df.shape[0]),
            "num_validation_rows": int(validation_df.shape[0]),
        },
        output_dir / "summary.json",
    )
    return {"status": "ok"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unsupervised latent-physics discovery for CA world models.")
    parser.add_argument("--config", nargs="*", default=None, help="Optional config path(s) when not using sibling config.json.")
    parser.add_argument("--checkpoint", nargs="*", default=None, help="Checkpoint path(s) to analyze.")
    parser.add_argument("--search_root", nargs="*", default=["outputs"], help="Search roots when checkpoint paths are omitted.")
    parser.add_argument("--rules", nargs="*", type=int, default=[30, 54, 110, 184], help="Rules to analyze.")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_root", default="analysis/outputs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cpu_threads", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=64, help="Maximum number of trajectories per run.")
    parser.add_argument("--cluster_K", nargs="*", type=int, default=[2, 4, 8, 16], help="Cluster sizes for symbolic discovery.")
    parser.add_argument("--radii", nargs="*", type=int, default=[0, 1, 2, 3], help="Local radii used for symbolic and continuity discovery.")
    parser.add_argument("--fit_linear", action="store_true", help="Fit linear conserved/slow/continuity heads. Enabled by default when no fit flags are passed.")
    parser.add_argument("--fit_mlp", action="store_true", help="Fit small MLP conserved/slow/continuity heads. Enabled by default when no fit flags are passed.")
    parser.add_argument("--checkpoint_name", default="best.ckpt")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configure_runtime(cpu_threads=args.cpu_threads, interop_threads=1)
    set_seed(args.seed)
    set_analysis_style()

    families = []
    if args.fit_linear or not args.fit_mlp:
        families.append("linear")
    if args.fit_mlp or not args.fit_linear:
        families.append("mlp")
    families = sorted(set(families), key=lambda item: ["linear", "mlp"].index(item))

    rules = set(args.rules) if args.rules else None
    specs = discover_run_specs(
        config_paths=args.config,
        checkpoint_paths=args.checkpoint,
        search_roots=args.search_root,
        output_root=args.output_root,
        rules=rules,
        checkpoint_name=args.checkpoint_name,
    )
    if not specs:
        log("No runs found for latent discovery.")
        return

    device = torch.device(args.device)
    manifest_rows: list[dict[str, Any]] = []
    for spec in specs:
        run_output_dir = _discovery_output_dir(spec, args.output_root)
        log(f"Starting latent discovery for {spec.checkpoint_path.parent.name} ({spec.rule_label}, {spec.model_name})")
        try:
            model = load_model_from_checkpoint(spec, device)
            trajectories = load_trajectories(spec.config, args.data_dir, "test")[: args.max_samples].to(device)
            result = run_discovery_for_spec(
                spec=spec,
                model=model,
                trajectories=trajectories,
                output_dir=run_output_dir,
                cluster_ks=[int(value) for value in args.cluster_K],
                radii=[int(value) for value in args.radii],
                families=families,
                seed=args.seed,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            log(f"{spec.checkpoint_path.parent.name}: latent discovery failed: {exc}")
            result = {"status": "failed", "reason": str(exc)}
            prepare_output_dir(run_output_dir)
            save_json(result, run_output_dir / "summary.json")

        manifest_rows.append(
            {
                "run_name": spec.checkpoint_path.parent.name,
                "rule": spec.rule,
                "model_name": spec.model_name,
                "status": result.get("status", "ok"),
                "output_dir": str(run_output_dir),
                "checkpoint": str(spec.checkpoint_path),
            }
        )

    save_manifest(manifest_rows, Path(args.output_root) / "latent_discovery_manifest.csv")
    log(f"Finished latent discovery for {len(specs)} run(s).")


if __name__ == "__main__":
    main()
