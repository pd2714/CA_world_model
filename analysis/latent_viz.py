"""Latent-channel visualization and low-dimensional latent embeddings."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch

from analysis.plotting import choose_latent_channels, plot_embedding_scatter, plot_spacetime_with_latents
from analysis.utils import (
    DEFAULT_SEED,
    RunSpec,
    collect_positionwise_latent_data,
    encode_spatial_latent_sequence,
    log,
    prepare_output_dir,
    rollout_main_model,
    save_skip,
    sanitize_numpy,
    supports_spatial_latent_analysis,
)

try:
    from sklearn.decomposition import PCA
except ImportError:  # pragma: no cover
    PCA = None

try:
    import umap  # type: ignore
except ImportError:  # pragma: no cover
    umap = None


def _project_pca(features: np.ndarray) -> np.ndarray:
    features = sanitize_numpy(features)
    if PCA is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return PCA(n_components=2, random_state=DEFAULT_SEED).fit_transform(features)
    centered = features - features.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    return u[:, :2] * s[:2]


def _project_umap(features: np.ndarray) -> np.ndarray | None:
    if umap is None:
        return None
    reducer = umap.UMAP(n_components=2, random_state=DEFAULT_SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return reducer.fit_transform(features)


def run_latent_channel_visualization(
    spec: RunSpec,
    model: torch.nn.Module,
    trajectories: torch.Tensor,
    device: torch.device,
    output_dir: str | Path,
    max_steps: int = 60,
) -> dict[str, str]:
    output_dir = prepare_output_dir(output_dir)
    supported, reason = supports_spatial_latent_analysis(model, spec.config)
    if not supported:
        save_skip(output_dir / "summary.json", reason)
        return {"status": "skipped", "reason": reason}

    sequence = trajectories[:1, : min(max_steps + 1, trajectories.shape[1])].to(device)
    latent_seq, latent_reason = encode_spatial_latent_sequence(model, sequence)
    if latent_seq is None:
        save_skip(output_dir / "summary.json", latent_reason or "Failed to encode latent sequence.")
        return {"status": "skipped", "reason": latent_reason or "latent encoding failed"}

    pred_states, _ = rollout_main_model(
        model,
        sequence[:, 0],
        steps=sequence.shape[1] - 1,
        model_name=spec.model_name,
        feedback_mode="hard",
        rollout_mode="latent",
    )

    target_np = sequence[0, :, 0].detach().cpu().numpy()
    pred_np = pred_states[0, : sequence.shape[1], 0].detach().cpu().numpy()
    error_np = np.abs(pred_np - target_np)
    latent_np = latent_seq[0].detach().cpu().numpy()
    channel_indices = choose_latent_channels(latent_np, num_channels=4)
    plot_spacetime_with_latents(
        input_spacetime=target_np,
        pred_spacetime=pred_np,
        error_map=error_np,
        latent_channels=latent_np[channel_indices],
        channel_indices=channel_indices,
        path=output_dir / "latent_channels.png",
        title=f"{spec.rule_label} {spec.model_name}: rollout + latent channels",
    )
    pd.DataFrame({"channel_index": channel_indices}).to_csv(output_dir / "selected_channels.csv", index=False)
    return {"status": "ok"}


def run_latent_embedding_analysis(
    spec: RunSpec,
    model: torch.nn.Module,
    trajectories: torch.Tensor,
    output_dir: str | Path,
    max_points: int = 6000,
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
        save_skip(output_dir / "summary.json", latent_reason or "Failed to collect latent features.")
        return {"status": "skipped", "reason": latent_reason or "latent feature collection failed"}

    embedding_frames: list[pd.DataFrame] = []
    pca_embedding = _project_pca(latent_data.features)
    embedding_frames.append(pd.DataFrame({"method": "pca", "x": pca_embedding[:, 0], "y": pca_embedding[:, 1]}))

    plot_embedding_scatter(pca_embedding, latent_data.time_idx, output_dir / "pca_time_idx.png", "PCA colored by time", "time", categorical=False)
    plot_embedding_scatter(pca_embedding, latent_data.center_cell, output_dir / "pca_center_cell.png", "PCA colored by center cell", "x_i", categorical=True)
    plot_embedding_scatter(pca_embedding, latent_data.neighborhood_class, output_dir / "pca_neighborhood_class.png", "PCA colored by neighborhood", "000-111", categorical=True)
    plot_embedding_scatter(pca_embedding, latent_data.next_cell, output_dir / "pca_next_cell.png", "PCA colored by next cell", "x_{t+1,i}", categorical=True)

    if spec.rule == 184 and latent_data.local_current is not None:
        plot_embedding_scatter(pca_embedding, latent_data.local_current, output_dir / "pca_local_current.png", "PCA colored by Rule 184 local current", "current", categorical=True)

    umap_embedding = _project_umap(latent_data.features)
    if umap_embedding is not None:
        embedding_frames.append(pd.DataFrame({"method": "umap", "x": umap_embedding[:, 0], "y": umap_embedding[:, 1]}))
        plot_embedding_scatter(umap_embedding, latent_data.time_idx, output_dir / "umap_time_idx.png", "UMAP colored by time", "time", categorical=False)
        plot_embedding_scatter(umap_embedding, latent_data.center_cell, output_dir / "umap_center_cell.png", "UMAP colored by center cell", "x_i", categorical=True)
        plot_embedding_scatter(umap_embedding, latent_data.neighborhood_class, output_dir / "umap_neighborhood_class.png", "UMAP colored by neighborhood", "000-111", categorical=True)
        plot_embedding_scatter(umap_embedding, latent_data.next_cell, output_dir / "umap_next_cell.png", "UMAP colored by next cell", "x_{t+1,i}", categorical=True)
        if spec.rule == 184 and latent_data.local_current is not None:
            plot_embedding_scatter(umap_embedding, latent_data.local_current, output_dir / "umap_local_current.png", "UMAP colored by Rule 184 local current", "current", categorical=True)
    else:
        log(f"{spec.run_name}: UMAP not installed, skipping optional UMAP plots.")

    metadata = pd.DataFrame(
        {
            "time_idx": latent_data.time_idx,
            "center_cell": latent_data.center_cell,
            "left_neighbor": latent_data.left_neighbor,
            "right_neighbor": latent_data.right_neighbor,
            "next_cell": latent_data.next_cell,
            "neighborhood_class": latent_data.neighborhood_class,
            "local_density": latent_data.local_density,
            "local_current": latent_data.local_current if latent_data.local_current is not None else np.full_like(latent_data.center_cell, np.nan, dtype=np.float32),
        }
    )
    for frame in embedding_frames:
        metadata = pd.concat([metadata, frame.reset_index(drop=True)], axis=1)
    metadata.to_csv(output_dir / "embedding_points.csv", index=False)
    return {"status": "ok"}
