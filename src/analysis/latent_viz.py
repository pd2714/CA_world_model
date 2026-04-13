"""Latent visualization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except ImportError:  # pragma: no cover - fallback path
    PCA = None
    TSNE = None

from src.utils.plotting import save_figure


def flatten_latents(latents: np.ndarray) -> np.ndarray:
    return latents.reshape(latents.shape[0], -1)


def pca_project(latents: np.ndarray, n_components: int = 2) -> np.ndarray:
    flat = flatten_latents(latents)
    if PCA is not None:
        return PCA(n_components=n_components).fit_transform(flat)
    centered = flat - flat.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    return u[:, :n_components] * s[:n_components]


def tsne_project(latents: np.ndarray, n_components: int = 2, perplexity: int = 30) -> np.ndarray:
    if TSNE is None:
        raise ImportError("t-SNE requires scikit-learn to be installed.")
    return TSNE(n_components=n_components, perplexity=perplexity, init="pca", learning_rate="auto").fit_transform(flatten_latents(latents))


def plot_latent_projection(
    projection: np.ndarray,
    color: np.ndarray,
    title: str,
    path: str | Path,
    color_label: str = "observable",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(projection[:, 0], projection[:, 1], c=color, cmap="viridis", s=16)
    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(color_label)
    save_figure(fig, path)


def save_projection_bundle(
    latents: np.ndarray,
    color: np.ndarray,
    out_dir: str | Path,
    title_prefix: str = "latent",
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pca = pca_project(latents)
    plot_latent_projection(pca, color, f"{title_prefix} PCA", out_dir / "latent_pca.png")
    results: dict[str, Any] = {"pca": pca}
    if len(latents) >= 20 and TSNE is not None:
        tsne = tsne_project(latents)
        plot_latent_projection(tsne, color, f"{title_prefix} t-SNE", out_dir / "latent_tsne.png")
        results["tsne"] = tsne
    return results
