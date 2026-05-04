"""Plot helpers for the analysis suite."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.plotting import color_cycle, save_figure, set_default_style


def set_analysis_style() -> None:
    set_default_style()
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.figsize": (8, 4.5),
            "image.interpolation": "nearest",
        }
    )


def plot_metric_curves(
    df: pd.DataFrame,
    path: str | Path,
    metric: str,
    hue: str,
    title: str,
    ylabel: str | None = None,
    x: str = "step",
    style: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = color_cycle(max(1, df[hue].nunique()))
    for color, (name, group) in zip(colors, df.groupby(hue, sort=False)):
        line_style = "-"
        if style is not None and style in group.columns:
            styles = sorted(group[style].dropna().unique())
            for style_value in styles:
                subgroup = group[group[style] == style_value]
                ax.plot(
                    subgroup[x],
                    subgroup[metric],
                    label=f"{name} ({style_value})",
                    color=color,
                    linewidth=2,
                    linestyle="--" if str(style_value).lower() in {"exact", "ground_truth"} else "-",
                )
            continue
        ax.plot(group[x], group[metric], label=str(name), color=color, linewidth=2)
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(ylabel or metric.replace("_", " ").title())
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    save_figure(fig, path)


def plot_spacetime_with_latents(
    input_spacetime: np.ndarray,
    pred_spacetime: np.ndarray,
    error_map: np.ndarray,
    latent_channels: np.ndarray,
    channel_indices: list[int],
    path: str | Path,
    title: str,
) -> None:
    num_rows = 3 + len(channel_indices)
    fig, axes = plt.subplots(num_rows, 1, figsize=(10, 1.5 * num_rows), sharex=True)
    panels = [
        ("Input CA", input_spacetime, "gray_r", 0.0, 1.0),
        ("Predicted CA", pred_spacetime, "gray_r", 0.0, 1.0),
        ("Error Map", error_map, "magma", 0.0, 1.0),
    ]
    for axis, (label, image, cmap, vmin, vmax) in zip(axes[:3], panels):
        axis.imshow(image, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_ylabel(label)
    vmax = float(np.abs(latent_channels).max()) if latent_channels.size else 1.0
    vmax = max(vmax, 1e-6)
    for axis, channel_index, image in zip(axes[3:], channel_indices, latent_channels):
        axis.imshow(image, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        axis.set_ylabel(f"z[{channel_index}]")
    axes[-1].set_xlabel("Position")
    axes[0].set_title(title)
    save_figure(fig, path)


def plot_embedding_scatter(
    embedding: np.ndarray,
    color: np.ndarray,
    path: str | Path,
    title: str,
    label: str,
    categorical: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if categorical:
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=color, s=8, cmap="tab10", alpha=0.75)
    else:
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=color, s=8, cmap="viridis", alpha=0.75)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(label)
    save_figure(fig, path)


def plot_probe_radius_curve(df: pd.DataFrame, path: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["radius"], df["mse"], marker="o", linewidth=2, label="MSE")
    if "r2" in df.columns:
        ax.plot(df["radius"], df["r2"], marker="s", linewidth=2, label="R2")
    ax.set_xlabel("Locality Radius")
    ax.set_ylabel("Probe Score")
    ax.set_title(title)
    ax.legend(frameon=False)
    save_figure(fig, path)


def plot_count_trajectories(
    exact_counts: np.ndarray,
    pred_counts: np.ndarray,
    path: str | Path,
    title: str,
) -> None:
    steps = np.arange(exact_counts.shape[1])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sample_count = min(8, exact_counts.shape[0])
    for index in range(sample_count):
        ax.plot(steps, exact_counts[index], color="black", alpha=0.12, linewidth=1)
        ax.plot(steps, pred_counts[index], color="tab:orange", alpha=0.12, linewidth=1)
    ax.plot(steps, exact_counts.mean(axis=0), color="black", linewidth=2.5, label="Ground truth")
    ax.plot(steps, pred_counts.mean(axis=0), color="tab:orange", linewidth=2.5, label="Model rollout")
    ax.set_xlabel("Step")
    ax.set_ylabel("Particle Count")
    ax.set_title(title)
    ax.legend(frameon=False)
    save_figure(fig, path)


def plot_q_vs_count(
    exact_mean: np.ndarray,
    q_teacher_mean: np.ndarray,
    q_rollout_mean: np.ndarray,
    path: str | Path,
    title: str,
) -> None:
    steps = np.arange(exact_mean.shape[0])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, exact_mean, color="black", linewidth=2.5, label="True count")
    ax.plot(steps, q_teacher_mean, color="tab:blue", linewidth=2.0, label="Q on encoded truth")
    ax.plot(steps, q_rollout_mean, color="tab:green", linewidth=2.0, label="Q on latent rollout")
    ax.set_xlabel("Step")
    ax.set_ylabel("Count / Q")
    ax.set_title(title)
    ax.legend(frameon=False)
    save_figure(fig, path)


def choose_latent_channels(latent_seq: np.ndarray, num_channels: int = 4) -> list[int]:
    if latent_seq.shape[1] <= num_channels:
        return list(range(latent_seq.shape[1]))
    variances = latent_seq.var(axis=(0, 2))
    order = np.argsort(variances)[::-1]
    return [int(index) for index in order[:num_channels]]

