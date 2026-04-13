"""Plotting and tabulation for rollout comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.plotting import add_legend_if_handles, color_cycle, save_figure


def plot_error_vs_horizon(df: pd.DataFrame, path: str | Path, metric: str = "hamming", hue: str = "model") -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = color_cycle(len(df[hue].unique()))
    for color, (name, group) in zip(colors, df.groupby(hue)):
        summary = group.groupby("horizon")[metric].mean()
        ax.plot(summary.index, summary.values, label=name, color=color, linewidth=2)
    ax.set_xlabel("Horizon")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} vs horizon")
    add_legend_if_handles(ax)
    save_figure(fig, path)


def plot_multi_metric_vs_horizon(
    df: pd.DataFrame,
    path: str | Path,
    metrics: list[str],
    labels: list[str] | None = None,
    x: str = "horizon",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = labels or metrics
    colors = color_cycle(len(metrics))
    for color, metric, label in zip(colors, metrics, labels):
        summary = df.groupby(x)[metric].mean()
        ax.plot(summary.index, summary.values, label=label, color=color, linewidth=2)
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel("Value")
    ax.set_title("Rollout diagnostics vs horizon")
    add_legend_if_handles(ax)
    save_figure(fig, path)
