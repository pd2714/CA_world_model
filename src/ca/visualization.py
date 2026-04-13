"""Plotting helpers for cellular automata."""

from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from src.utils.plotting import save_figure


def plot_1d_trajectory(trajectory: np.ndarray, title: str = "", path: str | Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(trajectory, aspect="auto", cmap="binary", interpolation="nearest", origin="upper")
    ax.set_xlabel("Cell index")
    ax.set_ylabel("Time")
    ax.set_title(title or "1D CA trajectory")
    if path is not None:
        save_figure(fig, path)
    return fig


def plot_trajectory_comparison(
    exact: np.ndarray,
    predicted: np.ndarray,
    title: str = "",
    path: str | Path | None = None,
) -> plt.Figure:
    diff = np.abs(exact - predicted)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, image, name in zip(axes, [exact, predicted, diff], ["Exact", "Predicted", "Difference"]):
        ax.imshow(image, aspect="auto", cmap="binary" if name != "Difference" else "magma", interpolation="nearest")
        ax.set_title(name)
        ax.set_xlabel("Cell index")
    axes[0].set_ylabel("Time")
    fig.suptitle(title or "Trajectory comparison")
    if path is not None:
        save_figure(fig, path)
    return fig


def animate_2d_trajectory(trajectory: np.ndarray, path: str | Path, interval: int = 200) -> None:
    fig, ax = plt.subplots()
    image = ax.imshow(trajectory[0], cmap="binary", animated=True)
    ax.set_title("2D CA rollout")

    def update(frame: int):
        image.set_array(trajectory[frame])
        ax.set_xlabel(f"t={frame}")
        return (image,)

    ani = animation.FuncAnimation(fig, update, frames=len(trajectory), interval=interval, blit=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(path, writer="pillow")
    plt.close(fig)
