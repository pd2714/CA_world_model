"""Shared plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 150) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def set_default_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (8, 4),
            "axes.grid": False,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "font.size": 10,
        }
    )


def add_legend_if_handles(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False)


def color_cycle(n: int) -> Iterable[str]:
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]
