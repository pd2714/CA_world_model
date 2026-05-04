"""Generate publication-style architecture figures for 1D CA world models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


BLACK = "#161616"
MID = "#6B7280"
LIGHT = "#EEF1F4"
ACCENT = "#D9DEE5"
ACCENT_DARK = "#9BA7B4"
BLUE = "#C9D8EE"
BLUE_DARK = "#99B5DB"
SAND = "#E9DFC9"
GREEN = "#D7E9D7"


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    *,
    fc: str = LIGHT,
    ec: str = BLACK,
    lw: float = 1.4,
    fontsize: float = 11,
    rounded: bool = True,
    dash: tuple[int, int] | None = None,
) -> None:
    if rounded:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            linestyle=(0, dash) if dash else "solid",
        )
    else:
        patch = Rectangle(
            (x, y),
            w,
            h,
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            linestyle=(0, dash) if dash else "solid",
        )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize, color=BLACK)


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    label: str | None = None,
    label_xy: tuple[float, float] | None = None,
    lw: float = 1.5,
    style: str = "-|>",
    mutation_scale: float = 12,
    linestyle: str = "solid",
    connectionstyle: str = "arc3",
    color: str = BLACK,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=lw,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        color=color,
    )
    ax.add_patch(arrow)
    if label:
        if label_xy is None:
            label_xy = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.03)
        ax.text(label_xy[0], label_xy[1], label, ha="center", va="bottom", fontsize=9.5, color=MID)


def draw_binary_state(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    values: list[int],
    *,
    subtitle: str | None = "1D binary CA state",
) -> None:
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=BLACK, linewidth=1.4))
    cell_w = w / len(values)
    for idx, value in enumerate(values):
        fc = BLACK if value else "white"
        ax.add_patch(
            Rectangle(
                (x + idx * cell_w, y),
                cell_w,
                h,
                facecolor=fc,
                edgecolor=BLACK,
                linewidth=0.75,
            )
    )
    ax.text(x + w / 2, y + h + 0.03, label, ha="center", va="bottom", fontsize=11, color=BLACK)
    if subtitle:
        ax.text(x + w / 2, y - 0.03, subtitle, ha="center", va="top", fontsize=8.8, color=MID)


def draw_latent_tensor(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    *,
    layers: int = 1,
    rows: int = 4,
    cols: int = 10,
    subtitle: str | None = None,
    highlight_cols: tuple[int, ...] = (),
    fill_pattern: list[list[str | None]] | None = None,
) -> None:
    offset = 0.012
    base_w = w - offset * (layers - 1)
    base_h = h - offset * (layers - 1)
    for layer in range(layers):
        lx = x + offset * (layers - layer - 1)
        ly = y + offset * (layers - layer - 1)
        ax.add_patch(Rectangle((lx, ly), base_w, base_h, facecolor="white", edgecolor=BLACK, linewidth=1.1))
    grid_x = x + offset * (layers - 1)
    grid_y = y + offset * (layers - 1)
    grid_w = base_w
    grid_h = base_h
    cell_w = grid_w / cols
    cell_h = grid_h / rows
    for row in range(rows):
        for col in range(cols):
            fc = "white"
            if fill_pattern is not None:
                fc = fill_pattern[row][col] or "white"
            elif col in highlight_cols:
                fc = ACCENT
            ax.add_patch(
                Rectangle(
                    (grid_x + col * cell_w, grid_y + row * cell_h),
                    cell_w,
                    cell_h,
                    facecolor=fc,
                    edgecolor=ACCENT_DARK if fc != "white" else BLACK,
                    linewidth=0.55,
                )
            )
    ax.text(x + w / 2, y + h + 0.03, label, ha="center", va="bottom", fontsize=11, color=BLACK)
    if subtitle:
        ax.text(x + w / 2, y - 0.03, subtitle, ha="center", va="top", fontsize=8.8, color=MID)


def add_caption(ax: plt.Axes, text: str) -> None:
    ax.text(0.5, 0.03, text, ha="center", va="bottom", fontsize=8.9, color=MID)


def add_panel_title(ax: plt.Axes, title: str) -> None:
    ax.text(0.03, 0.96, title, ha="left", va="top", fontsize=13.5, fontweight="bold", color=BLACK)


def draw_rollout_model_box(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    add_box(ax, x, y, w, h, "same\nmodel", fc=LIGHT, fontsize=10, lw=1.2)


def latent_pattern_a() -> list[list[str | None]]:
    return [
        [None, BLUE, BLUE, None, None, SAND, SAND, None, None, GREEN],
        [None, BLUE, BLUE_DARK, BLUE, None, SAND, SAND, None, GREEN, GREEN],
        [None, None, BLUE, BLUE, None, None, SAND, GREEN, GREEN, None],
        [SAND, None, None, BLUE, BLUE, None, GREEN, GREEN, None, None],
    ]


def latent_pattern_b() -> list[list[str | None]]:
    return [
        [None, None, BLUE, BLUE, None, SAND, SAND, None, GREEN, GREEN],
        [None, BLUE, BLUE_DARK, BLUE, None, None, SAND, GREEN, GREEN, None],
        [SAND, None, BLUE, BLUE, None, None, GREEN, GREEN, None, None],
        [SAND, SAND, None, BLUE, None, GREEN, GREEN, None, None, None],
    ]


def latent_pattern_c() -> list[list[str | None]]:
    return [
        [None, SAND, None, BLUE, BLUE, None, GREEN, GREEN, None, None],
        [SAND, SAND, None, BLUE_DARK, BLUE, None, None, GREEN, GREEN, None],
        [SAND, None, None, BLUE, BLUE, None, None, GREEN, GREEN, None],
        [None, None, BLUE, BLUE, None, SAND, None, None, GREEN, GREEN],
    ]


def draw_panel_a(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_panel_title(ax, "A. Pure rollout world model")

    add_box(ax, 0.15, 0.55, 0.74, 0.28, "", fc="white", ec=ACCENT_DARK, lw=1.2, dash=(4, 3))
    ax.text(0.52, 0.84, "single-step transition model", ha="center", va="bottom", fontsize=9.3, color=MID)

    draw_binary_state(ax, 0.04, 0.62, 0.12, 0.10, r"$x_t$", [0, 1, 1, 0, 1, 0, 0, 1])
    add_box(ax, 0.21, 0.61, 0.10, 0.12, "encoder")
    draw_latent_tensor(
        ax,
        0.36,
        0.60,
        0.12,
        0.12,
        r"$z_t$",
        subtitle="spatial latent",
        fill_pattern=latent_pattern_a(),
    )
    ax.text(0.53, 0.71, r"$f$", ha="center", va="center", fontsize=16, color=BLACK)
    draw_latent_tensor(
        ax,
        0.58,
        0.60,
        0.12,
        0.12,
        r"$z_{t+1}$",
        subtitle="predicted latent",
        fill_pattern=latent_pattern_b(),
    )
    add_box(ax, 0.76, 0.61, 0.10, 0.12, "decoder")
    draw_binary_state(ax, 0.90, 0.62, 0.08, 0.10, r"$\hat{x}_{t+1}$", [0, 1, 0, 1, 1, 0, 1, 0])

    add_arrow(ax, (0.16, 0.67), (0.21, 0.67))
    add_arrow(ax, (0.31, 0.67), (0.36, 0.67))
    add_arrow(ax, (0.48, 0.67), (0.58, 0.67))
    add_arrow(ax, (0.70, 0.67), (0.76, 0.67))
    add_arrow(ax, (0.86, 0.67), (0.90, 0.67))

    ax.text(0.63, 0.46, "recursive self-rollout", ha="center", va="center", fontsize=10.5, color=BLACK)

    draw_binary_state(ax, 0.36, 0.24, 0.10, 0.08, r"$\hat{x}_{t+1}$", [0, 1, 0, 1, 1, 0, 1, 0], subtitle=None)
    draw_rollout_model_box(ax, 0.49, 0.22, 0.10, 0.11)
    draw_binary_state(ax, 0.62, 0.24, 0.10, 0.08, r"$\hat{x}_{t+2}$", [1, 0, 1, 1, 0, 1, 0, 0], subtitle=None)
    draw_rollout_model_box(ax, 0.75, 0.22, 0.10, 0.11)
    draw_binary_state(ax, 0.88, 0.24, 0.10, 0.08, r"$\hat{x}_{t+3}$", [1, 0, 1, 0, 1, 1, 0, 0], subtitle=None)

    add_arrow(ax, (0.46, 0.28), (0.49, 0.28))
    add_arrow(ax, (0.59, 0.28), (0.62, 0.28))
    add_arrow(ax, (0.72, 0.28), (0.75, 0.28))
    add_arrow(ax, (0.85, 0.28), (0.88, 0.28))
    add_arrow(
        ax,
        (0.94, 0.62),
        (0.41, 0.33),
        label="prediction becomes next input",
        label_xy=(0.70, 0.40),
        linestyle="dashed",
        connectionstyle="arc3,rad=-0.3",
        color=MID,
    )

    add_caption(
        ax,
        "Single-step prediction is reused recursively:\n"
        "the model rolls forward by consuming its own outputs.",
    )


def draw_shared_weight_markers(ax: plt.Axes, x0: float, y: float, spacing: float, count: int) -> None:
    for idx in range(count):
        x = x0 + idx * spacing
        ax.add_patch(Rectangle((x, y), 0.022, 0.022, facecolor=ACCENT, edgecolor=BLACK, linewidth=0.9))
        ax.plot([x + 0.011, x + 0.011], [y - 0.01, y], color=BLACK, linewidth=0.8)
    ax.text(
        x0 + spacing * (count - 1) / 2 + 0.011,
        y + 0.028,
        "same local kernel at every position",
        ha="center",
        va="bottom",
        fontsize=8.3,
        color=MID,
    )


def draw_repeat_brace(ax: plt.Axes, x0: float, x1: float, y: float, text: str) -> None:
    ax.plot([x0, x1], [y, y], color=MID, linewidth=1.2)
    ax.plot([x0, x0], [y, y - 0.02], color=MID, linewidth=1.2)
    ax.plot([x1, x1], [y, y - 0.02], color=MID, linewidth=1.2)
    ax.text((x0 + x1) / 2, y + 0.02, text, ha="center", va="bottom", fontsize=8.8, color=MID)


def draw_panel_b(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_panel_title(ax, "B. Biased / CA-structured world model")

    draw_binary_state(ax, 0.04, 0.62, 0.10, 0.10, r"$x_t$", [0, 1, 1, 0, 1, 0, 0, 1])
    add_box(ax, 0.18, 0.61, 0.10, 0.12, "encoder")
    draw_latent_tensor(
        ax,
        0.31,
        0.60,
        0.12,
        0.12,
        r"$z_t$",
        subtitle="spatial latent",
        highlight_cols=(3, 4, 5),
    )

    add_box(ax, 0.47, 0.50, 0.34, 0.30, "", fc="#F8FAFC", ec=BLACK, lw=1.5)
    ax.text(0.64, 0.755, "CA-structured latent dynamics", ha="center", va="center", fontsize=11.1, color=BLACK)

    draw_latent_tensor(ax, 0.50, 0.60, 0.09, 0.08, "", layers=1, rows=3, cols=8, highlight_cols=(2, 3, 4))
    ax.text(0.545, 0.575, "local neighborhood", ha="center", va="top", fontsize=8.5, color=MID)
    add_box(ax, 0.63, 0.60, 0.08, 0.08, "local\nconv", fc=ACCENT, fontsize=9.4, lw=1.1)
    draw_latent_tensor(ax, 0.74, 0.60, 0.06, 0.08, "", layers=1, rows=3, cols=5, highlight_cols=(2,))
    draw_shared_weight_markers(ax, 0.55, 0.685, 0.04, 5)
    add_arrow(ax, (0.59, 0.64), (0.63, 0.64), lw=1.2, mutation_scale=11)
    add_arrow(ax, (0.71, 0.64), (0.74, 0.64), lw=1.2, mutation_scale=11)

    ax.text(
        0.64,
        0.535,
        r"$z_{t+1} = z_t + \alpha\, f(\mathrm{local\ neighborhood})$",
        ha="center",
        va="center",
        fontsize=9.6,
        color=BLACK,
    )
    add_box(ax, 0.69, 0.492, 0.09, 0.045, "norm / clamp", fc="white", fontsize=8.5, lw=1.0)

    add_box(ax, 0.85, 0.61, 0.09, 0.12, "decoder")
    draw_binary_state(ax, 0.97, 0.62, 0.08, 0.10, r"$\hat{x}_{t+1}$", [0, 1, 0, 1, 1, 0])

    add_arrow(ax, (0.14, 0.67), (0.18, 0.67))
    add_arrow(ax, (0.28, 0.67), (0.31, 0.67))
    add_arrow(ax, (0.43, 0.67), (0.47, 0.67))
    add_arrow(ax, (0.81, 0.67), (0.85, 0.67))
    add_arrow(ax, (0.94, 0.67), (0.97, 0.67))
    ax.text(0.64, 0.82, "local, translation-equivariant update", ha="center", va="center", fontsize=9.3, color=MID)

    add_caption(
        ax,
        "Inductive bias is injected into latent dynamics:\n"
        "local neighborhoods, shared weights, residual updates, and stability control.",
    )


def draw_panel_d(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_panel_title(ax, "D. Longer latent rollout")
    ax.text(
        0.03,
        0.90,
        "Encode once, advance many times in latent space with $f$, and decode only at the target horizon.",
        ha="left",
        va="top",
        fontsize=9.4,
        color=MID,
    )

    add_box(ax, 0.28, 0.50, 0.42, 0.20, "", fc="white", ec=ACCENT_DARK, lw=1.2, dash=(4, 3))
    ax.text(0.49, 0.71, "latent rollout for H steps", ha="center", va="bottom", fontsize=9.2, color=MID)

    draw_binary_state(ax, 0.05, 0.55, 0.11, 0.10, r"$x_t$", [0, 1, 1, 0, 1, 0, 0, 1])
    add_box(ax, 0.20, 0.54, 0.10, 0.12, "encoder")
    draw_latent_tensor(
        ax,
        0.34,
        0.53,
        0.11,
        0.12,
        r"$z_t$",
        subtitle="start latent",
        fill_pattern=latent_pattern_a(),
    )
    draw_latent_tensor(
        ax,
        0.62,
        0.53,
        0.11,
        0.12,
        r"$z_{t+H}$",
        subtitle="latent after H steps",
        fill_pattern=latent_pattern_c(),
    )
    add_box(ax, 0.78, 0.54, 0.10, 0.12, "decoder")
    draw_binary_state(
        ax,
        0.91,
        0.55,
        0.08,
        0.10,
        r"$\hat{x}_{t+H}$",
        [1, 0, 1, 1, 0, 1, 0, 0],
        subtitle="long-horizon prediction",
    )

    add_arrow(ax, (0.16, 0.60), (0.20, 0.60))
    add_arrow(ax, (0.30, 0.60), (0.34, 0.60))
    add_arrow(ax, (0.45, 0.60), (0.62, 0.60), lw=1.8)
    add_arrow(ax, (0.73, 0.60), (0.78, 0.60))
    add_arrow(ax, (0.88, 0.60), (0.91, 0.60))

    ax.text(0.535, 0.675, r"repeat $f$ for $H$ steps", ha="center", va="center", fontsize=12.5, color=BLACK)
    ax.text(0.535, 0.63, r"$H$ shared latent transitions", ha="center", va="top", fontsize=9.0, color=MID)

    add_caption(
        ax,
        "Long-horizon forecasting can be done by rolling forward repeatedly in latent space,\n"
        "then decoding only the final latent state into a late-time prediction.",
    )


def draw_f_architecture(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_panel_title(ax, "C. Latent transition function $f$")
    ax.text(
        0.03,
        0.90,
        "Implemented as `SpatialLatentDynamics1D`: a local residual Conv1d stack with optional normalization and clamp.",
        ha="left",
        va="top",
        fontsize=9.4,
        color=MID,
    )

    draw_latent_tensor(
        ax,
        0.05,
        0.52,
        0.12,
        0.12,
        r"$z_t$",
        subtitle="input latent",
        fill_pattern=latent_pattern_a(),
    )

    add_box(ax, 0.23, 0.53, 0.12, 0.10, "LocalChannel\nNorm1D", fc="white", fontsize=9.8)
    add_box(ax, 0.39, 0.53, 0.14, 0.10, "Conv1d\nkernel = 3\ncircular", fc=LIGHT, fontsize=9.5)
    add_box(ax, 0.57, 0.53, 0.09, 0.10, "GELU", fc="white", fontsize=10.2)
    draw_repeat_brace(ax, 0.23, 0.66, 0.68, "repeat this block `depth - 1` times")

    add_box(ax, 0.70, 0.53, 0.12, 0.10, "output\nnorm", fc="white", fontsize=9.8)
    add_box(ax, 0.85, 0.53, 0.11, 0.10, "final Conv1d\nkernel = 3", fc=LIGHT, fontsize=9.3)

    add_arrow(ax, (0.17, 0.58), (0.23, 0.58))
    add_arrow(ax, (0.35, 0.58), (0.39, 0.58))
    add_arrow(ax, (0.53, 0.58), (0.57, 0.58))
    add_arrow(ax, (0.66, 0.58), (0.70, 0.58))
    add_arrow(ax, (0.82, 0.58), (0.85, 0.58))

    add_box(ax, 0.18, 0.22, 0.16, 0.10, r"optional clamp" "\n" r"$\Delta z \in [-c, c]$", fc="white", fontsize=9.2)
    add_box(ax, 0.40, 0.22, 0.12, 0.10, r"scale" "\n" r"$\alpha \Delta z$", fc="white", fontsize=9.6)
    ax.text(0.65, 0.27, r"$z_{t+1} = z_t + \alpha \Delta z$", ha="center", va="center", fontsize=14, color=BLACK)
    draw_latent_tensor(
        ax,
        0.83,
        0.20,
        0.12,
        0.12,
        r"$z_{t+1}$",
        subtitle="updated latent",
        fill_pattern=latent_pattern_b(),
    )

    add_arrow(ax, (0.905, 0.53), (0.905, 0.34), label=r"$\Delta z$", label_xy=(0.94, 0.42))
    add_arrow(ax, (0.905, 0.34), (0.34, 0.27), connectionstyle="arc3,rad=0.0")
    add_arrow(ax, (0.34, 0.27), (0.40, 0.27))
    add_arrow(ax, (0.52, 0.27), (0.57, 0.27))
    add_arrow(ax, (0.76, 0.27), (0.83, 0.27))
    add_arrow(ax, (0.11, 0.52), (0.57, 0.27), linestyle="dashed", color=MID, connectionstyle="arc3,rad=0.15")
    ax.text(0.29, 0.38, "skip connection from $z_t$", ha="center", va="center", fontsize=8.8, color=MID)

    ax.text(0.23, 0.12, "All convolutions preserve channel count and act locally along the 1D lattice.", ha="left", va="center", fontsize=9.0, color=MID)
    ax.text(0.23, 0.07, "Circular padding makes the update wrap around the ends of the CA.", ha="left", va="center", fontsize=9.0, color=MID)


def build_single_panel_figure(panel: str) -> plt.Figure:
    set_style()
    fig, ax = plt.subplots(figsize=(10.8, 6.3))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.965, bottom=0.10)
    if panel == "a":
        draw_panel_a(ax)
    elif panel == "b":
        draw_panel_b(ax)
    elif panel == "d":
        draw_panel_d(ax)
    elif panel == "f":
        draw_f_architecture(ax)
    else:
        raise ValueError(f"Unknown panel: {panel}")
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/figures"),
        help="Directory to save the figure assets.",
    )
    parser.add_argument(
        "--basename",
        default="ca_world_model_architecture",
        help="Base filename for the saved PNG and PDF.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    outputs = [
        ("a", "pure_rollout"),
        ("b", "ca_structured"),
        ("d", "long_latent_rollout"),
        ("f", "f_dynamics"),
    ]
    for panel, suffix in outputs:
        fig = build_single_panel_figure(panel)
        png_path = args.outdir / f"{args.basename}_{suffix}.png"
        pdf_path = args.outdir / f"{args.basename}_{suffix}.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure to {png_path}")
        print(f"Saved figure to {pdf_path}")


if __name__ == "__main__":
    main()
