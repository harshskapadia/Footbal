"""Static charts on a top-down pitch: team heatmaps and passing networks.

Colours follow the validated reference palette: one sequential blue ramp for
magnitude (shared scale so teams are comparable) and fixed categorical slots for
team identity (team 0 = blue, team 1 = orange) rather than kit colours, which can
be white or too close to the background.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from src.pitch.draw import pitch_polylines  # noqa: E402
from src.pitch.template import PitchTemplate  # noqa: E402

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
PITCH_LINE = "#b5b4ae"
EDGE = "#52514e"
TEAM_COLOURS = {0: "#2a78d6", 1: "#eb6834"}
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", [SURFACE, "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def draw_pitch_axes(ax: plt.Axes, template: PitchTemplate) -> None:
    """Draw recessive pitch markings on a matplotlib axis (metres, y pointing down)."""
    ax.set_facecolor(SURFACE)
    for line in pitch_polylines(template):
        ax.plot(line[:, 0], line[:, 1], color=PITCH_LINE, lw=1, zorder=1)
    ax.set_xlim(-2, template.length + 2)
    ax.set_ylim(template.width + 2, -2)
    ax.set_aspect("equal")
    ax.axis("off")


def heatmap_grid(xy: np.ndarray, template: PitchTemplate, bin_m: float = 1.0, sigma_m: float = 2.5) -> np.ndarray:
    """Share of time (%) spent in each ``bin_m`` square, Gaussian-smoothed."""
    import cv2

    xy = xy[np.isfinite(xy).all(axis=1)]
    nx, ny = int(np.ceil(template.length / bin_m)), int(np.ceil(template.width / bin_m))
    grid, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], bins=[ny, nx],
                                range=[[0, template.width], [0, template.length]])
    if grid.sum() == 0:
        return grid
    grid = cv2.GaussianBlur(grid, (0, 0), sigma_m / bin_m)
    return 100 * grid / grid.sum()


def plot_heatmaps(
    positions: pd.DataFrame, template: PitchTemplate, team_names: dict[int, str],
    bin_m: float = 1.0, sigma_m: float = 2.5,
) -> dict[int, plt.Figure]:
    """One heatmap figure per team, on a shared colour scale."""
    grids = {t: heatmap_grid(positions.loc[positions["team_id"] == t, ["x_m", "y_m"]].to_numpy(float),
                             template, bin_m, sigma_m) for t in (0, 1)}
    vmax = max(float(g.max()) for g in grids.values()) or 1.0
    figs = {}
    for team, grid in grids.items():
        fig, ax = plt.subplots(figsize=(8, 5.6), facecolor=SURFACE)
        draw_pitch_axes(ax, template)
        im = ax.imshow(grid, extent=(0, template.length, template.width, 0), cmap=SEQUENTIAL,
                       vmin=0, vmax=vmax, interpolation="bilinear", zorder=0)
        for line in pitch_polylines(template):          # lines on top of the heat
            ax.plot(line[:, 0], line[:, 1], color=PITCH_LINE, lw=1, alpha=0.8, zorder=2)
        cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03)
        cbar.set_label("Share of time per m² (%)", color=TEXT_SECONDARY, fontsize=9)
        cbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        cbar.outline.set_visible(False)
        ax.set_title(f"{team_names.get(team, f'Team {team}')}: where the team spent its time",
                     color=TEXT_PRIMARY, fontsize=12, loc="left")
        fig.tight_layout()
        figs[team] = fig
    return figs


def plot_passing_network(
    nodes: pd.DataFrame, edges: pd.DataFrame, template: PitchTemplate, team: int, title: str,
) -> plt.Figure:
    """Players at their average position; line width = completed passes between the pair."""
    fig, ax = plt.subplots(figsize=(8, 5.6), facecolor=SURFACE)
    draw_pitch_axes(ax, template)
    pos = nodes.set_index("tracker_id")[["x_m", "y_m"]]
    max_count = max(int(edges["count"].max()), 1) if len(edges) else 1
    for e in edges.itertuples():
        if e.from_id in pos.index and e.to_id in pos.index:
            (x1, y1), (x2, y2) = pos.loc[e.from_id], pos.loc[e.to_id]
            ax.plot([x1, x2], [y1, y2], color=EDGE, alpha=0.55, lw=1.5 + 4.5 * e.count / max_count,
                    solid_capstyle="round", zorder=2)
    if len(nodes):
        sizes = 120 + 380 * nodes["passes"] / nodes["passes"].max()
        ax.scatter(nodes["x_m"], nodes["y_m"], s=sizes, color=TEAM_COLOURS[team],
                   edgecolors=SURFACE, linewidths=2, zorder=3)
        for n in nodes.itertuples():
            ax.annotate(f"#{n.tracker_id}", (n.x_m, n.y_m), xytext=(0, -14), textcoords="offset points",
                        ha="center", va="top", fontsize=8, color=TEXT_PRIMARY, zorder=4)
    else:
        ax.text(template.length / 2, template.width / 2, "No completed passes detected",
                ha="center", color=TEXT_SECONDARY)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=12, loc="left")
    fig.text(0.01, 0.01, "Node size = passes made + received. Line width = passes between the pair.",
             color=TEXT_SECONDARY, fontsize=8)
    fig.tight_layout()
    return fig
