"""Stage 6 CLI: possession, passes, speed/distance, heatmaps and passing networks.

Needs stage 5 outputs (``positions.csv``, ``ball_positions.csv``) in ``outputs/<clip>/``.
Without a pitch mapping (no ``x_m``), possession and passes are computed in pixels
and the metric outputs (speed, distance, heatmaps, networks) are skipped.

Example:
    python -m src.analyze --input data/raw/0bfacc_0.mp4
"""

from __future__ import annotations

import json
import logging
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.analytics.kinematics import compute_kinematics
from src.analytics.passes import detect_passes, pass_summary, passing_network
from src.analytics.plots import plot_heatmaps, plot_passing_network
from src.analytics.possession import compute_possession, possession_share
from src.pitch.template import PitchTemplate
from src.utils.cli import RunContext, base_parser, make_context
from src.utils.video import get_video_info

log = logging.getLogger(__name__)


def has_pitch_positions(positions: pd.DataFrame, min_fraction: float = 0.5) -> bool:
    """True if most rows were mapped to pitch metres."""
    return bool("x_m" in positions and positions["x_m"].notna().mean() >= min_fraction)


def team_stats(stats: pd.DataFrame) -> dict[int, dict[str, float]]:
    """Total distance and top speed per team."""
    out = {}
    for team in (0, 1):
        s = stats[stats["team_id"] == team]
        out[team] = {"players_tracked": int(len(s)),
                     "total_distance_m": round(float(s["distance_m"].sum()), 1),
                     "top_speed_kmh": round(float(s["max_speed_kmh"].max()), 1) if len(s) else 0.0}
    return out


def run_analytics(
    ctx: RunContext, positions: pd.DataFrame, ball_positions: pd.DataFrame, fps: float
) -> dict[str, Any]:
    """Compute every stat, write CSVs and figures to the run folder, return the summary."""
    a = ctx.cfg["analytics"]
    template = PitchTemplate.from_config(ctx.cfg)
    metric = has_pitch_positions(positions)

    possession = compute_possession(positions, ball_positions, ctx.cfg, fps, use_pitch=metric)
    possession.to_csv(ctx.paths.possession, index=False)
    passes = detect_passes(possession, positions if metric else None, fps, a["max_pass_gap_s"],
                           a["min_pass_distance_m"])
    passes.to_csv(ctx.paths.passes, index=False)

    summary: dict[str, Any] = {
        "clip": ctx.input_path.name,
        "frames": int(len(ball_positions)),
        "fps": fps,
        "duration_s": round(len(ball_positions) / fps, 1),
        "pitch_mapping": metric,
        "possession_pct": possession_share(possession),
        "passes": pass_summary(passes),
    }
    if metric:
        speeds, stats = compute_kinematics(
            positions, fps, max(1, round(a["speed_window_s"] * fps)), a["max_speed_kmh"])
        speeds.to_csv(ctx.paths.speeds, index=False)
        stats.to_csv(ctx.paths.player_stats, index=False)
        summary["teams"] = team_stats(stats)
        summary["top_players_by_distance"] = stats[stats["team_id"].isin([0, 1])].head(5)[
            ["tracker_id", "team_id", "distance_m", "max_speed_kmh"]].to_dict("records")
        names = {0: "Team 0", 1: "Team 1"}
        for team, fig in plot_heatmaps(positions, template, names, a["heatmap_bin_m"], a["heatmap_sigma_m"]).items():
            fig.savefig(ctx.paths.heatmap(team), dpi=120)
            plt.close(fig)
        for team in (0, 1):
            nodes, edges = passing_network(passes, positions, team)
            fig = plot_passing_network(nodes, edges, template, team, f"{names[team]}: passing network")
            fig.savefig(ctx.paths.passing_network(team), dpi=120)
            plt.close(fig)
    else:
        log.warning("No pitch mapping: possession/passes measured in pixels; speed, distance, "
                    "heatmaps and passing networks skipped.")
    ctx.paths.stats_json.write_text(json.dumps(summary, indent=2, default=float))
    return summary


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = base_parser("Compute match statistics from pitch positions.").parse_args(argv)
    ctx = make_context(args)
    if not ctx.paths.positions.exists():
        raise SystemExit(f"{ctx.paths.positions} not found: run `python -m src.map_pitch` first.")
    fps = float(get_video_info(ctx.input_path).fps)
    summary = run_analytics(ctx, pd.read_csv(ctx.paths.positions), pd.read_csv(ctx.paths.ball_positions), fps)
    print(json.dumps({k: summary[k] for k in ("possession_pct", "passes")}, indent=2))
    print(f"Stats: {ctx.paths.stats_json}")


if __name__ == "__main__":
    main()
