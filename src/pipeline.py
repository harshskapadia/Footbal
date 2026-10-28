"""End-to-end pipeline: detection -> tracking -> teams -> pitch mapping -> analytics -> video.

Each stage lives in its own module and can also be run on its own (``src.detect``,
``src.track``, ``src.assign_teams``, ``src.map_pitch``, ``src.analyze``); this module
chains them and degrades gracefully when the pitch model is not available.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analyze import run_analytics
from src.assign_teams import run_team_assignment, save_team_colours
from src.detect import run_detection
from src.map_pitch import load_or_detect_keypoints, run_pitch_mapping
from src.pitch.mapping import project_ball, project_tracks
from src.pitch.template import PitchTemplate
from src.track import clip_length, track_clip
from src.utils.cli import RunContext
from src.utils.progress import ProgressCallback, progress
from src.utils.video import get_video_info, open_writer, read_frames, to_h264
from src.visualization.final import FinalAnnotator
from src.visualization.minimap import Minimap

log = logging.getLogger(__name__)

# Share of total run time per stage, used to report overall progress.
STAGE_WEIGHTS = {"detect": 0.55, "track": 0.03, "teams": 0.07, "keypoints": 0.15, "analytics": 0.02,
                 "render": 0.18}
DEFAULT_TEAM_BGR = {0: (60, 60, 220), 1: (220, 160, 60)}


@dataclass
class PipelineOptions:
    """Optional inputs that replace a model's output (useful for testing and re-runs)."""

    detections_csv: Path | None = None
    keypoints_csv: Path | None = None
    render_video: bool = True


@dataclass
class PipelineResult:
    """Summary statistics, output folder and any warnings raised on the way."""

    root: Path
    summary: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def _overall(callback: ProgressCallback | None) -> ProgressCallback | None:
    """Convert per-stage progress to overall progress across all stages."""
    if callback is None:
        return None
    order = list(STAGE_WEIGHTS)

    def report(fraction: float, stage: str) -> None:
        done = sum(STAGE_WEIGHTS[s] for s in order[: order.index(stage)]) if stage in order else 0.0
        callback(min(done + STAGE_WEIGHTS.get(stage, 0.0) * fraction, 1.0), stage)

    return report


def team_summary_table(summary: dict[str, Any]) -> pd.DataFrame:
    """One row per team with the headline numbers (for CSV and the dashboard)."""
    rows = []
    for team in (0, 1):
        row = {"team_id": team, "possession_pct": summary["possession_pct"][team],
               **{f"passes_{k}": v for k, v in summary["passes"][team].items()}}
        if "teams" in summary:
            row.update(summary["teams"][team])
        rows.append(row)
    return pd.DataFrame(rows)


def render_final(
    ctx: RunContext, tracks: pd.DataFrame, positions: pd.DataFrame, ball: pd.DataFrame,
    ball_positions: pd.DataFrame, team_bgr: dict[int, tuple[int, int, int]],
    callback: ProgressCallback | None = None,
) -> Path:
    """Write ``annotated.mp4`` with players, speeds, ball, possession bar and minimap."""
    possession = pd.read_csv(ctx.paths.possession)
    speeds = pd.read_csv(ctx.paths.speeds) if ctx.paths.speeds.exists() else None
    people = tracks[["frame", "tracker_id", "x1", "y1", "x2", "y2"]].merge(
        positions.drop(columns=["x_px", "y_px"]), on=["frame", "tracker_id"], how="left")
    if speeds is not None:
        people = people.merge(speeds, on=["frame", "tracker_id"], how="left")
    people_by_frame = {f: g for f, g in people.groupby("frame")}
    empty = people.iloc[:0]

    team = possession["team_in_possession"].to_numpy()
    counts = np.column_stack([np.cumsum(team == 0), np.cumsum(team == 1)])
    template = PitchTemplate.from_config(ctx.cfg)
    annotator = FinalAnnotator(team_bgr, Minimap(template, team_bgr))
    ball_idx = ball.set_index("frame")
    ball_m = ball_positions.set_index("frame")[["x_m", "y_m"]]
    owner = possession.set_index("frame")["owner_id"]

    info = get_video_info(ctx.input_path)
    n = len(ball)
    with open_writer(ctx.paths.annotated_video, info) as sink:
        for f, frame in enumerate(progress(read_frames(ctx.input_path, n), n, "render", callback)):
            total = counts[f].sum()
            share = {0: 100 * counts[f, 0] / total, 1: 100 * counts[f, 1] / total} if total else {0: 50, 1: 50}
            sink.write_frame(annotator(frame, people_by_frame.get(f, empty), ball_idx.loc[f], int(owner.loc[f]),
                                       share, ball_m.loc[f].to_numpy(float)))
    return ctx.paths.annotated_video


def run_pipeline(
    ctx: RunContext, options: PipelineOptions = PipelineOptions(), callback: ProgressCallback | None = None
) -> PipelineResult:
    """Run every stage on ``ctx.input_path`` and write all outputs to ``ctx.paths.root``."""
    cb = _overall(callback)
    warnings: list[str] = []
    paths, cfg = ctx.paths, ctx.cfg
    n_frames, fps = clip_length(ctx.input_path, ctx.max_frames)

    # Stage 2: detection
    if options.detections_csv is not None:
        detections = pd.read_csv(options.detections_csv)
        detections = detections[detections["frame"] < n_frames]
    else:
        detections = run_detection(ctx.input_path, cfg, ctx.max_frames, callback=cb)
    detections.to_csv(paths.detections, index=False)

    # Stage 3: tracking
    tracks, ball = track_clip(detections, n_frames, fps, cfg, cb)
    tracks.to_csv(paths.tracks, index=False)
    ball.to_csv(paths.ball, index=False)

    # Stage 4: teams
    teams: pd.DataFrame | None = None
    team_bgr = DEFAULT_TEAM_BGR
    try:
        teams, clf = run_team_assignment(ctx, tracks, cb)
        team_bgr = dict(enumerate(clf.colours_bgr()))
    except ValueError as e:
        warnings.append(f"Team assignment skipped: {e}")

    # Stage 5: pitch mapping
    try:
        keypoints = load_or_detect_keypoints(ctx, n_frames, options.keypoints_csv, cb)
        positions, ball_positions, _ = run_pitch_mapping(ctx, tracks, ball, teams, keypoints)
    except FileNotFoundError as e:
        warnings.append(str(e).splitlines()[0] + " Speed, distance, heatmaps and minimap are skipped.")
        template = PitchTemplate.from_config(cfg)
        no_mapping = [None] * n_frames
        positions = project_tracks(tracks, teams, no_mapping, template, 0.0)
        ball_positions = project_ball(ball, no_mapping, template, 0.0)
        positions.to_csv(paths.positions, index=False)
        ball_positions.to_csv(paths.ball_positions, index=False)
    if teams is None:
        positions["team_id"], positions["role"] = -1, positions.get("class_name", "player")

    # Stage 6: analytics
    summary = run_analytics(ctx, positions, ball_positions, fps)
    team_summary_table(summary).to_csv(paths.root / "team_summary.csv", index=False)

    # Stage 7: output video
    if options.render_video:
        render_final(ctx, tracks, positions, ball, ball_positions, team_bgr, cb)
        if to_h264(paths.annotated_video, paths.web_video) is None:
            warnings.append("Could not create the H.264 copy; the dashboard may not play the video.")

    summary["warnings"] = warnings
    summary["team_colours_hex"] = {t: "#{:02X}{:02X}{:02X}".format(*bgr[::-1]) for t, bgr in team_bgr.items()}
    paths.stats_json.write_text(json.dumps(summary, indent=2, default=float))
    if teams is None:
        save_team_colours_default(paths.team_colours)
    for w in warnings:
        log.warning(w)
    return PipelineResult(paths.root, summary, warnings)


def save_team_colours_default(path: Path) -> None:
    """Write the fallback team colours when clustering was impossible."""
    path.write_text(json.dumps({str(t): {"bgr": list(c), "hex": "#{:02X}{:02X}{:02X}".format(*c[::-1])}
                                for t, c in DEFAULT_TEAM_BGR.items()}, indent=2))


__all__ = ["PipelineOptions", "PipelineResult", "run_pipeline", "save_team_colours"]
