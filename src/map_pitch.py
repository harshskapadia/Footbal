"""Stage 5 CLI: map players and ball from pixels to pitch metres.

Keypoints come from ``--keypoints`` (a CSV), an existing ``outputs/<clip>/keypoints.csv``,
or the pitch keypoint model (``models.pitch_detector`` in config.yaml).

Example:
    python -m src.map_pitch --input data/raw/0bfacc_0.mp4
    # -> outputs/0bfacc_0/positions.csv, ball_positions.csv, homographies.csv, pitch_overlay.mp4
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd

from src.pitch.draw import draw_pitch_overlay
from src.pitch.keypoints import build_keypoint_detector, keypoints_by_frame, keypoints_to_rows
from src.pitch.mapping import compute_homographies, project_ball, project_tracks
from src.pitch.template import PitchTemplate
from src.utils.cli import RunContext, base_parser, make_context
from src.utils.device import select_device
from src.utils.progress import ProgressCallback, progress
from src.utils.video import get_video_info, open_writer, read_frames

log = logging.getLogger(__name__)


def detect_keypoints(ctx: RunContext, n_frames: int, callback: ProgressCallback | None = None) -> pd.DataFrame:
    """Run the pitch keypoint model over the clip."""
    detector = build_keypoint_detector(ctx.cfg, select_device(ctx.cfg.get("device", "auto")))
    rows: list[dict] = []
    for f, frame in enumerate(progress(read_frames(ctx.input_path, n_frames), n_frames, "keypoints", callback)):
        rows.extend(keypoints_to_rows(f, *detector(frame)))
    return pd.DataFrame(rows)


def load_or_detect_keypoints(
    ctx: RunContext, n_frames: int, keypoints_csv: Path | None = None, callback: ProgressCallback | None = None
) -> pd.DataFrame:
    """Keypoints from a given CSV, the run folder, or the model (in that order)."""
    if keypoints_csv is not None:
        shutil.copyfile(keypoints_csv, ctx.paths.keypoints)
    if ctx.paths.keypoints.exists():
        log.info("Reusing keypoints from %s", ctx.paths.keypoints)
        return pd.read_csv(ctx.paths.keypoints)
    df = detect_keypoints(ctx, n_frames, callback)
    df.to_csv(ctx.paths.keypoints, index=False)
    return df


def run_pitch_mapping(
    ctx: RunContext, tracks: pd.DataFrame, ball: pd.DataFrame, teams: pd.DataFrame | None,
    keypoints: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Compute smoothed homographies and pitch positions; write them to the run folder."""
    info = get_video_info(ctx.input_path)
    n_frames = len(ball)
    homographies, table = compute_homographies(
        keypoints_by_frame(keypoints), n_frames, (info.width, info.height), ctx.cfg)
    table.to_csv(ctx.paths.homographies, index=False)

    template = PitchTemplate.from_config(ctx.cfg)
    margin = ctx.cfg["pitch"]["out_of_bounds_margin_m"]
    positions = project_tracks(tracks, teams, homographies, template, margin)
    ball_positions = project_ball(ball, homographies, template, margin)
    positions.to_csv(ctx.paths.positions, index=False)
    ball_positions.to_csv(ctx.paths.ball_positions, index=False)
    log.info("Homography: raw fit on %d/%d frames, mapped after smoothing on %d/%d",
             int(table["raw_fit"].sum()), n_frames, int(table["mapped"].sum()), n_frames)
    return positions, ball_positions, homographies


def render_overlay(ctx: RunContext, homographies: list, video_out: Path) -> None:
    """Write a video with the pitch markings projected through each frame's homography."""
    template = PitchTemplate.from_config(ctx.cfg)
    info = get_video_info(ctx.input_path)
    with open_writer(video_out, info) as sink:
        for f, frame in enumerate(progress(read_frames(ctx.input_path, len(homographies)),
                                           len(homographies), "overlay")):
            h = homographies[f]
            sink.write_frame(draw_pitch_overlay(frame, h, template) if h is not None else frame)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = base_parser("Map tracked players and the ball to pitch coordinates (metres).")
    parser.add_argument("--keypoints", type=Path, help="Pre-computed keypoints CSV (frame,keypoint,x,y,confidence)")
    parser.add_argument("--no-video", action="store_true", help="Skip the pitch overlay video")
    args = parser.parse_args(argv)
    ctx = make_context(args)
    if not ctx.paths.tracks.exists():
        raise SystemExit(f"{ctx.paths.tracks} not found: run `python -m src.track` first.")
    tracks, ball = pd.read_csv(ctx.paths.tracks), pd.read_csv(ctx.paths.ball)
    teams = pd.read_csv(ctx.paths.teams) if ctx.paths.teams.exists() else None
    try:
        keypoints = load_or_detect_keypoints(ctx, len(ball), args.keypoints)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from None
    _, _, homographies = run_pitch_mapping(ctx, tracks, ball, teams, keypoints)
    if not args.no_video:
        render_overlay(ctx, homographies, ctx.paths.root / "pitch_overlay.mp4")
    print(f"Positions:      {ctx.paths.positions}\nBall positions: {ctx.paths.ball_positions}")


if __name__ == "__main__":
    main()
