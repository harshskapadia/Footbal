"""Stage 3 CLI: track people with ByteTrack and clean up the ball trajectory.

Reuses ``outputs/<clip>/detections.csv`` when it exists, otherwise runs detection first.

Example:
    python -m src.track --input data/raw/0bfacc_0.mp4
    # -> outputs/0bfacc_0/tracks.csv, ball.csv and tracks.mp4
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import supervision as sv

from src.detection.classes import BALL
from src.detection.io import BOX_COLUMNS, detections_to_rows, group_by_frame
from src.shots.segmentation import shot_runs
from src.tracking.ball import clean_ball_track
from src.tracking.tracker import build_tracker, stabilise_track_classes, track_people
from src.utils.cli import RunContext, base_parser, make_context
from src.utils.progress import ProgressCallback, progress
from src.utils.video import frame_count, get_video_info, open_writer, read_frames
from src.visualization.draw import TrackAnnotator

log = logging.getLogger(__name__)


def clip_length(input_path: Path, max_frames: int | None) -> tuple[int, float]:
    """Return the number of frames to process and the clip's fps."""
    info = get_video_info(input_path)
    n = frame_count(info, max_frames)
    if n is None:  # container without a frame count: count by decoding
        n = sum(1 for _ in read_frames(input_path))
    return n, float(info.fps)


def track_clip(
    detections: pd.DataFrame,
    n_frames: int,
    fps: float,
    cfg: dict[str, Any],
    callback: ProgressCallback | None = None,
    shot_ids: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Track people and clean the ball trajectory from stage 2 detections.

    Only boxes are needed, not pixels, so this does not read the video.

    With ``shot_ids`` (one per frame, -1 = not analysed), the tracker restarts
    at every cut so IDs never jump between camera shots (new IDs continue from
    the highest one used so far), frames outside analysed shots get no tracks,
    and ball gaps are only interpolated within a shot.

    Returns:
        ``tracks`` (one row per person per frame, with ``tracker_id``) and
        ``ball`` (one row per frame, NaN where unknown, ``interpolated`` flag).
    """
    shots = shot_ids if shot_ids is not None else np.zeros(n_frames, dtype=int)
    per_frame = group_by_frame(detections) if not detections.empty else {}
    rows: list[dict] = []
    tracker = build_tracker(cfg, fps)
    offset, next_offset, current = 0, 0, shots[0] if n_frames else 0
    for i in progress(range(n_frames), n_frames, "track", callback):
        if shots[i] != current:                   # cut: fresh tracker, IDs continue upwards
            tracker, offset, current = build_tracker(cfg, fps), next_offset, shots[i]
        if shots[i] < 0:
            continue
        tracked = track_people(tracker, per_frame.get(i, sv.Detections.empty()))
        if len(tracked) and tracked.tracker_id is not None:
            tracked.tracker_id = tracked.tracker_id + offset
            next_offset = max(next_offset, int(tracked.tracker_id.max()) + 1)
        rows.extend(detections_to_rows(i, tracked))

    tracks = stabilise_track_classes(pd.DataFrame(rows))
    bi = cfg["ball_interpolation"]
    balls = detections[detections["class_id"] == BALL] if not detections.empty else detections
    parts = []
    for shot, start, end in shot_runs(shots):
        in_shot = balls[(balls["frame"] >= start) & (balls["frame"] < end)] if len(balls) else balls
        local = in_shot.assign(frame=in_shot["frame"] - start) if len(in_shot) else in_shot
        part = clean_ball_track(local, end - start, bi["max_gap_frames"], bi["max_speed_px_per_frame"])
        part["frame"] += start
        if shot < 0:                              # not analysed: no ball either
            part[[*BOX_COLUMNS, "confidence"]] = np.nan
        parts.append(part)
    ball = pd.concat(parts).sort_values("frame").reset_index(drop=True) if parts else \
        clean_ball_track(balls, n_frames, bi["max_gap_frames"], bi["max_speed_px_per_frame"])
    return tracks, ball


def ball_detection(row: pd.Series) -> sv.Detections:
    """Return the ball in one row of the ball table as ``sv.Detections`` (empty if unknown)."""
    if np.isnan(row["x1"]):
        return sv.Detections.empty()
    return sv.Detections(xyxy=row[BOX_COLUMNS].to_numpy(dtype=float)[None], class_id=np.array([BALL]))


def render_tracks(input_path: Path, video_out: Path, tracks: pd.DataFrame, ball: pd.DataFrame) -> None:
    """Write a debug video with track IDs and the (interpolated) ball."""
    info = get_video_info(input_path)
    people_by_frame = group_by_frame(tracks) if not tracks.empty else {}
    ball = ball.set_index("frame")
    annotator = TrackAnnotator()
    with open_writer(video_out, info) as sink:
        for i, frame in enumerate(progress(read_frames(input_path, len(ball)), len(ball), "render")):
            row = ball.loc[i]
            people = people_by_frame.get(i, sv.Detections.empty())
            sink.write_frame(annotator(frame, people, ball_detection(row), bool(row["interpolated"])))


def summarise(tracks: pd.DataFrame, ball: pd.DataFrame) -> dict[str, Any]:
    """Return headline tracking numbers."""
    detected = int(ball["confidence"].notna().sum())
    filled = int(ball["interpolated"].sum())
    per_class = tracks.groupby("class_name")["tracker_id"].nunique().to_dict() if not tracks.empty else {}
    return {
        "frames": len(ball),
        "unique_ids_by_class": per_class,
        "ball_detected_frames": detected,
        "ball_interpolated_frames": filled,
        "ball_coverage": round((detected + filled) / max(len(ball), 1), 3),
    }


def load_or_detect(ctx: RunContext, callback: ProgressCallback | None = None) -> pd.DataFrame:
    """Load stage 2 detections from the run folder, running detection if missing."""
    if ctx.paths.detections.exists():
        log.info("Reusing %s", ctx.paths.detections)
        return pd.read_csv(ctx.paths.detections)
    from src.detect import run_detection

    df = run_detection(ctx.input_path, ctx.cfg, ctx.max_frames, callback=callback)
    df.to_csv(ctx.paths.detections, index=False)
    return df


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = base_parser("Track players and the ball through a clip.").parse_args(argv)
    ctx = make_context(args)
    detections = load_or_detect(ctx)
    n_frames, fps = clip_length(ctx.input_path, ctx.max_frames)
    tracks, ball = track_clip(detections, n_frames, fps, ctx.cfg)
    tracks.to_csv(ctx.paths.tracks, index=False)
    ball.to_csv(ctx.paths.ball, index=False)
    render_tracks(ctx.input_path, ctx.paths.tracks_video, tracks, ball)
    log.info("Summary: %s", summarise(tracks, ball))
    print(f"Annotated video: {ctx.paths.tracks_video}\nTracks CSV:      {ctx.paths.tracks}")


if __name__ == "__main__":
    main()
