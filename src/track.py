"""Stage 3 CLI: track people with ByteTrack and clean up the ball trajectory.

Reuses a stage 2 detections CSV when given (fast, no model needed), otherwise
runs detection first.

Examples:
    python -m src.track --input data/raw/0bfacc_0.mp4 --detections outputs/0bfacc_0_detections.csv
    python -m src.track --input data/raw/0bfacc_0.mp4 --max-frames 150
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import supervision as sv
from tqdm import tqdm

from src.detection.classes import BALL
from src.detection.io import BOX_COLUMNS, detections_to_rows, group_by_frame, load_detections_csv
from src.tracking.ball import clean_ball_track
from src.tracking.tracker import build_tracker, stabilise_track_classes, track_people
from src.utils.config import load_config, resolve_path
from src.utils.device import select_device
from src.utils.video import get_video_info, open_writer, read_frames
from src.visualization.draw import TrackAnnotator

log = logging.getLogger(__name__)


def iter_detections(
    input_path: Path, cfg: dict[str, Any], detections_csv: Path | None, max_frames: int | None
) -> Iterator[tuple[int, np.ndarray, sv.Detections]]:
    """Yield ``(frame_idx, frame, detections)`` from a CSV or by running the detector."""
    if detections_csv:
        per_frame = load_detections_csv(detections_csv)
        for i, frame in enumerate(read_frames(input_path, max_frames)):
            yield i, frame, per_frame.get(i, sv.Detections.empty())
        return

    from src.detect import detect_frame
    from src.detection.detector import build_detector
    from src.detection.slicing import build_ball_detector

    device = select_device(cfg.get("device", "auto"))
    log.info("No detections CSV given: running detection on %s", device)
    detector = build_detector(cfg, device)
    ball_detector = build_ball_detector(cfg, device) if cfg["ball"]["use_slicing"] else None
    for i, frame in enumerate(read_frames(input_path, max_frames)):
        yield i, frame, detect_frame(frame, detector, ball_detector)


def track_clip(
    input_path: Path, cfg: dict[str, Any], detections_csv: Path | None = None, max_frames: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Track people and clean the ball trajectory for a whole clip.

    Returns:
        ``tracks`` (one row per person per frame, with ``tracker_id``) and
        ``ball`` (one row per frame, NaN where unknown, ``interpolated`` flag).
    """
    info = get_video_info(input_path)
    tracker = build_tracker(cfg, info.fps)
    track_rows: list[dict] = []
    ball_rows: list[dict] = []
    n_frames = 0
    total = min(info.total_frames, max_frames) if max_frames else info.total_frames
    for i, frame, detections in tqdm(iter_detections(input_path, cfg, detections_csv, max_frames),
                                     total=total, desc="track"):
        track_rows.extend(detections_to_rows(i, track_people(tracker, detections, frame)))
        if len(detections):
            ball_rows.extend(detections_to_rows(i, detections[detections.class_id == BALL]))
        n_frames += 1

    tracks = stabilise_track_classes(pd.DataFrame(track_rows))
    bi = cfg["ball_interpolation"]
    ball_detections = pd.DataFrame(ball_rows, columns=["frame", "class_id", "confidence", *BOX_COLUMNS])
    ball = clean_ball_track(ball_detections, n_frames, bi["max_gap_frames"], bi["max_speed_px_per_frame"])
    return tracks, ball


def render_tracks(input_path: Path, video_out: Path, tracks: pd.DataFrame, ball: pd.DataFrame) -> None:
    """Write a video with track IDs and the (interpolated) ball drawn on each frame."""
    info = get_video_info(input_path)
    people_by_frame = group_by_frame(tracks) if not tracks.empty else {}
    ball = ball.set_index("frame")
    annotator = TrackAnnotator()
    with open_writer(video_out, info) as sink:
        for i, frame in enumerate(tqdm(read_frames(input_path, len(ball)), total=len(ball), desc="render")):
            row = ball.loc[i]
            ball_det = sv.Detections.empty()
            if not np.isnan(row["x1"]):
                ball_det = sv.Detections(xyxy=row[BOX_COLUMNS].to_numpy(dtype=float)[None],
                                         class_id=np.array([BALL]))
            people = people_by_frame.get(i, sv.Detections.empty())
            sink.write_frame(annotator(frame, people, ball_det, bool(row["interpolated"])))


def summarise(tracks: pd.DataFrame, ball: pd.DataFrame) -> dict[str, Any]:
    """Return headline tracking numbers for logging."""
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Track players and the ball through a clip.")
    p.add_argument("--input", type=Path, help="Input video (default: paths.input_video in config)")
    p.add_argument("--detections", type=Path, help="Stage 2 detections CSV to reuse")
    p.add_argument("--output-dir", type=Path, help="Output directory (default: paths.output_dir)")
    p.add_argument("--config", type=Path, default=None, help="Config YAML (default: config.yaml)")
    p.add_argument("--max-frames", type=int, default=None, help="Only process the first N frames")
    p.add_argument("--device", default=None, help="Override device: auto, cpu, cuda:0, mps")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    input_path = resolve_path(args.input or cfg["paths"]["input_video"])
    output_dir = resolve_path(args.output_dir or cfg["paths"]["output_dir"])
    max_frames = args.max_frames or cfg.get("output", {}).get("max_frames")
    detections_csv = resolve_path(args.detections) if args.detections else None

    tracks, ball = track_clip(input_path, cfg, detections_csv, max_frames)
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks_csv = output_dir / f"{input_path.stem}_tracks.csv"
    ball_csv = output_dir / f"{input_path.stem}_ball.csv"
    video_out = output_dir / f"{input_path.stem}_tracks.mp4"
    tracks.to_csv(tracks_csv, index=False)
    ball.to_csv(ball_csv, index=False)
    render_tracks(input_path, video_out, tracks, ball)

    log.info("Summary: %s", summarise(tracks, ball))
    print(f"Annotated video: {video_out}\nTracks CSV:      {tracks_csv}\nBall CSV:        {ball_csv}")


if __name__ == "__main__":
    main()
