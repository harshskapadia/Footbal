"""Stage 2 CLI: run detection on a clip, write an annotated video and a CSV of boxes.

Example:
    python -m src.detect --input data/raw/0bfacc_0.mp4 --max-frames 150
    # -> outputs/0bfacc_0/detections.csv and detections.mp4
"""

from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import supervision as sv

from src.detection.classes import BALL, class_ids, subset
from src.detection.detector import ObjectDetector, build_detector, keep_best_ball
from src.detection.io import CSV_COLUMNS, detections_to_rows
from src.detection.slicing import SlicedBallDetector, build_ball_detector
from src.utils.cli import base_parser, make_context
from src.utils.device import select_device
from src.utils.progress import ProgressCallback, progress
from src.utils.video import frame_count, get_video_info, open_writer, read_frames
from src.visualization.draw import DetectionAnnotator

log = logging.getLogger(__name__)


def detect_frame(
    frame: np.ndarray, detector: ObjectDetector, ball_detector: SlicedBallDetector | None = None
) -> sv.Detections:
    """Detect people and the ball in one frame.

    With a sliced ball detector, its ball replaces the main detector's ball,
    but the main detector's ball is kept when slicing finds nothing.
    """
    detections = detector(frame)
    if ball_detector is not None:
        sliced_ball = ball_detector(frame)
        if len(sliced_ball):
            detections = sv.Detections.merge([subset(detections, class_ids(detections) != BALL), sliced_ball])
    return keep_best_ball(detections)


def run_detection(
    input_path: Path,
    cfg: dict[str, Any],
    max_frames: int | None = None,
    video_out: Path | None = None,
    callback: ProgressCallback | None = None,
    frame_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Run detection over a clip.

    Args:
        input_path: Input video.
        cfg: Loaded config.
        max_frames: Only process the first N frames.
        video_out: If given, also write an annotated video here.
        callback: Optional progress callback ``(fraction, stage)``.
        frame_mask: Optional boolean per frame; frames marked False are skipped
            (e.g. close-ups and replays in highlights).

    Returns:
        One row per box (see ``src.detection.io.CSV_COLUMNS``).
    """
    device = select_device(cfg.get("device", "auto"))
    log.info("Detection device: %s", device)
    detector = build_detector(cfg, device)
    ball_detector = build_ball_detector(cfg, device) if cfg["ball"]["use_slicing"] else None
    annotator = DetectionAnnotator()
    info = get_video_info(input_path)
    total = frame_count(info, max_frames)

    rows: list[dict] = []
    n_frames = 0
    start = time.perf_counter()
    with contextlib.ExitStack() as stack:
        sink = stack.enter_context(open_writer(video_out, info)) if video_out else None
        for i, frame in enumerate(progress(read_frames(input_path, max_frames), total, "detect", callback)):
            if frame_mask is not None and (i >= len(frame_mask) or not frame_mask[i]):
                detections = sv.Detections.empty()
            else:
                detections = detect_frame(frame, detector, ball_detector)
            rows.extend(detections_to_rows(i, detections))
            if sink is not None:
                sink.write_frame(annotator(frame, detections))
            n_frames += 1
    elapsed = time.perf_counter() - start

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    log.info("%d frames in %.1fs (%.2f fps)", n_frames, elapsed, n_frames / max(elapsed, 1e-9))
    log.info("Counts per class: %s", df["class_name"].value_counts().to_dict())
    return df


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = base_parser("Detect players, referees, goalkeepers and the ball.")
    parser.add_argument("--no-slicing", action="store_true", help="Disable sliced ball inference")
    args = parser.parse_args(argv)
    ctx = make_context(args)
    if args.no_slicing:
        ctx.cfg["ball"]["use_slicing"] = False
    df = run_detection(ctx.input_path, ctx.cfg, ctx.max_frames, ctx.paths.detections_video)
    df.to_csv(ctx.paths.detections, index=False)
    print(f"Annotated video: {ctx.paths.detections_video}\nDetections CSV:  {ctx.paths.detections}")


if __name__ == "__main__":
    main()
