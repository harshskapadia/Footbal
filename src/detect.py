"""Stage 2 CLI: run detection on a clip, write an annotated video and a CSV of boxes.

Example:
    python -m src.detect --input data/raw/0bfacc_0.mp4 --max-frames 150
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import supervision as sv
from tqdm import tqdm

from src.detection.classes import BALL, CLASS_NAMES
from src.detection.detector import ObjectDetector, build_detector, keep_best_ball
from src.detection.slicing import SlicedBallDetector, build_ball_detector
from src.utils.config import load_config, resolve_path
from src.utils.device import select_device
from src.utils.video import get_video_info, open_writer, read_frames
from src.visualization.draw import DetectionAnnotator

log = logging.getLogger(__name__)
CSV_COLUMNS = ["frame", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]


def detect_frame(
    frame, detector: ObjectDetector, ball_detector: SlicedBallDetector | None = None
) -> sv.Detections:
    """Detect people and the ball in one frame.

    With a sliced ball detector, its ball replaces the main detector's ball,
    but the main detector's ball is kept when slicing finds nothing.
    """
    detections = detector(frame)
    if ball_detector is not None:
        sliced_ball = ball_detector(frame)
        if len(sliced_ball):
            detections = sv.Detections.merge([detections[detections.class_id != BALL], sliced_ball])
    return keep_best_ball(detections)


def detections_to_rows(frame_idx: int, detections: sv.Detections) -> list[dict]:
    """Flatten detections into CSV rows."""
    return [
        {
            "frame": frame_idx, "class_id": int(c), "class_name": CLASS_NAMES[int(c)],
            "confidence": round(float(p), 4),
            "x1": round(float(b[0]), 1), "y1": round(float(b[1]), 1),
            "x2": round(float(b[2]), 1), "y2": round(float(b[3]), 1),
        }
        for b, c, p in zip(detections.xyxy, detections.class_id, detections.confidence)
    ]


def run_detection(
    input_path: Path, output_dir: Path, cfg: dict, max_frames: int | None = None
) -> tuple[Path, Path]:
    """Run detection over a video and write the annotated video plus CSV.

    Returns:
        Paths of the annotated video and the detections CSV.
    """
    device = select_device(cfg.get("device", "auto"))
    log.info("Using device: %s", device)
    detector = build_detector(cfg, device)
    ball_detector = build_ball_detector(cfg, device) if cfg["ball"]["use_slicing"] else None
    annotator = DetectionAnnotator()

    info = get_video_info(input_path)
    total = min(info.total_frames or 0, max_frames or info.total_frames or 0) or None
    video_out = output_dir / f"{input_path.stem}_detections.mp4"
    csv_out = output_dir / f"{input_path.stem}_detections.csv"

    rows: list[dict] = []
    n_frames = 0
    start = time.perf_counter()
    with open_writer(video_out, info) as sink:
        for i, frame in enumerate(tqdm(read_frames(input_path, max_frames), total=total, desc="detect")):
            detections = detect_frame(frame, detector, ball_detector)
            rows.extend(detections_to_rows(i, detections))
            sink.write_frame(annotator(frame, detections))
            n_frames += 1
    elapsed = time.perf_counter() - start

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df.to_csv(csv_out, index=False)
    log.info("%d frames in %.1fs (%.2f fps)", n_frames, elapsed, n_frames / max(elapsed, 1e-9))
    log.info("Counts per class: %s", df["class_name"].value_counts().to_dict())
    return video_out, csv_out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Detect players, referees, goalkeepers and the ball.")
    p.add_argument("--input", type=Path, help="Input video (default: paths.input_video in config)")
    p.add_argument("--output-dir", type=Path, help="Output directory (default: paths.output_dir)")
    p.add_argument("--config", type=Path, default=None, help="Config YAML (default: config.yaml)")
    p.add_argument("--max-frames", type=int, default=None, help="Only process the first N frames")
    p.add_argument("--device", default=None, help="Override device: auto, cpu, cuda:0, mps")
    p.add_argument("--no-slicing", action="store_true", help="Disable sliced ball inference")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    if args.no_slicing:
        cfg["ball"]["use_slicing"] = False
    input_path = resolve_path(args.input or cfg["paths"]["input_video"])
    output_dir = resolve_path(args.output_dir or cfg["paths"]["output_dir"])
    max_frames = args.max_frames or cfg.get("output", {}).get("max_frames")
    video_out, csv_out = run_detection(input_path, output_dir, cfg, max_frames)
    print(f"Annotated video: {video_out}\nDetections CSV:  {csv_out}")


if __name__ == "__main__":
    main()
