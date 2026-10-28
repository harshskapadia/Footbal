"""Run the whole pipeline on a clip.

Examples:
    python -m src.run --input data/raw/0bfacc_0.mp4
    python -m src.run --input clip.mp4 --max-frames 250 --device cpu
    # Synthetic match (detections and keypoints supplied by the simulator):
    python -m src.run --input data/raw/synthetic_match.mp4 \\
        --detections data/synthetic/synthetic_match/detections.csv \\
        --keypoints data/synthetic/synthetic_match/keypoints.csv

All outputs go to ``outputs/<clip name>/``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.pipeline import PipelineOptions, run_pipeline
from src.utils.cli import base_parser, make_context


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = base_parser("Football analytics: annotated video + match stats from a broadcast clip.")
    parser.add_argument("--detections", type=Path, help="Use this detections CSV instead of running YOLO")
    parser.add_argument("--keypoints", type=Path, help="Use this pitch keypoints CSV instead of the keypoint model")
    parser.add_argument("--no-video", action="store_true", help="Skip rendering the annotated video")
    args = parser.parse_args(argv)
    ctx = make_context(args)
    if not ctx.input_path.exists():
        raise SystemExit(f"Input video not found: {ctx.input_path}\nSee README 'Data and models' for sample clips.")

    start = time.perf_counter()
    result = run_pipeline(ctx, PipelineOptions(args.detections, args.keypoints, not args.no_video))
    s = result.summary
    print(f"\nDone in {time.perf_counter() - start:.0f}s. Outputs: {result.root}")
    print(f"Possession: team 0 {s['possession_pct'][0]}% / team 1 {s['possession_pct'][1]}%")
    print("Passes:", json.dumps(s["passes"]))
    if "teams" in s:
        print("Teams:", json.dumps(s["teams"]))
    for w in result.warnings:
        print(f"WARNING: {w}")


if __name__ == "__main__":
    main()
