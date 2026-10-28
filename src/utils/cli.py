"""Shared command-line options for the stage CLIs and ``src.run``."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.config import load_config, resolve_path
from src.utils.paths import RunPaths


@dataclass
class RunContext:
    """Everything a stage needs: config, input clip, output paths and frame limit."""

    cfg: dict[str, Any]
    input_path: Path
    paths: RunPaths
    max_frames: int | None


def base_parser(description: str) -> argparse.ArgumentParser:
    """Return a parser with the options every stage accepts."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--input", type=Path, help="Input video (default: paths.input_video in config)")
    p.add_argument("--output-dir", type=Path, help="Output root (default: paths.output_dir)")
    p.add_argument("--config", type=Path, default=None, help="Config YAML (default: config.yaml)")
    p.add_argument("--max-frames", type=int, default=None, help="Only process the first N frames")
    p.add_argument("--device", default=None, help="Override device: auto, cpu, cuda:0, mps")
    return p


def make_context(args: argparse.Namespace) -> RunContext:
    """Load config, apply CLI overrides and create the run directory."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    input_path = resolve_path(args.input or cfg["paths"]["input_video"])
    output_dir = resolve_path(args.output_dir or cfg["paths"]["output_dir"])
    max_frames = args.max_frames or cfg.get("output", {}).get("max_frames")
    return RunContext(cfg, input_path, RunPaths.for_clip(output_dir, input_path), max_frames)
