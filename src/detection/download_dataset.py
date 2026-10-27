"""Download a Roboflow Universe football dataset in YOLO format.

Requires ``pip install -r requirements-train.txt`` and a free Roboflow API key
(https://app.roboflow.com/settings/api) exported as ``ROBOFLOW_API_KEY``.

Example:
    python -m src.detection.download_dataset --dataset players
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.utils.config import load_config, resolve_path

# Workspace / project / version used by Roboflow's open-source "sports" project.
DATASETS: dict[str, dict[str, str | int]] = {
    "players": {"workspace": "roboflow-jvuqo", "project": "football-players-detection-3zvbc",
                "version": 10, "format": "yolov8"},
    "ball": {"workspace": "roboflow-jvuqo", "project": "football-ball-detection-rejhg",
             "version": 2, "format": "yolov8"},
    "pitch": {"workspace": "roboflow-jvuqo", "project": "football-field-detection-f07vi",
              "version": 12, "format": "yolov8"},
}


def download(name: str, out_dir: Path, version: int | None = None) -> Path:
    """Download one dataset and return the folder that contains ``data.yaml``."""
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY first (https://app.roboflow.com/settings/api).")
    try:
        from roboflow import Roboflow
    except ImportError as e:
        raise SystemExit("Install training extras: pip install -r requirements-train.txt") from e

    spec = DATASETS[name]
    ver = version or int(spec["version"])
    target = out_dir / f"{str(spec['project']).rsplit('-', 1)[0]}-{ver}"
    project = Roboflow(api_key=api_key).workspace(spec["workspace"]).project(spec["project"])
    project.version(ver).download(str(spec["format"]), location=str(target))
    return target


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", choices=sorted(DATASETS), default="players")
    p.add_argument("--version", type=int, default=None, help="Override dataset version")
    p.add_argument("--out-dir", type=Path, default=None, help="Default: paths.datasets_dir")
    args = p.parse_args(argv)
    out_dir = args.out_dir or resolve_path(load_config()["paths"]["datasets_dir"])
    path = download(args.dataset, out_dir, args.version)
    print(f"Dataset ready: {path / 'data.yaml'}")


if __name__ == "__main__":
    main()
