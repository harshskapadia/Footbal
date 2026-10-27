"""Fine-tune a YOLO detector on a football dataset.

Example:
    python -m src.detection.train --config configs/train_detector.yaml
    python -m src.detection.train --config configs/train_detector.yaml --epochs 1 --imgsz 640  # quick check
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

from src.utils.config import resolve_path
from src.utils.device import select_device

log = logging.getLogger(__name__)


def load_train_config(path: Path, overrides: dict[str, Any]) -> dict[str, Any]:
    """Read the training YAML and apply non-null CLI overrides to its ``train`` section."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("train", {}).update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def train(cfg: dict[str, Any], device: str) -> Path:
    """Train a model and copy the best weights to ``cfg['export_to']``.

    Returns:
        Path of the exported weights.
    """
    data = resolve_path(cfg["data"])
    if not data.exists():
        raise SystemExit(
            f"Dataset not found: {data}\n"
            "Download it first: python -m src.detection.download_dataset --dataset players"
        )
    model = YOLO(cfg["base_model"])
    train_args = dict(cfg["train"])
    train_args["project"] = str(resolve_path(train_args.get("project", "runs/detect")))
    model.train(data=str(data), device=device, **train_args)

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    export_to = resolve_path(cfg["export_to"])
    export_to.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, export_to)
    log.info("Best weights copied to %s", export_to)
    return export_to


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Fine-tune YOLO on a football dataset.")
    p.add_argument("--config", type=Path, default=Path("configs/train_detector.yaml"))
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int)
    p.add_argument("--imgsz", type=int)
    p.add_argument("--batch", type=int)
    args = p.parse_args(argv)
    cfg = load_train_config(
        resolve_path(args.config), {"epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch}
    )
    out = train(cfg, select_device(args.device))
    print(f"Trained weights: {out}")


if __name__ == "__main__":
    main()
