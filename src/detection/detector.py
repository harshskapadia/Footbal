"""YOLO object detection for players, goalkeepers, referees and the ball."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv
from ultralytics import YOLO

from src.detection.classes import BALL, build_id_map, remap_detections
from src.utils.config import resolve_path

log = logging.getLogger(__name__)


def load_yolo(path: str | Path | None, fallback: str | Path | None = None) -> tuple[YOLO, Path]:
    """Load YOLO weights, using ``fallback`` if ``path`` does not exist.

    Official Ultralytics weights (e.g. ``yolo11n.pt``) given as a fallback are
    downloaded automatically into that location.

    Returns:
        The model and the path it was loaded from.
    """
    if path and resolve_path(path).exists():
        weights = resolve_path(path)
    elif fallback:
        weights = resolve_path(fallback)
        weights.parent.mkdir(parents=True, exist_ok=True)
        log.warning("Weights %s not found, falling back to %s", path, weights.name)
    else:
        raise FileNotFoundError(f"Model weights not found: {path}. See README 'Data and models'.")
    return YOLO(str(weights)), weights


def predict(
    model: YOLO, image: np.ndarray, conf: float, iou: float, imgsz: int, device: str
) -> sv.Detections:
    """Run a YOLO model on one BGR image and return raw ``sv.Detections``."""
    result = model.predict(image, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)[0]
    return sv.Detections.from_ultralytics(result)


class ObjectDetector:
    """Detect people and the ball in a frame, returning canonical class ids."""

    def __init__(
        self,
        model: YOLO,
        class_map: Mapping[str, str],
        device: str = "cpu",
        conf: float = 0.3,
        iou: float = 0.5,
        imgsz: int = 1280,
    ) -> None:
        self.model = model
        self.device = device
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.id_map = build_id_map(model.names, class_map)
        if not self.id_map:
            raise ValueError(f"No model classes match class_map. Model classes: {model.names}")

    @property
    def model_class_ids(self) -> list[int]:
        """Model class ids that map to a canonical class."""
        return sorted(self.id_map)

    def __call__(self, frame: np.ndarray) -> sv.Detections:
        """Detect objects in a BGR frame."""
        detections = predict(self.model, frame, self.conf, self.iou, self.imgsz, self.device)
        return remap_detections(detections, self.id_map)


def keep_best_ball(detections: sv.Detections) -> sv.Detections:
    """Keep only the most confident ball detection; other classes are untouched."""
    is_ball = detections.class_id == BALL
    if is_ball.sum() <= 1:
        return detections
    ball_idx = np.flatnonzero(is_ball)
    best = ball_idx[np.argmax(detections.confidence[ball_idx])]
    keep = ~is_ball
    keep[best] = True
    return detections[keep]


def build_detector(cfg: Mapping[str, Any], device: str) -> ObjectDetector:
    """Create the main detector from the ``models`` and ``detection`` config sections."""
    model, _ = load_yolo(cfg["models"].get("player_detector"), cfg["models"].get("fallback_detector"))
    det = cfg["detection"]
    return ObjectDetector(model, det["class_map"], device, det["conf"], det["iou"], det["imgsz"])
