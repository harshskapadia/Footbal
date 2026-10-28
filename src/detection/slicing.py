"""Slice-based (tiled) inference for the small, fast-moving ball."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import supervision as sv
from ultralytics import YOLO

from src.detection.classes import BALL, build_id_map, remap_detections
from src.detection.detector import keep_best_ball, load_yolo, predict
from src.utils.config import resolve_path


class SlicedBallDetector:
    """Find the ball by running a detector on overlapping tiles of the frame.

    A 1920x1080 frame resized to the model input loses most of a 10 px ball;
    tiles keep it at native resolution. Results from all tiles are merged with
    NMS and only the most confident ball is returned.
    """

    def __init__(
        self,
        model: YOLO,
        class_map: Mapping[str, str],
        device: str = "cpu",
        conf: float = 0.15,
        iou: float = 0.1,
        imgsz: int = 640,
        slice_wh: tuple[int, int] = (640, 640),
        overlap_wh: tuple[int, int] = (128, 128),
    ) -> None:
        self.model = model
        self.device = device
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.id_map = {k: v for k, v in build_id_map(model.names, class_map).items() if v == BALL}
        if not self.id_map:
            raise ValueError(f"Model has no ball class. Model classes: {model.names}")
        self.slicer = sv.InferenceSlicer(
            callback=self._predict_tile,
            slice_wh=(int(slice_wh[0]), int(slice_wh[1])),
            overlap_wh=(int(overlap_wh[0]), int(overlap_wh[1])),
            overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
            iou_threshold=iou,
            thread_workers=1,
        )

    def _predict_tile(self, tile: np.ndarray) -> sv.Detections:
        detections = predict(self.model, tile, self.conf, self.iou, self.imgsz, self.device)
        return remap_detections(detections, self.id_map)

    def __call__(self, frame: np.ndarray) -> sv.Detections:
        """Return at most one ball detection for a BGR frame."""
        return keep_best_ball(self.slicer(frame))


def build_ball_detector(cfg: Mapping[str, Any], device: str) -> SlicedBallDetector:
    """Create the sliced ball detector from config.

    Prefers the dedicated ball model, then the player model, then the COCO fallback.
    """
    models = cfg["models"]
    player = models.get("player_detector")
    has_player_model = bool(player) and resolve_path(player).exists()
    model, _ = load_yolo(
        models.get("ball_detector"),
        player if has_player_model else models.get("fallback_detector"),
    )
    b = cfg["ball"]
    return SlicedBallDetector(
        model, cfg["detection"]["class_map"], device,
        conf=b["conf"], iou=b["iou"], imgsz=b["imgsz"],
        slice_wh=(b["slice_wh"][0], b["slice_wh"][1]), overlap_wh=(b["overlap_wh"][0], b["overlap_wh"][1]),
    )

