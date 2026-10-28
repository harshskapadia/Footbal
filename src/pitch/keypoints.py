"""Pitch keypoint detection (YOLO pose model) and keypoint CSV I/O."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from ultralytics import YOLO

from src.detection.detector import load_yolo
from src.utils.config import resolve_path

KEYPOINT_COLUMNS = ["frame", "keypoint", "x", "y", "confidence"]
MISSING_MODEL_MESSAGE = (
    "Pitch keypoint model not found at {path}.\n"
    "Download it with `bash scripts/download_assets.sh` (football-pitch-detection.pt), or train one on\n"
    "Roboflow's football-field-detection dataset (README, 'Data and models'). You can also pass\n"
    "pre-computed keypoints with --keypoints keypoints.csv."
)


class PitchKeypointDetector:
    """Detect the 32 pitch keypoints in a frame."""

    def __init__(self, model: YOLO, device: str = "cpu", imgsz: int = 640) -> None:
        self.model = model
        self.device = device
        self.imgsz = imgsz

    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(xy, confidence)`` with shapes ``(32, 2)`` and ``(32,)``.

        If the model finds several pitch instances, the most confident one is used.
        """
        result = self.model.predict(frame, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            return np.zeros((32, 2)), np.zeros(32)
        xy = result.keypoints.xy.cpu().numpy()
        conf = result.keypoints.conf
        conf = conf.cpu().numpy() if conf is not None else np.ones(xy.shape[:2])
        best = int(conf.mean(axis=1).argmax())
        return xy[best], conf[best]


def build_keypoint_detector(cfg: Mapping[str, Any], device: str) -> PitchKeypointDetector:
    """Load the pitch keypoint model, or raise with download instructions."""
    path = cfg["models"].get("pitch_detector")
    if not path or not resolve_path(path).exists():
        raise FileNotFoundError(MISSING_MODEL_MESSAGE.format(path=path))
    model, _ = load_yolo(path)
    return PitchKeypointDetector(model, device)


def keypoints_to_rows(frame_idx: int, xy: np.ndarray, conf: np.ndarray) -> list[dict]:
    """Flatten one frame's keypoints into CSV rows."""
    return [{"frame": frame_idx, "keypoint": k, "x": round(float(x), 2), "y": round(float(y), 2),
             "confidence": round(float(c), 3)} for k, ((x, y), c) in enumerate(zip(xy, conf))]


def keypoints_by_frame(df: pd.DataFrame, n_keypoints: int = 32) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Group keypoint rows into ``{frame: (xy (K, 2), confidence (K,))}``."""
    out = {}
    for f, g in df.groupby("frame"):
        xy = np.zeros((n_keypoints, 2))
        conf = np.zeros(n_keypoints)
        idx = g["keypoint"].to_numpy(int)
        xy[idx] = g[["x", "y"]].to_numpy(float)
        conf[idx] = g["confidence"].to_numpy(float)
        out[int(f)] = (xy, conf)
    return out
