"""Convert detections to and from flat CSV rows, so stages can run independently."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import supervision as sv

from src.detection.classes import CLASS_NAMES, class_ids

CSV_COLUMNS = ["frame", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]
BOX_COLUMNS = ["x1", "y1", "x2", "y2"]


def detections_to_rows(frame_idx: int, detections: sv.Detections) -> list[dict]:
    """Flatten detections into CSV rows (adds ``tracker_id`` when present)."""
    rows = []
    confidence = detections.confidence if detections.confidence is not None else np.ones(len(detections))
    for i, (b, c, p) in enumerate(zip(detections.xyxy, class_ids(detections), confidence, strict=True)):
        row = {
            "frame": frame_idx, "class_id": int(c), "class_name": CLASS_NAMES[int(c)],
            "confidence": round(float(p), 4),
            "x1": round(float(b[0]), 1), "y1": round(float(b[1]), 1),
            "x2": round(float(b[2]), 1), "y2": round(float(b[3]), 1),
        }
        if detections.tracker_id is not None:
            row["tracker_id"] = int(detections.tracker_id[i])
        rows.append(row)
    return rows


def rows_to_detections(df: pd.DataFrame) -> sv.Detections:
    """Build ``sv.Detections`` from rows with box, class and (optionally) tracker columns."""
    if df.empty:
        return sv.Detections.empty()
    detections = sv.Detections(
        xyxy=df[BOX_COLUMNS].to_numpy(dtype=float),
        confidence=df["confidence"].to_numpy(dtype=float),
        class_id=df["class_id"].to_numpy(dtype=int),
        tracker_id=df["tracker_id"].to_numpy(dtype=int) if "tracker_id" in df else None,
    )
    detections.data["class_name"] = np.array([CLASS_NAMES[int(c)] for c in class_ids(detections)])
    return detections


def group_by_frame(df: pd.DataFrame) -> dict[int, sv.Detections]:
    """Split a rows DataFrame into one ``sv.Detections`` per frame index."""
    return {int(frame): rows_to_detections(g) for frame, g in df.groupby("frame")}


def load_detections_csv(path: str | Path) -> dict[int, sv.Detections]:
    """Load a stage 2 detections CSV as ``{frame: sv.Detections}``."""
    return group_by_frame(pd.read_csv(path))
