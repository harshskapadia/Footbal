"""Canonical object classes and mapping from arbitrary model class names."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import supervision as sv

BALL, GOALKEEPER, PLAYER, REFEREE = 0, 1, 2, 3
CLASS_NAMES: dict[int, str] = {
    BALL: "ball",
    GOALKEEPER: "goalkeeper",
    PLAYER: "player",
    REFEREE: "referee",
}
NAME_TO_ID: dict[str, int] = {v: k for k, v in CLASS_NAMES.items()}


def build_id_map(model_names: Mapping[int, str], class_map: Mapping[str, str]) -> dict[int, int]:
    """Map model class ids to canonical ids using class names.

    Args:
        model_names: ``model.names`` from Ultralytics, e.g. ``{0: "person", 32: "sports ball"}``.
        class_map: Model class name to canonical name, e.g. ``{"person": "player"}``.

    Returns:
        Model class id to canonical class id, for every class that has a mapping.
    """
    lookup = {k.lower(): v for k, v in class_map.items()}
    id_map: dict[int, int] = {}
    for model_id, name in model_names.items():
        canonical = lookup.get(name.lower())
        if canonical in NAME_TO_ID:
            id_map[int(model_id)] = NAME_TO_ID[canonical]
    return id_map


def remap_detections(detections: sv.Detections, id_map: Mapping[int, int]) -> sv.Detections:
    """Convert detections to canonical class ids and drop unmapped classes."""
    if len(detections) == 0:
        return detections
    keep = np.array([int(c) in id_map for c in detections.class_id], dtype=bool)
    detections = detections[keep]
    detections.class_id = np.array([id_map[int(c)] for c in detections.class_id], dtype=int)
    detections.data["class_name"] = np.array([CLASS_NAMES[c] for c in detections.class_id])
    return detections
