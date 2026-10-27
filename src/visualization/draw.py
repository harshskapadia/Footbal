"""Frame annotation for detections."""

from __future__ import annotations

import numpy as np
import supervision as sv

from src.detection.classes import BALL, CLASS_NAMES

# Colours indexed by canonical class id: ball, goalkeeper, player, referee
CLASS_PALETTE = sv.ColorPalette.from_hex(["#FFD700", "#00BFFF", "#FF1493", "#7CFC00"])


class DetectionAnnotator:
    """Draw ellipses under people and a marker above the ball."""

    def __init__(self) -> None:
        lookup = sv.ColorLookup.CLASS
        self.ellipse = sv.EllipseAnnotator(color=CLASS_PALETTE, thickness=2, color_lookup=lookup)
        self.label = sv.LabelAnnotator(
            color=CLASS_PALETTE, text_color=sv.Color.BLACK, text_scale=0.4,
            text_padding=3, text_position=sv.Position.BOTTOM_CENTER, color_lookup=lookup,
        )
        self.ball = sv.TriangleAnnotator(color=sv.Color.from_hex("#FFD700"), base=18, height=15)

    def __call__(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """Return an annotated copy of ``frame``."""
        out = frame.copy()
        is_ball = detections.class_id == BALL
        people, ball = detections[~is_ball], detections[is_ball]
        labels = [f"{CLASS_NAMES[c]} {p:.2f}" for c, p in zip(people.class_id, people.confidence)]
        out = self.ellipse.annotate(out, people)
        out = self.label.annotate(out, people, labels=labels)
        return self.ball.annotate(out, ball)
