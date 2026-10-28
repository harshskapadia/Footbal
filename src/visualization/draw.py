"""Frame annotation for detections."""

from __future__ import annotations

import numpy as np
import supervision as sv

from src.detection.classes import BALL, CLASS_NAMES, class_ids, subset

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
        is_ball = class_ids(detections) == BALL
        people, ball = subset(detections, ~is_ball), subset(detections, is_ball)
        conf = people.confidence if people.confidence is not None else np.ones(len(people))
        labels = [f"{CLASS_NAMES[int(c)]} {p:.2f}" for c, p in zip(class_ids(people), conf, strict=True)]
        out = self.ellipse.annotate(out, people)
        out = self.label.annotate(out, people, labels=labels)
        return self.ball.annotate(out, ball)


ROLE_PREFIX = {1: "GK ", 2: "", 3: "REF "}  # goalkeeper, player, referee


class TrackAnnotator:
    """Draw tracked people with their IDs, and the ball (orange when interpolated)."""

    def __init__(self) -> None:
        self.ellipse = sv.EllipseAnnotator(thickness=2, color_lookup=sv.ColorLookup.TRACK)
        self.label = sv.LabelAnnotator(
            text_color=sv.Color.BLACK, text_scale=0.45, text_padding=3,
            text_position=sv.Position.BOTTOM_CENTER, color_lookup=sv.ColorLookup.TRACK,
        )
        self.ball = sv.TriangleAnnotator(color=sv.Color.from_hex("#FFD700"), base=18, height=15)
        self.ball_interp = sv.TriangleAnnotator(color=sv.Color.from_hex("#FF8C00"), base=18, height=15)

    def __call__(
        self, frame: np.ndarray, people: sv.Detections, ball: sv.Detections, ball_interpolated: bool = False
    ) -> np.ndarray:
        """Return an annotated copy of ``frame``."""
        out = frame.copy()
        if len(people) and people.tracker_id is not None:
            pairs = zip(class_ids(people), people.tracker_id, strict=True)
            labels = [f"{ROLE_PREFIX.get(int(c), '')}#{t}" for c, t in pairs]
            out = self.ellipse.annotate(out, people)
            out = self.label.annotate(out, people, labels=labels)
        if len(ball):
            out = (self.ball_interp if ball_interpolated else self.ball).annotate(out, ball)
        return out
