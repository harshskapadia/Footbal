"""2D top-down minimap of players and ball, overlaid on the broadcast video."""

from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np
import pandas as pd

from src.pitch.draw import PitchCanvas, draw_pitch
from src.pitch.template import PitchTemplate

NEUTRAL = (160, 160, 160)       # referees / unassigned (BGR)
OUTLINE = (20, 20, 20)
OWNER_RING = (0, 215, 255)      # gold ring around the player on the ball


class Minimap:
    """Draws one frame of the top-down view."""

    def __init__(self, template: PitchTemplate, team_bgr: Mapping[int, tuple[int, int, int]],
                 scale: float = 4.0, padding: float = 3.0) -> None:
        self.canvas = PitchCanvas(template, scale, padding)
        self.background = draw_pitch(self.canvas, line_px=1)
        self.team_bgr = dict(team_bgr)
        self.radius = max(4, round(0.9 * scale))

    def render(self, people: pd.DataFrame, ball_xy: np.ndarray | None = None, owner_id: int = -1) -> np.ndarray:
        """Draw ``people`` (needs ``x_m, y_m, team_id, tracker_id``) and the ball."""
        img = self.background.copy()
        people = people.dropna(subset=["x_m", "y_m"])
        pts = self.canvas.to_pixels(people[["x_m", "y_m"]].to_numpy(float)).round().astype(int)
        for (x, y), team, tid in zip(pts, people["team_id"], people["tracker_id"], strict=True):
            colour = self.team_bgr.get(int(team), NEUTRAL) if pd.notna(team) else NEUTRAL
            if tid == owner_id:
                cv2.circle(img, (x, y), self.radius + 3, OWNER_RING, 2, cv2.LINE_AA)
            cv2.circle(img, (x, y), self.radius, colour, -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), self.radius, OUTLINE, 1, cv2.LINE_AA)
        if ball_xy is not None and np.isfinite(ball_xy).all():
            bx, by = self.canvas.to_pixels(np.asarray(ball_xy)[None])[0].round().astype(int)
            cv2.circle(img, (bx, by), max(3, self.radius // 2 + 1), (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (bx, by), max(3, self.radius // 2 + 1), OUTLINE, 1, cv2.LINE_AA)
        return img


def overlay(frame: np.ndarray, image: np.ndarray, width_frac: float = 0.28, margin: int = 16,
            alpha: float = 0.9) -> np.ndarray:
    """Blend ``image`` into the bottom-centre of ``frame``, scaled to ``width_frac`` of its width."""
    h, w = frame.shape[:2]
    new_w = int(w * width_frac)
    new_h = int(image.shape[0] * new_w / image.shape[1])
    small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0, y0 = (w - new_w) // 2, h - new_h - margin
    roi = frame[y0:y0 + new_h, x0:x0 + new_w]
    frame[y0:y0 + new_h, x0:x0 + new_w] = cv2.addWeighted(small, alpha, roi, 1 - alpha, 0)
    cv2.rectangle(frame, (x0 - 1, y0 - 1), (x0 + new_w, y0 + new_h), (255, 255, 255), 1)
    return frame
