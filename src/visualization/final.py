"""Final annotated frame: team-coloured players, speeds, ball, possession bar and minimap."""

from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np
import pandas as pd
import supervision as sv

from src.visualization.minimap import NEUTRAL, Minimap, overlay

BALL_DETECTED = (0, 215, 255)      # BGR gold
BALL_INTERPOLATED = (0, 140, 255)  # BGR orange
TEXT = (255, 255, 255)
PANEL = (30, 30, 30)


def _bgr(colour: tuple[int, int, int]) -> sv.Color:
    b, g, r = colour
    return sv.Color(r=r, g=g, b=b)


class FinalAnnotator:
    """Draw everything the viewer sees on one frame."""

    def __init__(self, team_bgr: Mapping[int, tuple[int, int, int]], minimap: Minimap | None) -> None:
        self.team_bgr = dict(team_bgr)
        self.minimap = minimap

    def _colour(self, team: float) -> tuple[int, int, int]:
        return self.team_bgr.get(int(team), NEUTRAL) if pd.notna(team) else NEUTRAL

    def draw_people(self, frame: np.ndarray, people: pd.DataFrame, owner_id: int) -> None:
        """Ellipse at the feet in the team colour, with ID (and speed) below."""
        for r in people.itertuples():
            colour = self._colour(getattr(r, "team_id", np.nan))
            cx, foot = int((r.x1 + r.x2) / 2), int(r.y2)
            w = max(int(r.x2 - r.x1), 8)
            cv2.ellipse(frame, (cx, foot), (w // 2 + 2, max(w // 6, 3)), 0, -45, 235, colour, 2, cv2.LINE_AA)
            label = f"#{r.tracker_id}"
            speed = getattr(r, "speed_kmh", np.nan)
            if pd.notna(speed):
                label += f" {speed:.0f} km/h"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            x0, y0 = cx - tw // 2 - 3, foot + 6
            cv2.rectangle(frame, (x0, y0), (x0 + tw + 6, y0 + th + 6), colour, -1)
            text_colour = (0, 0, 0) if sum(colour) > 380 else TEXT
            cv2.putText(frame, label, (x0 + 3, y0 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_colour, 1,
                        cv2.LINE_AA)
            if r.tracker_id == owner_id:
                tip = (cx, int(r.y1) - 4)
                pts = np.array([tip, (tip[0] - 7, tip[1] - 12), (tip[0] + 7, tip[1] - 12)])
                cv2.fillPoly(frame, [pts], BALL_DETECTED)

    @staticmethod
    def draw_ball(frame: np.ndarray, ball: pd.Series | None) -> None:
        """Marker above the ball; orange when the position was interpolated."""
        if ball is None or not np.isfinite(ball["x1"]):
            return
        cx, top = int((ball["x1"] + ball["x2"]) / 2), int(ball["y1"])
        colour = BALL_INTERPOLATED if bool(ball["interpolated"]) else BALL_DETECTED
        pts = np.array([(cx, top - 3), (cx - 8, top - 16), (cx + 8, top - 16)])
        cv2.fillPoly(frame, [pts], colour)
        cv2.polylines(frame, [pts], True, (0, 0, 0), 1, cv2.LINE_AA)

    def draw_possession(self, frame: np.ndarray, share: Mapping[int, float]) -> None:
        """Possession bar (running total so far) in the top-left corner."""
        x, y, w, h = 20, 20, 260, 54
        cv2.rectangle(frame, (x, y), (x + w, y + h), PANEL, -1)
        cv2.putText(frame, "Possession", (x + 10, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT, 1, cv2.LINE_AA)
        total = share.get(0, 0) + share.get(1, 0)
        split = share.get(0, 0) / total if total else 0.5
        bx, by, bw, bh = x + 10, y + 28, w - 20, 16
        cv2.rectangle(frame, (bx, by), (bx + int(bw * split), by + bh), self._colour(0), -1)
        cv2.rectangle(frame, (bx + int(bw * split), by), (bx + bw, by + bh), self._colour(1), -1)
        widths = {0: int(bw * split), 1: bw - int(bw * split)}
        for team, tx in ((0, bx + 4), (1, bx + bw - 44)):
            if widths[team] < 48:          # too narrow to hold its label
                continue
            colour = self._colour(team)
            text_colour = (0, 0, 0) if sum(colour) > 380 else TEXT
            cv2.putText(frame, f"{share.get(team, 0):.0f}%", (tx, by + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        text_colour, 1, cv2.LINE_AA)

    @staticmethod
    def draw_not_analysed(frame: np.ndarray) -> None:
        """Banner for frames outside wide-camera shots (close-ups, replays, crowd)."""
        text = "Not analysed: close-up / replay / crowd shot"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        x = (frame.shape[1] - tw) // 2 - 12
        cv2.rectangle(frame, (x, 20), (x + tw + 24, 20 + th + 18), PANEL, -1)
        cv2.putText(frame, text, (x + 12, 20 + th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT, 1, cv2.LINE_AA)

    def __call__(
        self, frame: np.ndarray, people: pd.DataFrame, ball: pd.Series | None, owner_id: int,
        share: Mapping[int, float], ball_xy_m: np.ndarray | None,
    ) -> np.ndarray:
        """Return the fully annotated frame."""
        out = frame.copy()
        self.draw_people(out, people, owner_id)
        self.draw_ball(out, ball)
        self.draw_possession(out, share)
        if self.minimap is not None and "x_m" in people and people["x_m"].notna().any():
            out = overlay(out, self.minimap.render(people, ball_xy_m, owner_id))
        return out
