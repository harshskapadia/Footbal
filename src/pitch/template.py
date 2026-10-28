"""Real-world pitch geometry and the 32 keypoints used by the pitch keypoint model.

Coordinates are in metres with the origin at a corner flag: ``x`` runs along the
length (0..105) and ``y`` across the width (0..68). The keypoint order matches the
Roboflow ``football-field-detection`` dataset (see roboflow/sports
``sports/configs/soccer.py``), so a model trained on it can be used directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PitchTemplate:
    """Pitch dimensions in metres (IFAB standard markings, 105 x 68 m pitch)."""

    length: float = 105.0
    width: float = 68.0
    penalty_box_width: float = 40.32
    penalty_box_length: float = 16.5
    goal_box_width: float = 18.32
    goal_box_length: float = 5.5
    centre_circle_radius: float = 9.15
    penalty_spot_distance: float = 11.0

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> PitchTemplate:
        """Build from the ``pitch`` config section (length_m / width_m)."""
        p = cfg.get("pitch", {})
        return cls(length=float(p.get("length_m", 105.0)), width=float(p.get("width_m", 68.0)))

    @property
    def vertices(self) -> np.ndarray:
        """The 32 keypoints as a ``(32, 2)`` array of metres, in model order."""
        L, W = self.length, self.width
        pbw, pbl = self.penalty_box_width, self.penalty_box_length
        gbw, gbl = self.goal_box_width, self.goal_box_length
        r, spot = self.centre_circle_radius, self.penalty_spot_distance
        return np.array([
            (0, 0), (0, (W - pbw) / 2), (0, (W - gbw) / 2), (0, (W + gbw) / 2),          # 1-4
            (0, (W + pbw) / 2), (0, W),                                                  # 5-6
            (gbl, (W - gbw) / 2), (gbl, (W + gbw) / 2), (spot, W / 2),                   # 7-9
            (pbl, (W - pbw) / 2), (pbl, (W - gbw) / 2), (pbl, (W + gbw) / 2),            # 10-12
            (pbl, (W + pbw) / 2),                                                        # 13
            (L / 2, 0), (L / 2, W / 2 - r), (L / 2, W / 2 + r), (L / 2, W),              # 14-17
            (L - pbl, (W - pbw) / 2), (L - pbl, (W - gbw) / 2),                          # 18-19
            (L - pbl, (W + gbw) / 2), (L - pbl, (W + pbw) / 2),                          # 20-21
            (L - spot, W / 2), (L - gbl, (W - gbw) / 2), (L - gbl, (W + gbw) / 2),       # 22-24
            (L, 0), (L, (W - pbw) / 2), (L, (W - gbw) / 2), (L, (W + gbw) / 2),          # 25-28
            (L, (W + pbw) / 2), (L, W),                                                  # 29-30
            (L / 2 - r, W / 2), (L / 2 + r, W / 2),                                      # 31-32
        ], dtype=np.float64)

    @property
    def edges(self) -> list[tuple[int, int]]:
        """Pairs of 1-based vertex indices joined by a pitch line."""
        return [
            (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
            (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
            (16, 17), (18, 19), (19, 20), (20, 21), (23, 24),
            (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
            (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
            (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30),
        ]

    def contains(self, xy: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Boolean mask of points inside the pitch extended by ``margin`` metres."""
        xy = np.atleast_2d(xy)
        return (
            (xy[:, 0] >= -margin) & (xy[:, 0] <= self.length + margin)
            & (xy[:, 1] >= -margin) & (xy[:, 1] <= self.width + margin)
        )
