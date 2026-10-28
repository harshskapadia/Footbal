"""Top-down pitch rendering shared by the minimap, heatmaps and passing networks."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.pitch.template import PitchTemplate

GRASS = (60, 140, 60)        # BGR
GRASS_DARK = (52, 125, 52)
LINE = (255, 255, 255)


@dataclass(frozen=True)
class PitchCanvas:
    """Maps pitch metres to pixels of a top-down image."""

    template: PitchTemplate = PitchTemplate()
    scale: float = 8.0       # pixels per metre
    padding: float = 4.0     # metres of border around the pitch

    @property
    def size(self) -> tuple[int, int]:
        """Image ``(width, height)`` in pixels."""
        return (round((self.template.length + 2 * self.padding) * self.scale),
                round((self.template.width + 2 * self.padding) * self.scale))

    def to_pixels(self, xy_m: np.ndarray) -> np.ndarray:
        """Convert ``(N, 2)`` metres to ``(N, 2)`` float pixel coordinates."""
        return (np.asarray(xy_m, dtype=float) + self.padding) * self.scale

    def metres_to_pixels_matrix(self) -> np.ndarray:
        """3x3 affine matrix taking homogeneous metres to canvas pixels."""
        s, p = self.scale, self.padding * self.scale
        return np.array([[s, 0, p], [0, s, p], [0, 0, 1]], dtype=float)


def draw_pitch(canvas: PitchCanvas = PitchCanvas(), stripes: bool = True, line_px: int | None = None) -> np.ndarray:
    """Render a top-down pitch with markings as a BGR image."""
    w, h = canvas.size
    img = np.full((h, w, 3), GRASS, dtype=np.uint8)
    t = canvas.template
    if stripes:
        stripe = t.length / 14
        for k in range(0, 14, 2):
            x0, _ = canvas.to_pixels([[k * stripe, 0]])[0]
            x1, _ = canvas.to_pixels([[(k + 1) * stripe, 0]])[0]
            img[:, int(x0):int(x1)] = GRASS_DARK
    thickness = line_px or max(1, round(canvas.scale * 0.15))
    v = canvas.to_pixels(t.vertices).round().astype(int)
    for a, b in t.edges:
        cv2.line(img, tuple(v[a - 1]), tuple(v[b - 1]), LINE, thickness, cv2.LINE_AA)
    centre = tuple(canvas.to_pixels([[t.length / 2, t.width / 2]])[0].round().astype(int))
    radius = round(t.centre_circle_radius * canvas.scale)
    cv2.circle(img, centre, radius, LINE, thickness, cv2.LINE_AA)
    for spot_x in (t.penalty_spot_distance, t.length - t.penalty_spot_distance):
        spot = tuple(canvas.to_pixels([[spot_x, t.width / 2]])[0].round().astype(int))
        cv2.circle(img, spot, max(2, thickness), LINE, -1, cv2.LINE_AA)
    cv2.circle(img, centre, max(2, thickness), LINE, -1, cv2.LINE_AA)
    return img
