"""Shirt crops and colour features."""

from __future__ import annotations

import cv2
import numpy as np


def shirt_crop(
    frame: np.ndarray, box: np.ndarray, top: float = 0.15, bottom: float = 0.5, side: float = 0.2,
    min_px: int = 4,
) -> np.ndarray | None:
    """Cut the torso out of a person box.

    Args:
        frame: BGR frame.
        box: ``[x1, y1, x2, y2]`` person box.
        top, bottom: Vertical band of the box to keep, as fractions of its height.
        side: Fraction of the width trimmed from each side (removes arms and grass).
        min_px: Return ``None`` if the crop would be smaller than this in either direction.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx1, cx2 = int(round(x1 + side * w)), int(round(x2 - side * w))
    cy1, cy2 = int(round(y1 + top * h)), int(round(y1 + bottom * h))
    cx1, cy1 = max(cx1, 0), max(cy1, 0)
    cx2, cy2 = min(cx2, frame.shape[1]), min(cy2, frame.shape[0])
    if cx2 - cx1 < min_px or cy2 - cy1 < min_px:
        return None
    return frame[cy1:cy2, cx1:cx2]


def grass_mask(crop: np.ndarray) -> np.ndarray:
    """Boolean mask of green, saturated (grass) pixels in a BGR crop."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= 35) & (h <= 85) & (s >= 50) & (v >= 40)


def shirt_colour(crop: np.ndarray, min_fraction: float = 0.1) -> np.ndarray | None:
    """Median CIELAB colour of the non-grass pixels of a shirt crop.

    Lab is used because Euclidean distance in Lab roughly matches perceived
    colour difference, which makes K-means clusters correspond to kits. The
    median ignores a few stray pixels (numbers, logos, background).

    Returns:
        ``(3,)`` float array ``[L, a, b]`` (OpenCV 8-bit scaling), or ``None`` if
        fewer than ``min_fraction`` of the pixels are not grass.
    """
    keep = ~grass_mask(crop)
    if keep.mean() < min_fraction:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[keep].astype(np.float64)
    return np.median(lab, axis=0)


def lab_to_bgr(lab: np.ndarray) -> tuple[int, int, int]:
    """Convert one 8-bit-scaled Lab colour to a BGR tuple (for drawing)."""
    pixel = np.clip(np.asarray(lab, dtype=np.float64), 0, 255).astype(np.uint8).reshape(1, 1, 3)
    b, g, r = cv2.cvtColor(pixel, cv2.COLOR_LAB2BGR)[0, 0]
    return int(b), int(g), int(r)
