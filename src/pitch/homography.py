"""Image-to-pitch homographies: per-frame fitting and temporal smoothing."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd


def apply_homography(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Map ``(N, 2)`` points through a 3x3 homography."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    if len(pts) == 0:
        return np.zeros((0, 2))
    return cv2.perspectiveTransform(pts, np.asarray(h, dtype=np.float64)).reshape(-1, 2)


def fit_homography(
    image_points: np.ndarray, pitch_points: np.ndarray, ransac_threshold: float = 1.0
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit an image-to-pitch homography with RANSAC.

    Args:
        image_points: ``(N, 2)`` pixel coordinates of detected keypoints.
        pitch_points: ``(N, 2)`` matching pitch coordinates in metres.
        ransac_threshold: Max distance in metres for a point to count as an inlier.

    Returns:
        ``(H, inlier_mask)``, or ``None`` with fewer than 4 points or a degenerate fit.
    """
    if len(image_points) < 4:
        return None
    h, mask = cv2.findHomography(
        np.asarray(image_points, np.float64), np.asarray(pitch_points, np.float64),
        cv2.RANSAC, ransac_threshold,
    )
    if h is None or not np.isfinite(h).all():
        return None
    return h, mask.ravel().astype(bool)


def reprojection_error(h: np.ndarray, image_points: np.ndarray, pitch_points: np.ndarray) -> float:
    """Mean distance in metres between mapped image points and their true pitch positions."""
    return float(np.linalg.norm(apply_homography(h, image_points) - pitch_points, axis=1).mean())


def smoothing_grid(image_size: tuple[int, int], y_range: tuple[float, float] = (0.4, 0.95)) -> np.ndarray:
    """A 4x3 grid of image points in the lower part of the frame (where the pitch is)."""
    w, h = image_size
    xs = np.linspace(0.05, 0.95, 4) * w
    ys = np.linspace(y_range[0], y_range[1], 3) * h
    return np.array([(x, y) for y in ys for x in xs], dtype=np.float64)


def smooth_homographies(
    homographies: list[np.ndarray | None],
    image_size: tuple[int, int],
    window: int = 7,
    max_gap: int = 25,
    y_range: tuple[float, float] = (0.4, 0.95),
) -> list[np.ndarray | None]:
    """Smooth per-frame homographies over time.

    Averaging matrix entries has no geometric meaning, so instead each
    homography is represented by where a fixed grid of image points lands on
    the pitch. Those pitch-space trajectories are:

    1. linearly interpolated across gaps of up to ``max_gap`` frames
       (frames with too few keypoints),
    2. filtered with a centred rolling median (removes one-frame glitches),
    3. averaged with a centred rolling mean (removes jitter),

    and a homography is refit from the grid for every frame. Centred windows
    are possible because the whole clip is processed offline, so the mapping
    does not lag behind the camera during pans.

    Returns:
        One homography per frame, or ``None`` where no mapping is available.
    """
    grid = smoothing_grid(image_size, y_range)
    n, g = len(homographies), len(grid)
    traj = np.full((n, g * 2), np.nan)
    for f, h in enumerate(homographies):
        if h is not None:
            traj[f] = apply_homography(h, grid).ravel()

    df = pd.DataFrame(traj)
    known = df[0].notna().to_numpy()
    df = df.interpolate(limit_area="inside")
    run_id = np.cumsum(known)
    gap_len = pd.Series(~known).groupby(run_id).transform("sum").to_numpy()
    df[~known & (gap_len > max_gap)] = np.nan
    df = df.rolling(window, center=True, min_periods=1).median()
    df = df.rolling(window, center=True, min_periods=1).mean()

    out: list[np.ndarray | None] = []
    for row in df.to_numpy():
        if np.isnan(row).any():
            out.append(None)
            continue
        h, _ = cv2.findHomography(grid, row.reshape(g, 2), 0)
        out.append(h)
    return out
