"""Turn keypoints into smoothed homographies and project tracks and ball onto the pitch."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.pitch.homography import apply_homography, fit_homography, reprojection_error, smooth_homographies
from src.pitch.template import PitchTemplate

H_COLUMNS = [f"h{i}{j}" for i in range(3) for j in range(3)]


def frame_homography(
    xy: np.ndarray, conf: np.ndarray, template: PitchTemplate, cfg: Mapping[str, Any]
) -> tuple[np.ndarray | None, int, float]:
    """Fit one frame's image-to-pitch homography from its keypoints.

    Returns:
        ``(H or None, inlier count, reprojection error in metres)``.
    """
    p = cfg["pitch"]
    keep = conf >= p["keypoint_conf"]
    if keep.sum() < p["min_keypoints"]:
        return None, int(keep.sum()), float("nan")
    image_pts, pitch_pts = xy[keep], template.vertices[keep]
    fit = fit_homography(image_pts, pitch_pts, p["ransac_threshold_m"])
    if fit is None:
        return None, 0, float("nan")
    h, inliers = fit
    if inliers.sum() < p["min_keypoints"]:
        return None, int(inliers.sum()), float("nan")
    err = reprojection_error(h, image_pts[inliers], pitch_pts[inliers])
    if err > p["max_reprojection_error_m"]:
        return None, int(inliers.sum()), err
    return h, int(inliers.sum()), err


def compute_homographies(
    keypoints: Mapping[int, tuple[np.ndarray, np.ndarray]],
    n_frames: int,
    image_size: tuple[int, int],
    cfg: Mapping[str, Any],
) -> tuple[list[np.ndarray | None], pd.DataFrame]:
    """Fit per-frame homographies, then smooth them over time.

    Returns:
        Smoothed homographies (``None`` where unavailable) and a per-frame
        table with the raw fit quality and the smoothed matrix entries.
    """
    template = PitchTemplate.from_config(cfg)
    p = cfg["pitch"]
    raw: list[np.ndarray | None] = []
    rows = []
    for f in range(n_frames):
        xy, conf = keypoints.get(f, (np.zeros((32, 2)), np.zeros(32)))
        h, n_inliers, err = frame_homography(xy, conf, template, cfg)
        raw.append(h)
        rows.append({"frame": f, "raw_fit": h is not None, "inliers": n_inliers,
                     "reprojection_error_m": round(err, 3) if np.isfinite(err) else np.nan})
    smoothed = smooth_homographies(raw, image_size, p["smoothing_window"], p["max_gap_frames"],
                                   tuple(p["smoothing_grid_y"]))
    table = pd.DataFrame(rows)
    mats = np.array([h.ravel() / h[2, 2] if h is not None else np.full(9, np.nan) for h in smoothed])
    table[H_COLUMNS] = mats
    table["mapped"] = [h is not None for h in smoothed]
    return smoothed, table


def homographies_from_table(table: pd.DataFrame) -> list[np.ndarray | None]:
    """Rebuild the smoothed homography list from ``homographies.csv``."""
    return [None if not row.mapped else np.array([getattr(row, c) for c in H_COLUMNS]).reshape(3, 3)
            for row in table.itertuples()]


def _project(points_px: np.ndarray, h: np.ndarray | None) -> np.ndarray:
    return apply_homography(h, points_px) if h is not None else np.full((len(points_px), 2), np.nan)


def project_tracks(
    tracks: pd.DataFrame, teams: pd.DataFrame | None, homographies: Sequence[np.ndarray | None],
    template: PitchTemplate, margin: float,
) -> pd.DataFrame:
    """Project each person's feet (bottom-centre of their box) onto the pitch.

    Positions further than ``margin`` metres outside the pitch are set to NaN
    (usually a bad homography or a person in the stands).
    """
    df = tracks[["frame", "tracker_id", "class_name", "x1", "y1", "x2", "y2"]].copy()
    df["x_px"] = (df["x1"] + df["x2"]) / 2
    df["y_px"] = df["y2"]
    df[["x_m", "y_m"]] = np.nan
    for f, idx in df.groupby("frame").groups.items():
        h = homographies[f] if f < len(homographies) else None
        df.loc[idx, ["x_m", "y_m"]] = _project(df.loc[idx, ["x_px", "y_px"]].to_numpy(), h)
    outside = ~template.contains(df[["x_m", "y_m"]].to_numpy(), margin)
    df.loc[outside, ["x_m", "y_m"]] = np.nan
    if teams is not None:
        df = df.merge(teams[["tracker_id", "team_id", "role"]], on="tracker_id", how="left")
    cols = ["frame", "tracker_id", *(["team_id", "role"] if teams is not None else ["class_name"]),
            "x_px", "y_px", "x_m", "y_m"]
    return df[cols].round({"x_px": 1, "y_px": 1, "x_m": 2, "y_m": 2})


def project_ball(
    ball: pd.DataFrame, homographies: Sequence[np.ndarray | None], template: PitchTemplate, margin: float
) -> pd.DataFrame:
    """Project the ball onto the pitch for every frame.

    Uses the bottom-centre of the ball box (where a rolling ball touches the
    grass). A ball in the air is projected to the wrong place: a homography only
    knows the ground plane.
    """
    df = ball[["frame", "interpolated"]].copy()
    df["x_px"] = (ball["x1"] + ball["x2"]) / 2
    df["y_px"] = ball["y2"]
    xy = np.full((len(df), 2), np.nan)
    for i, (f, x, y) in enumerate(zip(df["frame"], df["x_px"], df["y_px"], strict=True)):
        h = homographies[f] if f < len(homographies) else None
        if h is not None and np.isfinite(x):
            xy[i] = apply_homography(h, np.array([[x, y]]))[0]
    xy[~template.contains(xy, margin)] = np.nan
    df["x_m"], df["y_m"] = xy[:, 0], xy[:, 1]
    return df.round({"x_px": 1, "y_px": 1, "x_m": 2, "y_m": 2})
