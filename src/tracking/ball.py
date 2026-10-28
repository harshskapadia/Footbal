"""Clean up the ball trajectory: drop false positives and fill short gaps."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.detection.classes import BALL
from src.detection.io import BOX_COLUMNS

BALL_COLUMNS = ["frame", *BOX_COLUMNS, "confidence", "interpolated"]


def ball_table(detection_rows: pd.DataFrame, n_frames: int) -> pd.DataFrame:
    """Return one row per frame with the ball box, NaN where it was not detected.

    Args:
        detection_rows: Rows with ``frame``, ``class_id``, box and ``confidence`` columns.
        n_frames: Number of frames in the clip.
    """
    balls = detection_rows[detection_rows["class_id"] == BALL].astype({"frame": int})
    balls = balls.sort_values("confidence").drop_duplicates("frame", keep="last")
    table = pd.DataFrame({"frame": np.arange(n_frames)})
    table = table.merge(balls[["frame", *BOX_COLUMNS, "confidence"]], on="frame", how="left")
    table[[*BOX_COLUMNS, "confidence"]] = table[[*BOX_COLUMNS, "confidence"]].astype(float)
    table["interpolated"] = False
    return table[BALL_COLUMNS]


def _centres(table: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        (table["x1"] + table["x2"]) / 2,
        (table["y1"] + table["y2"]) / 2,
    ])


def remove_ball_outliers(table: pd.DataFrame, max_speed_px: float) -> pd.DataFrame:
    """Blank out ball detections that jump implausibly far from both neighbours.

    A detection is an outlier when, compared with the previous and the next
    detected frame, its implied speed exceeds ``max_speed_px`` pixels per frame
    in both directions. Comparing with both sides avoids letting one false
    positive reject every real detection after it. The first and last
    detections have one neighbour only, so they are checked afterwards against
    their nearest detection that survived the first pass.
    """
    out = table.copy()
    valid = np.flatnonzero(out["x1"].notna().to_numpy())
    if len(valid) < 3:
        return out
    centres = _centres(out)

    def speed(i: int, j: int) -> float:
        return float(np.linalg.norm(centres[i] - centres[j]) / abs(i - j))

    outliers = {
        i for prev, i, nxt in zip(valid[:-2], valid[1:-1], valid[2:])
        if speed(i, prev) > max_speed_px and speed(i, nxt) > max_speed_px
    }
    kept = [i for i in valid if i not in outliers]
    if len(kept) >= 3:
        if speed(kept[0], kept[1]) > max_speed_px:
            outliers.add(kept[0])
        if speed(kept[-1], kept[-2]) > max_speed_px:
            outliers.add(kept[-1])
    out.loc[sorted(outliers), [*BOX_COLUMNS, "confidence"]] = np.nan
    return out


def interpolate_ball(table: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    """Linearly fill missing ball boxes in gaps of at most ``max_gap`` frames.

    Gaps at the start or end of the clip, and gaps longer than ``max_gap``, are
    left empty: guessing there would invent ball positions.
    """
    out = table.copy()
    missing = out["x1"].isna().to_numpy()
    filled = out[BOX_COLUMNS].interpolate(method="linear", limit_area="inside")
    # Size of the NaN run each row belongs to
    run_id = np.cumsum(~missing)
    run_len = pd.Series(missing).groupby(run_id).transform("sum").to_numpy()
    fill = missing & (run_len <= max_gap) & filled["x1"].notna().to_numpy()
    out.loc[fill, BOX_COLUMNS] = filled.loc[fill, BOX_COLUMNS]
    out.loc[fill, "interpolated"] = True
    return out


def clean_ball_track(
    detection_rows: pd.DataFrame, n_frames: int, max_gap: int, max_speed_px: float
) -> pd.DataFrame:
    """Build the per-frame ball table, remove outliers, then interpolate gaps."""
    table = ball_table(detection_rows, n_frames)
    return interpolate_ball(remove_ball_outliers(table, max_speed_px), max_gap)
