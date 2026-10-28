"""Player speed and distance covered from pitch positions."""

from __future__ import annotations

import numpy as np
import pandas as pd

STAT_COLUMNS = ["tracker_id", "team_id", "role", "frames", "minutes_tracked", "distance_m",
                "avg_speed_kmh", "max_speed_kmh"]


def track_speeds(track: pd.DataFrame, fps: float, window: int, max_speed_kmh: float) -> pd.DataFrame:
    """Speed per frame for one track.

    Positions are smoothed with a centred rolling mean (removes box jitter),
    then speed is the distance between consecutive smoothed positions times fps.
    Steps across gaps in the track, and readings above ``max_speed_kmh``, are
    discarded rather than counted.

    Returns:
        ``frame, speed_kmh, step_m`` (``step_m`` is NaN where discarded).
    """
    t = track.sort_values("frame").dropna(subset=["x_m", "y_m"])
    if len(t) < 2:
        return pd.DataFrame({"frame": t["frame"], "speed_kmh": np.nan, "step_m": np.nan})
    xy = t[["x_m", "y_m"]].rolling(window, center=True, min_periods=1).mean().to_numpy()
    frames = t["frame"].to_numpy()
    step = np.r_[np.nan, np.linalg.norm(np.diff(xy, axis=0), axis=1)]
    consecutive = np.r_[False, np.diff(frames) == 1]
    speed = step * fps * 3.6
    bad = ~consecutive | (speed > max_speed_kmh)
    step[bad] = np.nan
    speed[bad] = np.nan
    return pd.DataFrame({"frame": frames, "speed_kmh": np.round(speed, 2), "step_m": step})


def compute_kinematics(
    positions: pd.DataFrame, fps: float, window: int = 5, max_speed_kmh: float = 40.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Speeds per frame and distance/speed statistics per player.

    Returns:
        ``speeds`` (``frame, tracker_id, speed_kmh``) and per-player ``stats`` (``STAT_COLUMNS``).
    """
    speed_parts, stats = [], []
    for tid, track in positions.groupby("tracker_id"):
        s = track_speeds(track, fps, window, max_speed_kmh).assign(tracker_id=tid)
        speed_parts.append(s)
        valid = s["speed_kmh"].dropna()
        stats.append({
            "tracker_id": tid,
            "team_id": track["team_id"].iloc[0] if "team_id" in track.columns else -1,
            "role": track["role"].iloc[0] if "role" in track.columns else track["class_name"].iloc[0],
            "frames": len(track),
            "minutes_tracked": round(len(track) / fps / 60, 2),
            "distance_m": round(float(s["step_m"].sum()), 1),
            "avg_speed_kmh": round(float(valid.mean()), 1) if len(valid) else np.nan,
            # 95th percentile, not the single highest reading, to ignore one-frame spikes
            "max_speed_kmh": round(float(np.percentile(valid, 95)), 1) if len(valid) else np.nan,
        })
    speeds = pd.concat(speed_parts)[["frame", "tracker_id", "speed_kmh"]] if speed_parts else \
        pd.DataFrame(columns=["frame", "tracker_id", "speed_kmh"])
    stats_df = pd.DataFrame(stats, columns=STAT_COLUMNS).sort_values("distance_m", ascending=False)
    return speeds, stats_df.reset_index(drop=True)
