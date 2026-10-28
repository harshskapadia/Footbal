"""Ball possession: who controls the ball in each frame, and each team's share."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

NO_OWNER = -1


def nearest_owner(
    people: pd.DataFrame, ball_xy: np.ndarray, radius: float, cols: tuple[str, str]
) -> tuple[int, float]:
    """Return ``(tracker_id, distance)`` of the closest person within ``radius`` of the ball."""
    if people.empty or not np.isfinite(ball_xy).all():
        return NO_OWNER, np.nan
    xy = people[list(cols)].to_numpy(float)
    d = np.linalg.norm(xy - ball_xy, axis=1)
    d[~np.isfinite(d)] = np.inf
    i = int(np.argmin(d))
    if d[i] > radius:
        return NO_OWNER, float(d[i]) if np.isfinite(d[i]) else np.nan
    return int(people["tracker_id"].iloc[i]), float(d[i])


def debounce(owner: np.ndarray, min_frames: int) -> np.ndarray:
    """Drop control spells shorter than ``min_frames`` (a ball flying past a player)."""
    out = owner.copy()
    start = 0
    for i in range(1, len(owner) + 1):
        if i == len(owner) or owner[i] != owner[start]:
            if owner[start] != NO_OWNER and i - start < min_frames:
                out[start:i] = NO_OWNER
            start = i
    return out


def compute_possession(
    positions: pd.DataFrame, ball_positions: pd.DataFrame, cfg: Mapping[str, Any], fps: float,
    use_pitch: bool = True,
) -> pd.DataFrame:
    """Frame-by-frame ball control.

    The closest outfield player or goalkeeper within the possession radius
    controls the ball. Short spells are ignored, and while nobody controls it
    (a pass in flight) the last team in control keeps possession.

    Args:
        positions: Stage 5 positions (needs ``team_id``; pitch or pixel coordinates).
        ball_positions: Stage 5 ball positions.
        cfg: Loaded config.
        fps: Frames per second (to convert ``min_control_s`` to frames).
        use_pitch: Measure distance in metres (``x_m, y_m``) or, if False, in pixels.

    Returns:
        ``frame, owner_id, owner_team, distance, team_in_possession``.
    """
    a = cfg["analytics"]
    cols = ("x_m", "y_m") if use_pitch else ("x_px", "y_px")
    radius = a["possession_radius_m"] if use_pitch else a["possession_radius_px"]
    players = positions[positions["team_id"].isin([0, 1])]
    team_of = players.groupby("tracker_id")["team_id"].first()
    by_frame = {f: g for f, g in players.groupby("frame")}

    owners, dists = [], []
    for row in ball_positions.itertuples():
        people = by_frame.get(row.frame, players.iloc[:0])
        owner, dist = nearest_owner(people, np.array([getattr(row, cols[0]), getattr(row, cols[1])]), radius, cols)
        owners.append(owner)
        dists.append(dist)

    stable = debounce(np.array(owners), max(1, round(a["min_control_s"] * fps)))
    df = pd.DataFrame({"frame": ball_positions["frame"].to_numpy(), "owner_id": stable,
                       "distance": np.round(dists, 2)})
    df["owner_team"] = df["owner_id"].map(team_of).fillna(NO_OWNER).astype(int)
    # Shot id per frame (-1 = not analysed); possession is never carried across a cut.
    df["shot_id"] = ball_positions["shot_id"].to_numpy() if "shot_id" in ball_positions else 0
    carried = df["owner_team"].replace(NO_OWNER, np.nan).groupby(df["shot_id"]).ffill()
    df["team_in_possession"] = carried.fillna(NO_OWNER).astype(int)
    df.loc[df["shot_id"] < 0, ["owner_id", "owner_team", "team_in_possession"]] = NO_OWNER
    return df


def possession_share(possession: pd.DataFrame) -> dict[int, float]:
    """Percentage of frames each team was in possession (frames before first control excluded)."""
    known = possession[possession["team_in_possession"] >= 0]["team_in_possession"]
    if known.empty:
        return {0: 0.0, 1: 0.0}
    share = known.value_counts(normalize=True) * 100
    return {t: round(float(share.get(t, 0.0)), 1) for t in (0, 1)}
