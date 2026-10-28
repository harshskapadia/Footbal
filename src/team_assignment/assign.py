"""Assign every track a team and a role (player, goalkeeper, referee, other)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.detection.io import BOX_COLUMNS
from src.team_assignment.classifier import TeamClassifier
from src.team_assignment.features import shirt_colour, shirt_crop
from src.utils.progress import ProgressCallback, progress
from src.utils.video import read_frames

NO_TEAM = -1
TEAM_COLUMNS = ["tracker_id", "role", "team_id", "n_crops", "L", "a", "b", "colour_distance"]


def sample_track_frames(tracks: pd.DataFrame, per_track: int) -> pd.DataFrame:
    """Pick up to ``per_track`` rows per track, evenly spaced along it."""
    parts = []
    for _, g in tracks.sort_values("frame").groupby("tracker_id"):
        idx = np.unique(np.linspace(0, len(g) - 1, min(per_track, len(g))).round().astype(int))
        parts.append(g.iloc[idx])
    return pd.concat(parts) if parts else tracks.iloc[:0]


def extract_colours(
    video: Path, samples: pd.DataFrame, cfg: Mapping[str, Any], callback: ProgressCallback | None = None
) -> pd.DataFrame:
    """Read the clip once and compute a shirt colour for every sampled box.

    Returns:
        Rows of ``frame, tracker_id, L, a, b`` (crops that are too small or all
        grass are skipped).
    """
    t = cfg["teams"]
    by_frame = {f: g for f, g in samples.groupby("frame")}
    last = int(samples["frame"].max()) + 1 if len(samples) else 0
    rows = []
    for f, frame in enumerate(progress(read_frames(video, last), last, "teams", callback)):
        if f not in by_frame:
            continue
        for r in by_frame[f].itertuples():
            box = np.array([r.x1, r.y1, r.x2, r.y2])
            crop = shirt_crop(frame, box, t["crop_top"], t["crop_bottom"], t["crop_side"], t["min_crop_px"])
            colour = shirt_colour(crop) if crop is not None else None
            if colour is not None:
                rows.append({"frame": f, "tracker_id": r.tracker_id, "L": colour[0], "a": colour[1], "b": colour[2]})
    return pd.DataFrame(rows, columns=["frame", "tracker_id", "L", "a", "b"])


def assign_goalkeepers(tracks: pd.DataFrame, team_of: Mapping[int, int], goalkeepers: list[int]) -> dict[int, int]:
    """Put each goalkeeper in the team whose players are, on average, nearest to them.

    Kits do not help (goalkeepers wear their own colours), but a goalkeeper
    stands near their own defence. Compared frame by frame in image ``x``, so it
    still works while the camera pans.
    """
    feet = tracks.assign(x=(tracks["x1"] + tracks["x2"]) / 2)
    feet["team"] = feet["tracker_id"].map(team_of)
    result = {}
    for gk in goalkeepers:
        gk_rows = feet[feet["tracker_id"] == gk][["frame", "x"]]
        others = feet[feet["team"].isin([0, 1]) & feet["frame"].isin(gk_rows["frame"])]
        if others.empty:
            result[gk] = NO_TEAM
            continue
        team_x = others.groupby(["frame", "team"])["x"].mean().unstack()
        gap = (team_x.sub(gk_rows.set_index("frame")["x"], axis=0)).abs().mean()
        result[gk] = int(gap.idxmin()) if gap.notna().any() else NO_TEAM
    return result


def assign_teams(
    tracks: pd.DataFrame, colours: pd.DataFrame, cfg: Mapping[str, Any]
) -> tuple[pd.DataFrame, TeamClassifier]:
    """Cluster player shirt colours into two teams and label every track.

    * Players: K-means (k=2) on per-crop shirt colours, then a majority vote per track.
    * Players whose colour fits neither team (optional): role ``other``, no team.
      With the COCO fallback model this is how referees and goalkeepers are separated.
    * Referees: no team.
    * Goalkeepers: team from position (see ``assign_goalkeepers``).

    Returns:
        One row per track (``TEAM_COLUMNS``) and the fitted classifier.
    """
    t = cfg["teams"]
    roles = tracks.groupby("tracker_id")["class_name"].first()
    player_ids = roles[roles == "player"].index
    player_colours = colours[colours["tracker_id"].isin(player_ids)]
    clf = TeamClassifier().fit(player_colours[["L", "a", "b"]].to_numpy())

    per_crop = colours.assign(team=clf.predict(colours[["L", "a", "b"]].to_numpy()))
    vote = per_crop.groupby("tracker_id")["team"].agg(lambda s: s.value_counts().idxmax())
    median = colours.groupby("tracker_id")[["L", "a", "b"]].median()
    n_crops = colours.groupby("tracker_id").size()

    teams = pd.DataFrame({"tracker_id": roles.index, "role": roles.values})
    teams["team_id"] = teams["tracker_id"].map(vote).fillna(NO_TEAM).astype(int)
    teams["n_crops"] = teams["tracker_id"].map(n_crops).fillna(0).astype(int)
    teams = teams.join(median, on="tracker_id")
    has_colour = teams["L"].notna()
    teams["colour_distance"] = np.nan
    if has_colour.any():
        d = clf.distances(teams.loc[has_colour, ["L", "a", "b"]].to_numpy())
        teams.loc[has_colour, "colour_distance"] = d[np.arange(len(d)), teams.loc[has_colour, "team_id"].clip(0)]

    is_player = teams["role"] == "player"
    if t.get("flag_colour_outliers", True) and is_player.sum() > 2:
        typical = teams.loc[is_player, "colour_distance"].median()
        limit = max(t["outlier_factor"] * typical, t["min_outlier_distance"])
        outlier = is_player & (teams["colour_distance"] > limit)
        teams.loc[outlier, ["role", "team_id"]] = ["other", NO_TEAM]

    teams.loc[teams["role"] == "referee", "team_id"] = NO_TEAM
    team_of = dict(zip(teams.loc[teams["role"] == "player", "tracker_id"],
                       teams.loc[teams["role"] == "player", "team_id"]))
    gks = teams.loc[teams["role"] == "goalkeeper", "tracker_id"].tolist()
    for gk, team in assign_goalkeepers(tracks, team_of, gks).items():
        teams.loc[teams["tracker_id"] == gk, "team_id"] = team
    return teams[TEAM_COLUMNS].round(2), clf
