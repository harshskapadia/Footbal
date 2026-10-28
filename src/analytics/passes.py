"""Passes and turnovers from the sequence of ball owners, and passing networks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.possession import NO_OWNER

PASS_COLUMNS = ["frame_start", "frame_end", "from_id", "to_id", "team", "to_team", "completed",
                "distance_m"]


def control_spells(possession: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-frame ownership into spells: ``owner_id, team, first_frame, last_frame``."""
    owned = possession[possession["owner_id"] != NO_OWNER]
    if owned.empty:
        return pd.DataFrame(columns=["owner_id", "team", "first_frame", "last_frame"])
    new_spell = (owned["owner_id"] != owned["owner_id"].shift()) | (owned["frame"].diff() > 1)
    spell_id = new_spell.cumsum()
    return owned.groupby(spell_id).agg(
        owner_id=("owner_id", "first"), team=("owner_team", "first"),
        first_frame=("frame", "min"), last_frame=("frame", "max"),
    ).reset_index(drop=True)


def detect_passes(
    possession: pd.DataFrame, positions: pd.DataFrame | None, fps: float, max_gap_s: float = 4.0,
    min_distance_m: float = 0.0,
) -> pd.DataFrame:
    """A change of owner between consecutive control spells is a pass.

    To a teammate it is a completed pass; to an opponent it is a failed pass
    (turnover). Changes separated by more than ``max_gap_s`` of loose ball are
    ignored, as are "passes" from a player to themselves after a short loss and,
    when pitch positions are known, passes shorter than ``min_distance_m``
    (typically one player whose track was split into two IDs).

    Returns:
        One row per pass (``PASS_COLUMNS``); ``distance_m`` is NaN without pitch positions.
    """
    spells = control_spells(possession)
    xy = positions.set_index(["frame", "tracker_id"])[["x_m", "y_m"]] if positions is not None else None
    rows = []
    for prev, nxt in zip(spells.itertuples(), spells.iloc[1:].itertuples()):
        if prev.owner_id == nxt.owner_id or (nxt.first_frame - prev.last_frame) / fps > max_gap_s:
            continue
        dist = np.nan
        if xy is not None:
            try:
                a = xy.loc[(prev.last_frame, prev.owner_id)].to_numpy(float)
                b = xy.loc[(nxt.first_frame, nxt.owner_id)].to_numpy(float)
                dist = round(float(np.linalg.norm(a - b)), 1)
            except KeyError:
                pass
        if dist < min_distance_m:
            continue
        rows.append({"frame_start": int(prev.last_frame), "frame_end": int(nxt.first_frame),
                     "from_id": int(prev.owner_id), "to_id": int(nxt.owner_id), "team": int(prev.team),
                     "to_team": int(nxt.team), "completed": bool(prev.team == nxt.team), "distance_m": dist})
    return pd.DataFrame(rows, columns=PASS_COLUMNS)


def pass_summary(passes: pd.DataFrame) -> dict[int, dict[str, float]]:
    """Attempted, completed and accuracy per team."""
    out = {}
    for team in (0, 1):
        p = passes[passes["team"] == team]
        done = int(p["completed"].sum())
        out[team] = {"attempted": int(len(p)), "completed": done,
                     "accuracy_pct": round(100 * done / len(p), 1) if len(p) else 0.0}
    return out


def passing_network(passes: pd.DataFrame, positions: pd.DataFrame, team: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nodes (players at their average position) and weighted edges for one team.

    Returns:
        ``nodes``: ``tracker_id, x_m, y_m, passes`` (made + received);
        ``edges``: ``from_id, to_id, count`` (direction ignored).
    """
    done = passes[(passes["team"] == team) & passes["completed"]]
    pairs = pd.DataFrame({"a": done[["from_id", "to_id"]].min(axis=1), "b": done[["from_id", "to_id"]].max(axis=1)})
    edges = pairs.value_counts().rename("count").reset_index().rename(columns={"a": "from_id", "b": "to_id"})
    involved = pd.concat([done["from_id"], done["to_id"]]).value_counts()
    mean_pos = positions[positions["team_id"] == team].groupby("tracker_id")[["x_m", "y_m"]].mean()
    ids = [t for t in involved.index if t in mean_pos.index]
    nodes = pd.DataFrame({"tracker_id": ids, "x_m": mean_pos.loc[ids, "x_m"].to_numpy(),
                          "y_m": mean_pos.loc[ids, "y_m"].to_numpy(), "passes": involved.loc[ids].to_numpy()})
    return nodes, edges
