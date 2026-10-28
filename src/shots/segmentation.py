"""Find cuts in edited footage and decide which shots come from the wide tactical camera.

Highlights mix the wide camera with close-ups, replays, crowd shots and graphics.
Tracking, ball interpolation and homographies only make sense within one
continuous wide shot, so the pipeline analyses those and restarts at each cut.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.team_assignment.features import grass_mask
from src.utils.progress import ProgressCallback, progress
from src.utils.video import read_frames

NOT_ANALYSED = -1
SHOT_COLUMNS = ["shot_id", "start", "end", "frames", "grass_ratio", "person_height", "wide"]


def frame_signature(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """Hue/saturation histogram (for cut detection) and share of grass pixels (for shot type)."""
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist, float(grass_mask(small).mean())


def scan_video(path: Path, max_frames: int | None, total: int | None,
               callback: ProgressCallback | None = None) -> pd.DataFrame:
    """Per-frame histogram change from the previous frame and grass ratio.

    Returns:
        ``frame, change, grass_ratio`` (``change`` is a Bhattacharyya distance, 0 = identical).
    """
    rows = []
    prev: np.ndarray | None = None
    for f, frame in enumerate(progress(read_frames(path, max_frames), total, "shots", callback)):
        hist, grass = frame_signature(frame)
        change = 0.0 if prev is None else float(cv2.compareHist(prev, hist, cv2.HISTCMP_BHATTACHARYYA))
        rows.append({"frame": f, "change": round(change, 4), "grass_ratio": round(grass, 3)})
        prev = hist
    return pd.DataFrame(rows, columns=["frame", "change", "grass_ratio"])


def find_cuts(change: np.ndarray, threshold: float = 0.35, relative: float = 4.0, window: int = 12,
              min_shot_frames: int = 5) -> list[int]:
    """Frames that start a new shot.

    A cut is a large jump in colour distribution that also stands out from the
    local level of change, so fast camera pans (steady, moderate change) are not
    mistaken for cuts. Shots shorter than ``min_shot_frames`` (flashes, logo
    wipes, encoder glitches) are merged into their neighbour.
    """
    local = pd.Series(change).rolling(2 * window + 1, center=True, min_periods=1).median().to_numpy()
    candidates = np.flatnonzero((change > threshold) & (change > relative * np.maximum(local, 0.01)))
    cuts: list[int] = []
    for c in candidates:
        if c >= min_shot_frames and c <= len(change) - min_shot_frames and \
                (not cuts or c - cuts[-1] >= min_shot_frames):
            cuts.append(int(c))
    return cuts


def shot_runs(shot_ids: np.ndarray) -> list[tuple[int, int, int]]:
    """Contiguous runs of equal shot id as ``(shot_id, start, end_exclusive)``.

    Use this rather than grouping by id: "not analysed" (-1) occurs in many separate stretches.
    """
    if len(shot_ids) == 0:
        return []
    starts = np.r_[0, np.flatnonzero(np.diff(shot_ids)) + 1]
    ends = np.r_[starts[1:], len(shot_ids)]
    return [(int(shot_ids[s]), int(s), int(e)) for s, e in zip(starts, ends, strict=True)]


def segment_shots(scan: pd.DataFrame, cfg: Mapping[str, Any], fps: float) -> pd.DataFrame:
    """Split the clip at cuts and mark long, grass-dominated shots as wide (``SHOT_COLUMNS``)."""
    s = cfg["shots"]
    n = len(scan)
    cuts = find_cuts(scan["change"].to_numpy(), s["cut_threshold"], s["cut_relative"]) if s["enabled"] else []
    bounds = [0, *cuts, n]
    rows = []
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        grass = float(scan["grass_ratio"].iloc[a:b].median()) if b > a else 0.0
        wide = (not s["enabled"]) or (grass >= s["min_grass_ratio"] and (b - a) >= s["min_shot_s"] * fps)
        rows.append({"shot_id": i, "start": a, "end": b - 1, "frames": b - a, "grass_ratio": round(grass, 3),
                     "person_height": np.nan, "wide": bool(wide)})
    return pd.DataFrame(rows, columns=SHOT_COLUMNS)


def refine_with_detections(shots: pd.DataFrame, detections: pd.DataFrame, frame_height: int,
                           cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Drop "wide" shots whose players look too big (close-ups on grass) or that have no players."""
    if not cfg["shots"]["enabled"]:
        return shots
    out = shots.copy()
    people = detections[detections["class_name"] != "ball"] if len(detections) else detections
    heights = (people["y2"] - people["y1"]) / frame_height if len(people) else pd.Series(dtype=float)
    for i, shot in out.iterrows():
        in_shot = (people["frame"] >= shot["start"]) & (people["frame"] <= shot["end"]) if len(people) else []
        h = heights[in_shot] if len(people) else heights
        out.loc[i, "person_height"] = round(float(h.median()), 3) if len(h) else np.nan
        if shot["wide"] and (len(h) == 0 or h.median() > cfg["shots"]["max_person_height"]):
            out.loc[i, "wide"] = False
    return out


def frame_shot_ids(shots: pd.DataFrame, n_frames: int) -> np.ndarray:
    """Shot id of every frame, or ``NOT_ANALYSED`` (-1) outside wide shots."""
    ids = np.full(n_frames, NOT_ANALYSED, dtype=int)
    for shot in shots[shots["wide"]].itertuples():
        ids[shot.start: shot.end + 1] = shot.shot_id
    return ids


def shot_summary(shots: pd.DataFrame, fps: float) -> dict[str, Any]:
    """Headline numbers for stats.json."""
    wide = shots[shots["wide"]]
    total = int(shots["frames"].sum())
    return {"shots": int(len(shots)), "wide_shots": int(len(wide)),
            "analysed_s": round(float(wide["frames"].sum()) / fps, 1),
            "analysed_fraction": round(float(wide["frames"].sum()) / max(total, 1), 3)}
