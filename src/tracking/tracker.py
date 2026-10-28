"""ByteTrack wrapper that gives players, goalkeepers and referees stable IDs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import supervision as sv
from trackers import ByteTrackTracker

from src.detection.classes import BALL, CLASS_NAMES, class_ids, subset


def build_tracker(cfg: Mapping[str, Any], fps: float) -> ByteTrackTracker:
    """Create a ByteTrack tracker from the ``tracking`` config section."""
    t = cfg["tracking"]
    return ByteTrackTracker(
        frame_rate=fps,
        lost_track_buffer=max(1, round(t["lost_track_buffer_s"] * fps)),
        track_activation_threshold=t["track_activation_threshold"],
        high_conf_det_threshold=t["high_conf_det_threshold"],
        minimum_consecutive_frames=t["minimum_consecutive_frames"],
        minimum_iou_threshold=t["minimum_iou_threshold"],
    )


def track_people(
    tracker: ByteTrackTracker, detections: sv.Detections, frame: np.ndarray | None = None
) -> sv.Detections:
    """Update the tracker with one frame and return people that have a confirmed ID.

    The ball is excluded: it is small, fast and often missing, so it is handled
    by interpolation instead (see ``src.tracking.ball``).
    """
    people = subset(detections, class_ids(detections) != BALL) if len(detections) else detections
    tracked = tracker.update(people, frame)
    if len(tracked) == 0 or tracked.tracker_id is None:
        return tracked
    return subset(tracked, tracked.tracker_id >= 0)


def stabilise_track_classes(tracks: pd.DataFrame) -> pd.DataFrame:
    """Give each track the class it was assigned most often.

    The detector may flicker between player and goalkeeper (or referee) for the
    same person; a majority vote over the whole track removes that noise.
    """
    if tracks.empty:
        return tracks
    majority = tracks.groupby("tracker_id")["class_id"].agg(lambda s: s.value_counts().idxmax())
    out = tracks.copy()
    out["class_id"] = out["tracker_id"].map(majority).astype(int)
    out["class_name"] = out["class_id"].map(CLASS_NAMES)
    return out
