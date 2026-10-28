"""Tests for ByteTrack wrapping, class stabilisation and ball interpolation."""

import numpy as np
import pandas as pd
import supervision as sv

from src.detection.classes import BALL, GOALKEEPER, PLAYER, REFEREE
from src.tracking.ball import ball_table, clean_ball_track, interpolate_ball, remove_ball_outliers
from src.tracking.tracker import build_tracker, stabilise_track_classes, track_people
from src.utils.config import load_config


def people_frame(t: int, include: tuple[bool, bool] = (True, True)) -> sv.Detections:
    """Two players running towards each other on the same row, plus a ball."""
    boxes, classes = [], []
    if include[0]:
        boxes.append([100 + 8 * t, 300, 140 + 8 * t, 400]); classes.append(PLAYER)
    if include[1]:
        boxes.append([600 - 8 * t, 310, 640 - 8 * t, 410]); classes.append(REFEREE)
    boxes.append([400, 500, 410, 510]); classes.append(BALL)
    return sv.Detections(xyxy=np.array(boxes, float), confidence=np.full(len(boxes), 0.9),
                         class_id=np.array(classes))


def test_ids_stay_stable_through_missed_detections() -> None:
    tracker = build_tracker(load_config(), fps=25)
    ids_by_class: dict[int, set[int]] = {PLAYER: set(), REFEREE: set()}
    for t in range(40):
        include = (t not in (10, 11, 12), True)  # player missed for 3 frames
        out = track_people(tracker, people_frame(t, include))
        assert BALL not in out.class_id
        for c, tid in zip(out.class_id, out.tracker_id):
            ids_by_class[int(c)].add(int(tid))
    assert len(ids_by_class[PLAYER]) == 1
    assert len(ids_by_class[REFEREE]) == 1
    assert ids_by_class[PLAYER] != ids_by_class[REFEREE]


def test_stabilise_track_classes_majority_vote() -> None:
    tracks = pd.DataFrame({
        "tracker_id": [1, 1, 1, 2, 2],
        "class_id": [GOALKEEPER, PLAYER, GOALKEEPER, REFEREE, REFEREE],
        "class_name": ["goalkeeper", "player", "goalkeeper", "referee", "referee"],
    })
    out = stabilise_track_classes(tracks)
    assert out["class_id"].tolist() == [GOALKEEPER] * 3 + [REFEREE] * 2
    assert out["class_name"].tolist() == ["goalkeeper"] * 3 + ["referee"] * 2


def ball_rows(positions: dict[int, float]) -> pd.DataFrame:
    """Ball detections at x=position, 10 px wide, keyed by frame."""
    return pd.DataFrame([
        {"frame": f, "class_id": BALL, "confidence": 0.8, "x1": x, "y1": 50.0, "x2": x + 10, "y2": 60.0}
        for f, x in positions.items()
    ])


def test_interpolation_fills_short_gaps_only() -> None:
    # detected at 0, 4 (gap of 3) and 20 (gap of 15); nothing after 20
    table = ball_table(ball_rows({0: 0.0, 4: 40.0, 20: 200.0}), n_frames=25)
    out = interpolate_ball(table, max_gap=5)
    np.testing.assert_allclose(out.loc[1:3, "x1"], [10.0, 20.0, 30.0])
    assert out.loc[1:3, "interpolated"].all()
    assert out.loc[5:19, "x1"].isna().all()      # long gap left empty
    assert out.loc[21:, "x1"].isna().all()       # no extrapolation past the end
    assert not out.loc[[0, 4, 20], "interpolated"].any()


def test_outlier_far_from_both_neighbours_is_removed() -> None:
    table = ball_table(ball_rows({0: 0.0, 1: 10.0, 2: 900.0, 3: 30.0, 4: 40.0}), n_frames=5)
    out = remove_ball_outliers(table, max_speed_px=80)
    assert np.isnan(out.loc[2, "x1"])
    assert out["x1"].notna().sum() == 4


def test_clean_ball_track_replaces_outlier_with_interpolation() -> None:
    out = clean_ball_track(ball_rows({0: 0.0, 1: 10.0, 2: 900.0, 3: 30.0}), 4, max_gap=5, max_speed_px=80)
    assert out.loc[2, "x1"] == 20.0
    assert bool(out.loc[2, "interpolated"])


def test_ball_table_keeps_best_detection_per_frame() -> None:
    rows = pd.concat([ball_rows({0: 0.0}), ball_rows({0: 50.0}).assign(confidence=0.95)])
    table = ball_table(rows, n_frames=2)
    assert table.loc[0, "x1"] == 50.0
    assert np.isnan(table.loc[1, "x1"])


def test_outliers_at_clip_edges_are_removed() -> None:
    table = ball_table(ball_rows({0: 900.0, 1: 10.0, 2: 20.0, 3: 30.0, 4: 1500.0}), n_frames=5)
    out = remove_ball_outliers(table, max_speed_px=80)
    assert out["x1"].isna().tolist() == [True, False, False, False, True]


def test_clip_without_any_ball_detection() -> None:
    empty = pd.DataFrame(columns=["frame", "class_id", "confidence", "x1", "y1", "x2", "y2"])
    out = clean_ball_track(empty, n_frames=3, max_gap=5, max_speed_px=80)
    assert len(out) == 3
    assert out["x1"].isna().all()
    assert not out["interpolated"].any()
