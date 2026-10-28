"""Tests for shirt colour features, the K-means classifier and team/role assignment."""

import cv2
import numpy as np
import pandas as pd
import pytest

from src.team_assignment.assign import assign_goalkeepers, assign_teams, sample_track_frames
from src.team_assignment.classifier import TeamClassifier
from src.team_assignment.features import shirt_colour, shirt_crop
from src.utils.config import load_config

RED, BLUE, BLACK, GRASS = (40, 40, 200), (221, 171, 108), (25, 25, 25), (60, 140, 60)


def lab(bgr: tuple[int, int, int]) -> np.ndarray:
    return cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)


def test_shirt_colour_ignores_grass() -> None:
    crop = np.full((20, 20, 3), GRASS, np.uint8)
    crop[:, :8] = RED                     # 40 % shirt, 60 % grass
    np.testing.assert_allclose(shirt_colour(crop), lab(RED), atol=1)


def test_shirt_colour_none_when_all_grass() -> None:
    assert shirt_colour(np.full((10, 10, 3), GRASS, np.uint8)) is None


def test_shirt_crop_takes_torso_and_rejects_tiny_boxes() -> None:
    frame = np.zeros((100, 100, 3), np.uint8)
    crop = shirt_crop(frame, np.array([10, 0, 30, 100]), top=0.2, bottom=0.5, side=0.25)
    assert crop.shape[:2] == (30, 10)
    assert shirt_crop(frame, np.array([10, 10, 12, 14])) is None
    assert shirt_crop(frame, np.array([95, 0, 120, 100]), side=0.0) is not None  # clipped at border


def test_classifier_separates_kits_and_orders_by_lightness() -> None:
    rng = np.random.default_rng(0)
    feats = np.vstack([lab(RED) + rng.normal(0, 3, (30, 3)), lab(BLUE) + rng.normal(0, 3, (30, 3))])
    clf = TeamClassifier().fit(feats)
    pred = clf.predict(feats)
    assert len(set(pred[:30])) == 1 and len(set(pred[30:])) == 1 and pred[0] != pred[-1]
    assert clf.centroids[0, 0] <= clf.centroids[1, 0]          # team 0 = darker kit
    assert pred[0] == 0                                         # red is darker than sky blue


def test_classifier_needs_two_samples() -> None:
    with pytest.raises(ValueError):
        TeamClassifier().fit(np.zeros((1, 3)))


def make_tracks() -> pd.DataFrame:
    """Frames 0-9: red players on the left, blue on the right, a referee, two goalkeepers.

    Tracks 1-2 and 9-11 are red, 3-4 and 12-14 blue, 8 is an odd-coloured "player".
    """
    rows = []
    people = [(1, "player", 100), (2, "player", 200), (3, "player", 700), (4, "player", 800),
              (5, "referee", 450), (6, "goalkeeper", 20), (7, "goalkeeper", 950), (8, "player", 500),
              (9, "player", 150), (10, "player", 250), (11, "player", 300),
              (12, "player", 650), (13, "player", 750), (14, "player", 850)]
    for f in range(10):
        for tid, role, x in people:
            rows.append({"frame": f, "tracker_id": tid, "class_name": role,
                         "x1": x, "y1": 100, "x2": x + 20, "y2": 160})
    return pd.DataFrame(rows)


def make_colours(tracks: pd.DataFrame) -> pd.DataFrame:
    kit = {1: RED, 2: RED, 3: BLUE, 4: BLUE, 5: BLACK, 6: (0, 220, 230), 7: (0, 140, 255),
           8: (200, 0, 200),  # a magenta "player" that matches neither team
           9: RED, 10: RED, 11: RED, 12: BLUE, 13: BLUE, 14: BLUE}
    rows = []
    rng = np.random.default_rng(1)
    for r in tracks.itertuples():
        L, a, b = lab(kit[r.tracker_id]) + rng.normal(0, 2, 3)
        rows.append({"frame": r.frame, "tracker_id": r.tracker_id, "L": L, "a": a, "b": b})
    return pd.DataFrame(rows)


def test_assign_teams_roles_and_goalkeepers() -> None:
    tracks = make_tracks()
    teams, _ = assign_teams(tracks, make_colours(tracks), load_config())
    t = teams.set_index("tracker_id")
    red, blue = t.loc[1, "team_id"], t.loc[3, "team_id"]
    assert {red, blue} == {0, 1}
    assert all(t.loc[[2, 9, 10, 11], "team_id"] == red)
    assert all(t.loc[[4, 12, 13, 14], "team_id"] == blue)
    assert t.loc[5, "role"] == "referee" and t.loc[5, "team_id"] == -1
    assert t.loc[8, "role"] == "other" and t.loc[8, "team_id"] == -1
    assert t.loc[6, "team_id"] == red      # left goalkeeper is nearest the red players
    assert t.loc[7, "team_id"] == blue


def test_assign_goalkeepers_without_teammates_in_view() -> None:
    tracks = make_tracks()
    only_gk = tracks[tracks["tracker_id"] == 6]
    assert assign_goalkeepers(only_gk, {1: 0}, [6]) == {6: -1}


def test_sample_track_frames_spreads_samples() -> None:
    tracks = make_tracks()
    s = sample_track_frames(tracks, per_track=3)
    assert s.groupby("tracker_id").size().max() == 3
    assert sorted(s[s["tracker_id"] == 1]["frame"]) == [0, 4, 9]


def test_trimmed_fit_ignores_odd_colour_crops() -> None:
    rng = np.random.default_rng(0)
    feats = np.vstack([lab(RED) + rng.normal(0, 3, (50, 3)), lab(BLUE) + rng.normal(0, 3, (50, 3)),
                       lab((200, 0, 200)) + rng.normal(0, 3, (10, 3))])
    clf = TeamClassifier().fit(feats)
    np.testing.assert_allclose(np.sort(clf.centroids[:, 0]), np.sort([lab(RED)[0], lab(BLUE)[0]]), atol=2)
