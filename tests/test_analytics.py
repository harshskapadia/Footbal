"""Tests for possession, passes, speed/distance, heatmaps and the minimap."""

import numpy as np
import pandas as pd

from src.analytics.kinematics import compute_kinematics, track_speeds
from src.analytics.passes import control_spells, detect_passes, pass_summary, passing_network
from src.analytics.plots import heatmap_grid, plot_heatmaps, plot_passing_network
from src.analytics.possession import NO_OWNER, compute_possession, debounce, possession_share
from src.pitch.template import PitchTemplate
from src.utils.config import load_config
from src.visualization.minimap import Minimap, overlay

FPS = 25.0


def test_debounce_drops_short_spells_only() -> None:
    owner = np.array([1, 1, 1, 1, 2, 2, NO_OWNER, 3, 3, 3])
    assert debounce(owner, 3).tolist() == [1, 1, 1, 1, NO_OWNER, NO_OWNER, NO_OWNER, 3, 3, 3]


def scenario() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Player 1 (team 0) holds the ball, passes to teammate 2, who loses it to opponent 3."""
    frames = range(60)
    pos = {1: (10.0, 30.0), 2: (30.0, 30.0), 3: (45.0, 30.0)}
    team = {1: 0, 2: 0, 3: 1}
    rows = [{"frame": f, "tracker_id": t, "team_id": team[t], "role": "player",
             "x_m": x, "y_m": y, "x_px": x * 10, "y_px": y * 10} for f in frames for t, (x, y) in pos.items()]
    ball_x = np.concatenate([np.full(15, 10.5), np.linspace(10.5, 29.5, 10), np.full(15, 29.5),
                             np.linspace(29.5, 44.5, 10), np.full(10, 44.5)])
    ball = pd.DataFrame({"frame": list(frames), "x_m": ball_x, "y_m": 30.0, "x_px": ball_x * 10, "y_px": 300.0,
                         "interpolated": False})
    return pd.DataFrame(rows), ball


def test_possession_carries_over_while_ball_travels() -> None:
    positions, ball = scenario()
    p = compute_possession(positions, ball, load_config(), FPS)
    assert p.loc[0, "owner_id"] == 1
    assert p.loc[20, "owner_id"] == NO_OWNER and p.loc[20, "team_in_possession"] == 0
    assert p.loc[55, "owner_id"] == 3 and p.loc[55, "team_in_possession"] == 1
    share = possession_share(p)
    assert share[0] + share[1] == 100.0 and share[0] > share[1]


def test_possession_in_pixels_matches_metres() -> None:
    positions, ball = scenario()
    cfg = load_config()
    cfg["analytics"]["possession_radius_px"] = 20
    metres = compute_possession(positions, ball, cfg, FPS, use_pitch=True)
    pixels = compute_possession(positions, ball, cfg, FPS, use_pitch=False)
    assert (metres["owner_id"] == pixels["owner_id"]).all()


def test_passes_completed_and_turnover() -> None:
    positions, ball = scenario()
    p = compute_possession(positions, ball, load_config(), FPS)
    assert control_spells(p)["owner_id"].tolist() == [1, 2, 3]
    passes = detect_passes(p, positions, FPS)
    assert passes[["from_id", "to_id", "completed"]].values.tolist() == [[1, 2, True], [2, 3, False]]
    assert passes["distance_m"].tolist() == [20.0, 15.0]
    s = pass_summary(passes)
    assert s[0] == {"attempted": 2, "completed": 1, "accuracy_pct": 50.0}


def test_short_and_late_passes_are_ignored() -> None:
    poss = pd.DataFrame({"frame": range(200),
                         "owner_id": [1] * 10 + [NO_OWNER] * 120 + [2] * 10 + [4] * 60,
                         "owner_team": [0] * 10 + [NO_OWNER] * 120 + [0] * 10 + [0] * 60})
    positions = pd.DataFrame({"frame": [139, 140], "tracker_id": [2, 4], "x_m": [50.0, 51.0], "y_m": [30.0, 30.0]})
    passes = detect_passes(poss, positions, FPS, max_gap_s=4.0, min_distance_m=3.0)
    assert passes.empty                        # 1->2: 4.8 s gap; 2->4: only 1 m apart


def test_passing_network_counts_pairs() -> None:
    passes = pd.DataFrame({"frame_start": [0, 10, 20], "frame_end": [5, 15, 25], "from_id": [1, 2, 1],
                           "to_id": [2, 1, 3], "team": [0, 0, 0], "to_team": [0, 0, 1],
                           "completed": [True, True, False], "distance_m": [10.0, 10.0, 10.0]})
    positions = pd.DataFrame({"tracker_id": [1, 2, 3], "team_id": [0, 0, 1], "x_m": [10.0, 20.0, 30.0],
                              "y_m": [30.0, 30.0, 30.0]})
    nodes, edges = passing_network(passes, positions, team=0)
    assert edges.to_dict("records") == [{"from_id": 1, "to_id": 2, "count": 2}]
    assert sorted(nodes["tracker_id"]) == [1, 2]


def test_speed_and_distance_of_constant_run() -> None:
    # 5 m/s (18 km/h) along x for 4 s, with a 1-frame-long position glitch
    f = np.arange(100)
    track = pd.DataFrame({"frame": f, "tracker_id": 1, "team_id": 0, "role": "player",
                          "x_m": 10 + 5 * f / FPS, "y_m": 30.0})
    s = track_speeds(track, FPS, window=1, max_speed_kmh=40)
    np.testing.assert_allclose(s["speed_kmh"].dropna(), 18.0, atol=1e-6)
    _, stats = compute_kinematics(track, FPS, window=5)
    assert abs(stats.loc[0, "distance_m"] - 5 * 99 / FPS) < 0.5
    assert abs(stats.loc[0, "max_speed_kmh"] - 18.0) < 0.5


def test_gaps_and_glitches_are_not_counted() -> None:
    track = pd.DataFrame({"frame": [0, 1, 2, 50, 51], "tracker_id": 1, "team_id": 0, "role": "player",
                          "x_m": [0.0, 0.2, 0.4, 30.0, 30.2], "y_m": 0.0})
    s = track_speeds(track, FPS, window=1, max_speed_kmh=40)
    assert np.isnan(s["step_m"].iloc[3])                     # jump across a 48-frame gap ignored
    glitch = track.assign(frame=[0, 1, 2, 3, 4], x_m=[0.0, 0.2, 9.0, 0.6, 0.8])
    assert np.isnan(track_speeds(glitch, FPS, 1, 40)["speed_kmh"].iloc[2])   # 220 m/s reading dropped


def test_heatmap_sums_to_100_and_peaks_where_players_are() -> None:
    t = PitchTemplate()
    xy = np.tile([[20.0, 10.0]], (500, 1))
    grid = heatmap_grid(xy, t, bin_m=1.0, sigma_m=2.0)
    assert abs(grid.sum() - 100) < 1e-6
    assert np.unravel_index(grid.argmax(), grid.shape) == (10, 20)


def test_charts_and_minimap_render() -> None:
    positions, _ = scenario()
    t = PitchTemplate()
    figs = plot_heatmaps(positions, t, {0: "A", 1: "B"})
    assert set(figs) == {0, 1}
    nodes = pd.DataFrame({"tracker_id": [1, 2], "x_m": [10.0, 30.0], "y_m": [30.0, 30.0], "passes": [1, 1]})
    edges = pd.DataFrame({"from_id": [1], "to_id": [2], "count": [1]})
    assert plot_passing_network(nodes, edges, t, 0, "A") is not None
    mm = Minimap(t, {0: (0, 0, 255), 1: (255, 0, 0)})
    img = mm.render(positions[positions["frame"] == 0], np.array([10.5, 30.0]), owner_id=1)
    assert img.shape[2] == 3 and img.shape[1] > img.shape[0]
    frame = np.zeros((720, 1280, 3), np.uint8)
    assert overlay(frame, img).sum() > 0
