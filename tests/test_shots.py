"""Tests for shot segmentation and shot-aware tracking, mapping and possession (highlights)."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.analytics.passes import detect_passes
from src.analytics.possession import compute_possession
from src.detection.classes import PLAYER
from src.pipeline import PipelineOptions, run_pipeline
from src.shots.segmentation import (
    find_cuts,
    frame_shot_ids,
    refine_with_detections,
    segment_shots,
    shot_runs,
)
from src.simulation.__main__ import generate
from src.track import track_clip
from src.utils.cli import RunContext
from src.utils.config import load_config
from src.utils.paths import RunPaths

FPS = 25.0


def test_find_cuts_ignores_pans_and_tiny_shots() -> None:
    change = np.full(100, 0.03)
    change[30:40] = 0.2                 # a fast pan: high but steady change
    change[50] = 0.6                    # a real cut
    change[52] = 0.6                    # a flash 2 frames later: merged
    change[1] = 0.5                     # encoder glitch on frame 1: ignored
    assert find_cuts(change) == [50]


def test_shot_runs_splits_repeated_ids() -> None:
    ids = np.array([0, 0, -1, -1, 1, 1, -1])
    assert shot_runs(ids) == [(0, 0, 2), (-1, 2, 4), (1, 4, 6), (-1, 6, 7)]


def test_segment_and_refine_mark_wide_shots() -> None:
    cfg = load_config()
    scan = pd.DataFrame({"frame": range(150), "change": 0.02,
                         "grass_ratio": [0.9] * 60 + [0.1] * 40 + [0.85] * 50})
    scan.loc[60, "change"] = scan.loc[100, "change"] = 0.7
    shots = segment_shots(scan, cfg, FPS)
    assert shots[["start", "end", "wide"]].values.tolist() == [[0, 59, True], [60, 99, False], [100, 149, True]]
    # Shot 2 looks like pitch, but its "players" fill half the frame: a close-up on grass
    det = pd.DataFrame({"frame": [10, 120], "class_name": ["player", "player"], "y1": [500, 100], "y2": [560, 500]})
    refined = refine_with_detections(shots, det, 720, cfg)
    assert refined["wide"].tolist() == [True, False, False]
    assert frame_shot_ids(refined, 150).tolist() == [0] * 60 + [-1] * 90


def test_disabled_segmentation_is_one_wide_shot() -> None:
    cfg = load_config()
    cfg["shots"]["enabled"] = False
    scan = pd.DataFrame({"frame": range(50), "change": [0.0] * 25 + [0.9] + [0.0] * 24, "grass_ratio": 0.0})
    shots = segment_shots(scan, cfg, FPS)
    assert len(shots) == 1 and bool(shots.loc[0, "wide"])


def test_tracker_restarts_at_cuts_with_unique_ids() -> None:
    rows = [{"frame": f, "class_id": PLAYER, "class_name": "player", "confidence": 0.9,
             "x1": 100.0 + f, "y1": 100.0, "x2": 140.0 + f, "y2": 200.0} for f in range(60)]
    shot_ids = np.array([0] * 25 + [-1] * 10 + [2] * 25)
    tracks, ball = track_clip(pd.DataFrame(rows), 60, FPS, load_config(), shot_ids=shot_ids)
    spans = tracks.groupby("tracker_id")["frame"].agg(["min", "max"])
    assert len(spans) == 2                                   # same box, but a new ID after the cut
    assert not ((spans["min"] < 25) & (spans["max"] >= 25)).any()
    assert not tracks["frame"].between(25, 34).any()          # nothing tracked outside wide shots
    assert len(ball) == 60 and ball["frame"].is_unique


def test_possession_and_passes_do_not_cross_cuts() -> None:
    cfg = load_config()
    positions = pd.DataFrame([{"frame": f, "tracker_id": t, "team_id": 0, "role": "player", "x_m": x, "y_m": 30.0}
                              for f in range(80) for t, x in ((1, 10.0), (2, 30.0))])
    ball_x = [10.5] * 30 + [np.nan] * 20 + [30.5] * 30       # player 1 has it, cut, then player 2
    ball = pd.DataFrame({"frame": range(80), "x_m": ball_x, "y_m": 30.0, "interpolated": False,
                         "shot_id": [0] * 30 + [-1] * 20 + [1] * 30})
    poss = compute_possession(positions, ball, cfg, FPS)
    assert (poss.loc[30:49, "team_in_possession"] == -1).all()  # not carried into the cut
    assert detect_passes(poss, positions, FPS).empty            # 1 -> 2 spans a cut: not a pass


def write_close_up(path: Path, n: int, size: tuple[int, int] = (1280, 720)) -> None:
    """Frames that look like a close-up: dark background, one huge figure."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), FPS, size)
    for i in range(n):
        frame = np.full((size[1], size[0], 3), (70, 60, 140), np.uint8)
        cv2.rectangle(frame, (500 + i, 100), (800 + i, 720), (40, 40, 200), -1)
        w.write(frame)
    w.release()


@pytest.fixture(scope="module")
def highlights(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Wide shot (frames 0-99), close-up (100-139), wide shot again (140-239)."""
    tmp = tmp_path_factory.mktemp("highlights")
    data = generate(tmp / "match.mp4", seconds=8, seed=2, data_dir=tmp / "data")
    write_close_up(tmp / "close.mp4", 40)
    out = tmp / "highlights.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter.fourcc(*"mp4v"), FPS, (1280, 720))
    match, close = cv2.VideoCapture(str(tmp / "match.mp4")), cv2.VideoCapture(str(tmp / "close.mp4"))
    frames = [match.read()[1] for _ in range(200)]
    for f in frames[:100] + [close.read()[1] for _ in range(40)] + frames[100:200]:
        writer.write(f)
    writer.release()
    remap = {**{f: f for f in range(100)}, **{f: f + 40 for f in range(100, 200)}}
    for name in ("detections.csv", "keypoints.csv"):
        df = pd.read_csv(data / name)
        df = df[df["frame"] < 200].assign(frame=lambda d: d["frame"].map(remap))
        df.to_csv(tmp / name, index=False)
    return out, tmp


def test_pipeline_on_highlights(highlights: tuple[Path, Path]) -> None:
    video, tmp = highlights
    ctx = RunContext(load_config(), video, RunPaths.for_clip(tmp / "outputs", video), None)
    result = run_pipeline(ctx, PipelineOptions(tmp / "detections.csv", tmp / "keypoints.csv"))
    shots = pd.read_csv(result.root / "shots.csv")
    assert shots[["start", "end", "wide"]].values.tolist() == [[0, 99, True], [100, 139, False], [140, 239, True]]
    assert result.summary["shots"]["wide_shots"] == 2
    tracks = pd.read_csv(result.root / "tracks.csv")
    spans = tracks.groupby("tracker_id")["frame"].agg(["min", "max"])
    assert not ((spans["min"] < 100) & (spans["max"] >= 140)).any()
    assert not tracks["frame"].between(100, 139).any()
    homographies = pd.read_csv(result.root / "homographies.csv")
    assert not homographies.loc[100:139, "mapped"].any() and homographies.loc[150:230, "mapped"].all()
