"""End-to-end test: the full pipeline on a short synthetic match, scored against ground truth."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.pipeline import PipelineOptions, run_pipeline
from src.simulation.__main__ import generate
from src.simulation.evaluate import evaluate
from src.utils.cli import RunContext
from src.utils.config import load_config
from src.utils.paths import RunPaths


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, dict]:
    tmp = tmp_path_factory.mktemp("synthetic")
    video = tmp / "match.mp4"
    data = generate(video, seconds=8, seed=2, data_dir=tmp / "data")
    ctx = RunContext(load_config(), video, RunPaths.for_clip(tmp / "outputs", video), None)
    result = run_pipeline(ctx, PipelineOptions(data / "detections.csv", data / "keypoints.csv"))
    return result.root, data, result.summary


def test_pipeline_writes_every_output(synthetic_run) -> None:
    root, _, summary = synthetic_run
    for name in ("tracks.csv", "teams.csv", "positions.csv", "possession.csv", "passes.csv", "player_stats.csv",
                 "stats.json", "team_summary.csv", "heatmap_team0.png", "passing_network_team1.png",
                 "annotated.mp4", "annotated_h264.mp4"):
        assert (root / name).exists(), name
    assert json.loads((root / "stats.json").read_text())["pitch_mapping"] is True
    assert summary["warnings"] == []
    assert len(pd.read_csv(root / "team_summary.csv")) == 2


def test_pipeline_matches_ground_truth(synthetic_run) -> None:
    root, data, _ = synthetic_run
    report = evaluate(root, data)
    assert report["teams"]["team_accuracy"] >= 0.95
    assert report["pitch_positions"]["people"]["mean_m"] < 0.5
    assert abs(report["possession"]["estimated_pct"][0] - report["possession"]["true_pct"][0]) < 5
    assert report["passes"]["recall"] >= 0.75 and report["passes"]["precision"] >= 0.75
    assert abs(report["distance"]["mean_error_pct"]) < 10


def test_pipeline_without_pitch_model_degrades_gracefully(synthetic_run, tmp_path: Path) -> None:
    _, data, _ = synthetic_run
    video = data.parent / "match.mp4"
    cfg = load_config()
    cfg["models"]["pitch_detector"] = "models/does-not-exist.pt"
    ctx = RunContext(cfg, video, RunPaths.for_clip(tmp_path, video), 60)
    result = run_pipeline(ctx, PipelineOptions(data / "detections.csv", None, render_video=False))
    assert result.summary["pitch_mapping"] is False
    assert any("Pitch keypoint model not found" in w for w in result.warnings)
    assert (result.root / "possession.csv").exists() and not (result.root / "player_stats.csv").exists()
