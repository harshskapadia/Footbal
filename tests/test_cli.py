"""Every stage CLI parses its arguments and fails with a clear message on bad input."""

import importlib

import pytest

STAGE_MODULES = ["src.detect", "src.track", "src.assign_teams", "src.map_pitch", "src.analyze", "src.run",
                 "src.detection.train", "src.detection.download_dataset", "src.simulation.__main__",
                 "src.simulation.evaluate"]


@pytest.mark.parametrize("module", STAGE_MODULES)
def test_help_exits_cleanly(module: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        importlib.import_module(module).main(["--help"])
    assert exc.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_run_reports_missing_input(tmp_path) -> None:
    from src.run import main

    with pytest.raises(SystemExit, match="Input video not found"):
        main(["--input", str(tmp_path / "nope.mp4"), "--output-dir", str(tmp_path)])


def test_map_pitch_requires_tracks(tmp_path) -> None:
    from src.map_pitch import main
    from src.utils.config import REPO_ROOT

    clip = REPO_ROOT / "README.md"   # any existing path; the check happens before the video is read
    with pytest.raises(SystemExit, match="run `python -m src.track` first"):
        main(["--input", str(clip), "--output-dir", str(tmp_path)])
