"""Tests for config loading and device selection."""

from pathlib import Path

from src.utils.config import REPO_ROOT, load_config, resolve_path
from src.utils.device import select_device


def test_default_config_has_required_sections() -> None:
    cfg = load_config()
    for key in ("paths", "models", "detection", "ball", "pitch"):
        assert key in cfg
    assert cfg["pitch"]["length_m"] == 105.0
    assert cfg["pitch"]["width_m"] == 68.0


def test_resolve_path_is_repo_relative() -> None:
    assert resolve_path("models/x.pt") == REPO_ROOT / "models" / "x.pt"
    assert resolve_path("/abs/x.pt") == Path("/abs/x.pt")


def test_select_device_respects_explicit_choice() -> None:
    assert select_device("cpu") == "cpu"
    assert select_device("auto") in {"cpu", "mps"} or select_device("auto").startswith("cuda")
