"""Load ``config.yaml`` and resolve repository-relative paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read a YAML config file into a dict.

    Args:
        path: Config file path. Defaults to ``config.yaml`` at the repo root.

    Returns:
        The parsed configuration.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path: str | Path) -> Path:
    """Return ``path`` as absolute, interpreting relative paths from the repo root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p
