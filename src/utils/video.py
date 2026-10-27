"""Video reading and writing helpers built on OpenCV / supervision."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import supervision as sv


def get_video_info(path: str | Path) -> sv.VideoInfo:
    """Return resolution, fps and frame count of a video file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    return sv.VideoInfo.from_video_path(str(path))


def read_frames(path: str | Path, max_frames: int | None = None) -> Iterator[np.ndarray]:
    """Yield BGR frames from a video, optionally stopping after ``max_frames``."""
    get_video_info(path)  # fail early with a clear error if missing
    for i, frame in enumerate(sv.get_video_frames_generator(str(path))):
        if max_frames is not None and i >= max_frames:
            break
        yield frame


def open_writer(path: str | Path, info: sv.VideoInfo) -> sv.VideoSink:
    """Create a video writer (use as a context manager)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return sv.VideoSink(str(path), video_info=info, codec="mp4v")
