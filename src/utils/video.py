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


def frame_count(info: sv.VideoInfo, max_frames: int | None = None) -> int | None:
    """Frames to process: the clip length capped at ``max_frames`` (None if the length is unknown)."""
    total = info.total_frames
    if total is None:
        return max_frames
    return min(total, max_frames) if max_frames else total


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


def to_h264(src: str | Path, dst: str | Path) -> Path | None:
    """Re-encode a video as H.264 so browsers (and Streamlit) can play it.

    OpenCV's pip wheels can only write ``mp4v``, which browsers refuse. Uses the
    ffmpeg binary bundled with ``imageio-ffmpeg``.

    Returns:
        ``dst``, or ``None`` if ffmpeg is unavailable or fails.
    """
    import logging
    import subprocess

    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        logging.getLogger(__name__).warning("imageio-ffmpeg not available; skipping H.264 copy")
        return None
    cmd = [exe, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logging.getLogger(__name__).warning("H.264 encode failed: %s", result.stderr.strip())
        return None
    return Path(dst)
