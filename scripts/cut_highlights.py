"""Cut short highlight clips out of a long match video.

Two modes:

1. SoccerNet game folder (uses the event labels in Labels-v2.json):
       python scripts/cut_highlights.py --soccernet "path/to/soccernet/england_epl/2016-2017/<game>" \\
           --events Goal "Shots on target" --before 10 --after 15

2. Any video you have the right to use, at times you choose (mm:ss or hh:mm:ss):
       python scripts/cut_highlights.py --video match.mp4 --at 12:30 55:10 --before 10 --after 15

Clips are written to data/raw/ as H.264 MP4, ready for `python -m src.run`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]


def parse_clock(text: str) -> float:
    """``"mm:ss"`` or ``"hh:mm:ss"`` to seconds."""
    parts = [float(p) for p in text.strip().split(":")]
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def cut(video: Path, start_s: float, duration_s: float, out: Path) -> None:
    """Re-encode ``duration_s`` seconds of ``video`` from ``start_s`` into ``out`` (H.264)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{max(start_s, 0):.2f}",
           "-i", str(video), "-t", f"{duration_s:.2f}", "-an", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "20", "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(cmd, check=True)
    print(f"wrote {out}")


def soccernet_events(game: Path, events: list[str]) -> list[tuple[int, float, str]]:
    """``(half, seconds into that half, label)`` for matching events in ``Labels-v2.json``."""
    labels = game / "Labels-v2.json"
    if not labels.exists():
        raise SystemExit(f"{labels} not found. Download it with SoccerNetDownloader "
                         "(files=['Labels-v2.json']); see README 'Real match footage'.")
    found = []
    for a in json.loads(labels.read_text())["annotations"]:
        if a.get("label") in events:
            half = int(a["gameTime"].split(" - ")[0])
            found.append((half, int(a["position"]) / 1000.0, a["label"]))
    return found


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Cut highlight clips from a long match video.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--soccernet", type=Path, help="SoccerNet game folder (with 1_720p.mkv, 2_720p.mkv)")
    src.add_argument("--video", type=Path, help="Any match video")
    p.add_argument("--events", nargs="+", default=["Goal"], help="SoccerNet event labels to cut around")
    p.add_argument("--at", nargs="+", default=[], help="Times to cut around (with --video), e.g. 12:30")
    p.add_argument("--before", type=float, default=10.0, help="Seconds before the moment")
    p.add_argument("--after", type=float, default=15.0, help="Seconds after the moment")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "raw")
    args = p.parse_args(argv)
    length = args.before + args.after

    if args.soccernet:
        events = soccernet_events(args.soccernet, args.events)
        if not events:
            raise SystemExit(f"No {args.events} events in this game.")
        name = args.soccernet.name.replace(" ", "_")[:60]
        for half, t, label in events:
            video = args.soccernet / f"{half}_720p.mkv"
            if not video.exists():
                raise SystemExit(f"{video} not found. Download the 720p videos first (README).")
            clip = args.out_dir / f"{name}_h{half}_{int(t // 60):02d}m{int(t % 60):02d}s_{label.replace(' ', '')}.mp4"
            cut(video, t - args.before, length, clip)
    else:
        if not args.video.exists():
            raise SystemExit(f"Video not found: {args.video}. Check that the download worked.")
        if not args.at:
            raise SystemExit("Give one or more times with --at, e.g. --at 12:30 55:10")
        for stamp in args.at:
            t = parse_clock(stamp)
            cut(args.video, t - args.before, length, args.out_dir / f"{args.video.stem}_{stamp.replace(':', 'm')}s.mp4")


if __name__ == "__main__":
    main()
