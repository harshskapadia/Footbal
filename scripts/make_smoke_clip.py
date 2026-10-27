"""Build a short smoke-test clip from the sample images that ship with Ultralytics.

This is NOT football footage. It only checks that the pipeline runs end to end
before the real clips are downloaded (see README "Data and models").

Usage: python scripts/make_smoke_clip.py --out data/raw/smoke.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics.utils import ASSETS


def make_clip(out: Path, seconds: int = 3, fps: int = 25, size: tuple[int, int] = (1280, 720)) -> Path:
    """Write a clip that slowly pans across each sample image (simulated camera motion)."""
    w, h = size
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    images = [cv2.imread(str(ASSETS / name)) for name in ("zidane.jpg", "bus.jpg")]
    n = seconds * fps // len(images)
    for img in images:
        scale = max(w / img.shape[1], h / img.shape[0]) * 1.2
        big = cv2.resize(img, None, fx=scale, fy=scale)
        max_dx, max_dy = big.shape[1] - w, big.shape[0] - h
        for t in np.linspace(0, 1, n):
            x, y = int(t * max_dx), int(t * max_dy / 2)
            writer.write(big[y:y + h, x:x + w])
    writer.release()
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path("data/raw/smoke.mp4"))
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Wrote {make_clip(args.out)}")
