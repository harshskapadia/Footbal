"""Generate a synthetic match clip plus noisy detections, keypoints and ground truth.

Usage:
    python -m src.simulation --out data/raw/synthetic_match.mp4 --seconds 20
    # -> data/raw/synthetic_match.mp4
    #    data/synthetic/synthetic_match/{detections.csv, keypoints.csv, ground_truth/...}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.detection.classes import BALL, NAME_TO_ID
from src.pitch.template import PitchTemplate
from src.simulation.match import MatchSimulation, simulate_match
from src.simulation.render import apply_h, camera_homographies, render_match
from src.utils.config import resolve_path
from src.utils.progress import progress


def visible_fraction(box: np.ndarray, w: int, h: int) -> float:
    """Fraction of a box's area inside a ``w`` x ``h`` image."""
    x1, y1, x2, y2 = box
    ix = max(0.0, min(x2, w) - max(x1, 0)) * max(0.0, min(y2, h) - max(y1, 0))
    return ix / max((x2 - x1) * (y2 - y1), 1e-9)


def noisy_detections(
    rng: np.random.Generator, f: int, sim: MatchSimulation, boxes: dict[int, np.ndarray],
    ball_box: np.ndarray, size: tuple[int, int],
) -> tuple[list[dict], list[dict]]:
    """Turn ground-truth boxes into detector-like output: jitter, misses, a few false balls.

    Returns:
        Detection rows (stage 2 CSV format) and ground-truth box rows.
    """
    w, h = size
    det, gt = [], []
    for a in sim.agents:
        box = boxes[a.agent_id]
        if visible_fraction(box, w, h) < 0.6:
            continue
        clipped = np.clip(box, [0, 0, 0, 0], [w, h, w, h])
        gt.append({"frame": f, "agent_id": a.agent_id, "x1": clipped[0], "y1": clipped[1],
                   "x2": clipped[2], "y2": clipped[3]})
        if rng.random() < 0.03:               # missed detection
            continue
        jitter = rng.normal(0, 1.0, 4)
        det.append({"frame": f, "class_id": NAME_TO_ID[a.role], "class_name": a.role,
                    "confidence": round(float(rng.uniform(0.55, 0.95)), 4),
                    **dict(zip(["x1", "y1", "x2", "y2"], np.round(clipped + jitter, 1), strict=True))})
    if visible_fraction(ball_box, w, h) == 1.0 and rng.random() < 0.7:
        det.append({"frame": f, "class_id": BALL, "class_name": "ball",
                    "confidence": round(float(rng.uniform(0.3, 0.8)), 4),
                    **dict(zip(["x1", "y1", "x2", "y2"], np.round(ball_box + rng.normal(0, 0.5, 4), 1), strict=True))})
    elif rng.random() < 0.02:                 # false positive somewhere on the grass
        x, y = rng.uniform(50, w - 50), rng.uniform(0.3 * h, h - 20)
        det.append({"frame": f, "class_id": BALL, "class_name": "ball", "confidence": 0.3,
                    "x1": x - 4, "y1": y - 4, "x2": x + 4, "y2": y + 4})
    return det, gt


def noisy_keypoints(rng: np.random.Generator, f: int, h: np.ndarray, template: PitchTemplate,
                    size: tuple[int, int]) -> list[dict]:
    """Project the 32 pitch keypoints, add pixel noise and drop some, like a keypoint model."""
    w, hh = size
    pts = apply_h(h, template.vertices)
    rows = []
    for k, (x, y) in enumerate(pts):
        inside = 0 <= x < w and 0 <= y < hh
        seen = inside and rng.random() > 0.1
        nx, ny = (x, y) + rng.normal(0, 1.5, 2)
        rows.append({"frame": f, "keypoint": k, "x": round(float(nx), 2), "y": round(float(ny), 2),
                     "confidence": round(float(rng.uniform(0.7, 0.99)), 3) if seen else 0.0})
    return rows


def write_ground_truth(sim: MatchSimulation, gt_boxes: list[dict], out: Path) -> None:
    """Save true positions, teams, possession and passes for evaluation."""
    out.mkdir(parents=True, exist_ok=True)
    n_frames, n_agents, _ = sim.positions.shape
    pd.DataFrame({
        "frame": np.repeat(np.arange(n_frames), n_agents),
        "agent_id": np.tile(np.arange(n_agents), n_frames),
        "x_m": sim.positions[:, :, 0].ravel().round(3),
        "y_m": sim.positions[:, :, 1].ravel().round(3),
    }).to_csv(out / "positions.csv", index=False)
    pd.DataFrame([{"agent_id": a.agent_id, "team": a.team, "role": a.role} for a in sim.agents]) \
        .to_csv(out / "agents.csv", index=False)
    pd.DataFrame({"frame": np.arange(n_frames), "x_m": sim.ball[:, 0].round(3),
                  "y_m": sim.ball[:, 1].round(3), "holder": sim.holder}).to_csv(out / "ball.csv", index=False)
    pd.DataFrame(sim.passes).to_csv(out / "passes.csv", index=False)
    pd.DataFrame(gt_boxes).round(1).to_csv(out / "boxes.csv", index=False)


def generate(out_video: Path, seconds: float = 10.0, fps: float = 25.0, seed: int = 0,
             size: tuple[int, int] = (1280, 720), data_dir: Path | None = None) -> Path:
    """Simulate, render and save a synthetic match.

    Returns:
        The folder with the CSVs (default ``data/synthetic/<video stem>/``).
    """
    rng = np.random.default_rng(seed)
    template = PitchTemplate()
    n_frames = int(seconds * fps)
    sim = simulate_match(n_frames, fps, seed, template)
    homographies = camera_homographies(sim.ball, fps, size, template)

    data_dir = data_dir or resolve_path("data/synthetic") / out_video.stem
    data_dir.mkdir(parents=True, exist_ok=True)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter.fourcc(*"mp4v"), fps, size)
    det_rows, kp_rows, gt_rows = [], [], []
    frames = render_match(sim, homographies, size, template)
    for f, (frame, boxes, ball_box) in enumerate(progress(frames, n_frames, "simulate")):
        writer.write(frame)
        det, gt = noisy_detections(rng, f, sim, boxes, ball_box, size)
        det_rows += det
        gt_rows += gt
        kp_rows += noisy_keypoints(rng, f, homographies[f], template, size)
    writer.release()

    pd.DataFrame(det_rows).to_csv(data_dir / "detections.csv", index=False)
    pd.DataFrame(kp_rows).to_csv(data_dir / "keypoints.csv", index=False)
    write_ground_truth(sim, gt_rows, data_dir / "ground_truth")
    (data_dir / "meta.json").write_text(json.dumps(
        {"video": str(out_video), "fps": fps, "frames": n_frames, "seed": seed,
         "passes": len(sim.passes)}, indent=2))
    return data_dir


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Generate a synthetic match for testing.")
    p.add_argument("--out", type=Path, default=Path("data/raw/synthetic_match.mp4"))
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--seed", type=int, default=2)
    args = p.parse_args(argv)
    data_dir = generate(resolve_path(args.out), args.seconds, args.fps, args.seed)
    print(f"Video: {resolve_path(args.out)}\nData:  {data_dir}")


if __name__ == "__main__":
    main()
