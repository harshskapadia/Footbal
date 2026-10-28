"""Compare pipeline outputs on a synthetic match with its ground truth.

Usage:
    python -m src.simulation.evaluate --run outputs/synthetic_match --truth data/synthetic/synthetic_match
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.detection.io import BOX_COLUMNS


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between ``(N, 4)`` and ``(M, 4)`` xyxy boxes."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_tracks_to_agents(tracks: pd.DataFrame, gt_boxes: pd.DataFrame, min_iou: float = 0.5) -> pd.Series:
    """Map each ``tracker_id`` to the ground-truth agent its boxes overlap most often."""
    hits = []
    gt_by_frame = {f: g for f, g in gt_boxes.groupby("frame")}
    for f, g in tracks.groupby("frame"):
        if f not in gt_by_frame:
            continue
        gt = gt_by_frame[f]
        iou = box_iou(g[BOX_COLUMNS].to_numpy(float), gt[BOX_COLUMNS].to_numpy(float))
        best = iou.argmax(axis=1)
        ok = iou[np.arange(len(g)), best] >= min_iou
        hits.append(pd.DataFrame({"tracker_id": g["tracker_id"].to_numpy()[ok],
                                  "agent_id": gt["agent_id"].to_numpy()[best[ok]]}))
    if not hits:
        return pd.Series(dtype=int)
    allhits = pd.concat(hits)
    return allhits.groupby("tracker_id")["agent_id"].agg(lambda s: s.value_counts().idxmax())


def tracking_metrics(tracks: pd.DataFrame, track_agent: pd.Series, gt_boxes: pd.DataFrame) -> dict[str, Any]:
    """ID fragmentation: how many track IDs each true person was split into."""
    per_agent = track_agent.groupby(track_agent).size()
    return {
        "true_people_seen": int(gt_boxes["agent_id"].nunique()),
        "track_ids": int(tracks["tracker_id"].nunique()),
        "mean_ids_per_person": round(float(per_agent.mean()), 2),
    }


def team_metrics(teams: pd.DataFrame, track_agent: pd.Series, agents: pd.DataFrame) -> dict[str, Any]:
    """Team accuracy over tracks of outfield players and goalkeepers, best label permutation."""
    t = teams.assign(agent_id=teams["tracker_id"].map(track_agent)).dropna(subset=["agent_id"])
    t = t.merge(agents.rename(columns={"team": "true_team", "role": "true_role"}), on="agent_id")
    on_team = t[t["true_team"] >= 0]
    same = (on_team["team_id"] == on_team["true_team"]).mean()
    swapped = (on_team["team_id"] == 1 - on_team["true_team"]).mean()
    refs = t[t["true_role"] == "referee"]
    return {
        "team_accuracy": round(float(max(same, swapped)), 3),
        "tracks_evaluated": int(len(on_team)),
        "referee_tracks_without_team": f"{int((refs['team_id'] == -1).sum())}/{len(refs)}",
        "label_mapping": "same" if same >= swapped else "swapped",
    }


def error_summary(err: np.ndarray) -> dict[str, float]:
    """Mean, median and 95th percentile of an error array (NaNs ignored)."""
    err = err[np.isfinite(err)]
    if len(err) == 0:
        return {"n": 0}
    return {"n": int(len(err)), "mean_m": round(float(err.mean()), 3),
            "median_m": round(float(np.median(err)), 3), "p95_m": round(float(np.percentile(err, 95)), 3)}


def position_metrics(
    positions: pd.DataFrame, ball_positions: pd.DataFrame, track_agent: pd.Series,
    gt_positions: pd.DataFrame, gt_ball: pd.DataFrame,
) -> dict[str, Any]:
    """Distance in metres between projected and true positions of people and the ball."""
    p = positions.assign(agent_id=positions["tracker_id"].map(track_agent)).dropna(subset=["agent_id", "x_m"])
    p = p.merge(gt_positions, on=["frame", "agent_id"], suffixes=("", "_true"))
    people = np.hypot(p["x_m"] - p["x_m_true"], p["y_m"] - p["y_m_true"]).to_numpy()
    b = ball_positions.merge(gt_ball, on="frame", suffixes=("", "_true")).dropna(subset=["x_m"])
    ball = np.hypot(b["x_m"] - b["x_m_true"], b["y_m"] - b["y_m_true"]).to_numpy()
    return {"people": error_summary(people), "ball": error_summary(ball),
            "people_coverage": round(len(p) / max(len(positions), 1), 3)}


def evaluate(run: Path, truth: Path) -> dict[str, Any]:
    """Compute every metric whose outputs exist in ``run``."""
    gt = truth / "ground_truth"
    gt_boxes = pd.read_csv(gt / "boxes.csv")
    agents = pd.read_csv(gt / "agents.csv")
    tracks = pd.read_csv(run / "tracks.csv")
    track_agent = match_tracks_to_agents(tracks, gt_boxes)
    report: dict[str, Any] = {"tracking": tracking_metrics(tracks, track_agent, gt_boxes)}
    if (run / "teams.csv").exists():
        report["teams"] = team_metrics(pd.read_csv(run / "teams.csv"), track_agent, agents)
    if (run / "positions.csv").exists():
        report["pitch_positions"] = position_metrics(
            pd.read_csv(run / "positions.csv"), pd.read_csv(run / "ball_positions.csv"), track_agent,
            pd.read_csv(gt / "positions.csv"), pd.read_csv(gt / "ball.csv"))
    return report


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Evaluate a pipeline run on a synthetic match.")
    p.add_argument("--run", type=Path, default=Path("outputs/synthetic_match"))
    p.add_argument("--truth", type=Path, default=Path("data/synthetic/synthetic_match"))
    args = p.parse_args(argv)
    report = evaluate(args.run, args.truth)
    print(json.dumps(report, indent=2))
    (args.run / "evaluation.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
