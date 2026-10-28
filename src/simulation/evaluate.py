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


def possession_metrics(possession: pd.DataFrame, gt_ball: pd.DataFrame, agents: pd.DataFrame) -> dict[str, Any]:
    """Possession share and frame-level agreement with the true holder's team."""
    team_of = agents.set_index("agent_id")["team"]
    true_team = gt_ball["holder"].map(team_of).where(gt_ball["holder"] >= 0).ffill()
    both = possession.assign(true_team=true_team.to_numpy()[: len(possession)])
    both = both[(both["team_in_possession"] >= 0) & both["true_team"].notna()]
    true_share = true_team.dropna().value_counts(normalize=True) * 100
    est_share = possession.loc[possession["team_in_possession"] >= 0, "team_in_possession"] \
        .value_counts(normalize=True) * 100
    return {
        "true_pct": {int(t): round(float(true_share.get(t, 0)), 1) for t in (0, 1)},
        "estimated_pct": {int(t): round(float(est_share.get(t, 0)), 1) for t in (0, 1)},
        "frame_agreement": round(float((both["team_in_possession"] == both["true_team"]).mean()), 3),
    }


def pass_metrics(passes: pd.DataFrame, gt_passes: pd.DataFrame, track_agent: pd.Series,
                 tolerance: int = 15) -> dict[str, Any]:
    """Match detected passes to true ones (same passer and receiver, reception within ``tolerance`` frames)."""
    det = passes.assign(from_agent=passes["from_id"].map(track_agent), to_agent=passes["to_id"].map(track_agent))
    matched_gt: set[int] = set()
    tp = 0
    for d in det.itertuples():
        for g in gt_passes.itertuples():
            if g.Index in matched_gt:
                continue
            if d.from_agent == g.from_id and d.to_agent == g.to_id and abs(d.frame_end - g.frame_end) <= tolerance:
                matched_gt.add(g.Index)
                tp += 1
                break
    return {"true_passes": int(len(gt_passes)), "detected": int(len(passes)), "matched": tp,
            "precision": round(tp / len(passes), 3) if len(passes) else 0.0,
            "recall": round(tp / len(gt_passes), 3) if len(gt_passes) else 0.0}


def distance_metrics(stats: pd.DataFrame, positions: pd.DataFrame, track_agent: pd.Series,
                     gt_positions: pd.DataFrame) -> dict[str, Any]:
    """Per-track distance covered vs the true distance over the same frames."""
    gt = gt_positions.set_index(["agent_id", "frame"]).sort_index()
    errors = []
    for row in stats.itertuples():
        agent = track_agent.get(row.tracker_id)
        if agent is None or row.distance_m < 20:
            continue
        frames = np.sort(positions.loc[positions["tracker_id"] == row.tracker_id, "frame"].to_numpy())
        steps = [np.linalg.norm(gt.loc[(agent, b)].to_numpy() - gt.loc[(agent, a)].to_numpy())
                 for a, b in zip(frames[:-1], frames[1:]) if b == a + 1]
        true_d = float(np.sum(steps))
        if true_d > 0:
            errors.append(100 * (row.distance_m - true_d) / true_d)
    e = np.array(errors)
    return {"tracks": int(len(e)), "mean_error_pct": round(float(e.mean()), 1) if len(e) else None,
            "mean_abs_error_pct": round(float(np.abs(e).mean()), 1) if len(e) else None}


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
    if (run / "possession.csv").exists():
        report["possession"] = possession_metrics(pd.read_csv(run / "possession.csv"),
                                                  pd.read_csv(gt / "ball.csv"), agents)
        report["passes"] = pass_metrics(pd.read_csv(run / "passes.csv"), pd.read_csv(gt / "passes.csv"),
                                        track_agent)
    if (run / "player_stats.csv").exists():
        report["distance"] = distance_metrics(pd.read_csv(run / "player_stats.csv"),
                                              pd.read_csv(run / "positions.csv"), track_agent,
                                              pd.read_csv(gt / "positions.csv"))
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
