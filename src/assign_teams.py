"""Stage 4 CLI: assign each tracked person to a team by shirt colour.

Reuses ``outputs/<clip>/tracks.csv`` when it exists, otherwise runs stages 2-3 first.

Example:
    python -m src.assign_teams --input data/raw/0bfacc_0.mp4
    # -> outputs/0bfacc_0/teams.csv, team_colours.json, team_crops.jpg
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.team_assignment.assign import assign_teams, extract_colours, sample_track_frames
from src.team_assignment.classifier import TeamClassifier
from src.team_assignment.features import shirt_crop
from src.utils.cli import RunContext, base_parser, make_context
from src.utils.progress import ProgressCallback
from src.utils.video import read_frames

log = logging.getLogger(__name__)


def run_team_assignment(
    ctx: RunContext, tracks: pd.DataFrame, callback: ProgressCallback | None = None
) -> tuple[pd.DataFrame, TeamClassifier]:
    """Sample shirt crops, cluster them and write ``teams.csv`` + ``team_colours.json``."""
    samples = sample_track_frames(tracks, ctx.cfg["teams"]["crops_per_track"])
    colours = extract_colours(ctx.input_path, samples, ctx.cfg, callback)
    teams, clf = assign_teams(tracks, colours, ctx.cfg)
    teams.to_csv(ctx.paths.teams, index=False)
    save_team_colours(ctx.paths.team_colours, clf)
    log.info("Teams: %s", teams.groupby(["role", "team_id"]).size().to_dict())
    return teams, clf


def save_team_colours(path: Path, clf: TeamClassifier) -> None:
    """Write team colours (BGR and hex) to JSON."""
    colours = {
        str(i): {"bgr": list(bgr), "hex": "#{:02X}{:02X}{:02X}".format(*bgr[::-1])}
        for i, bgr in enumerate(clf.colours_bgr())
    }
    path.write_text(json.dumps(colours, indent=2))


def load_team_colours(path: Path) -> dict[int, tuple[int, int, int]]:
    """Read team colours written by ``save_team_colours`` as ``{team_id: BGR}``."""
    data = json.loads(Path(path).read_text())
    return {int(k): tuple(v["bgr"]) for k, v in data.items()}


def crop_sheet(video: Path, tracks: pd.DataFrame, teams: pd.DataFrame, per_group: int = 12,
               size: tuple[int, int] = (40, 80)) -> np.ndarray:
    """One row of person crops per (role, team), to check the clustering by eye."""
    groups = teams.groupby(["role", "team_id"])["tracker_id"].apply(list)
    wanted: dict[int, list[tuple[int, np.ndarray]]] = {}
    for row_idx, ids in enumerate(groups):
        picks = tracks[tracks["tracker_id"].isin(ids)].sample(
            min(per_group, tracks["tracker_id"].isin(ids).sum()), random_state=0)
        for r in picks.itertuples():
            wanted.setdefault(int(r.frame), []).append((row_idx, np.array([r.x1, r.y1, r.x2, r.y2])))
    rows: list[list[np.ndarray]] = [[] for _ in groups]
    last = max(wanted) + 1 if wanted else 0
    for f, frame in enumerate(read_frames(video, last)):
        for row_idx, box in wanted.get(f, []):
            crop = shirt_crop(frame, box, 0.0, 1.0, 0.0, min_px=2)
            if crop is not None:
                rows[row_idx].append(cv2.resize(crop, size))
    label_w = 170
    sheet = []
    for (role, team), crops in zip(groups.index, rows):
        strip = np.full((size[1], label_w + per_group * size[0], 3), 255, np.uint8)
        cv2.putText(strip, f"{role} team={team}", (5, size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        for k, c in enumerate(crops[:per_group]):
            strip[:, label_w + k * size[0]: label_w + (k + 1) * size[0]] = c
        sheet.append(strip)
    return np.vstack(sheet) if sheet else np.zeros((10, 10, 3), np.uint8)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = base_parser("Assign tracked people to teams by shirt colour.").parse_args(argv)
    ctx = make_context(args)
    if ctx.paths.tracks.exists():
        tracks = pd.read_csv(ctx.paths.tracks)
    else:
        from src.track import clip_length, load_or_detect, track_clip

        n_frames, fps = clip_length(ctx.input_path, ctx.max_frames)
        tracks, ball = track_clip(load_or_detect(ctx), n_frames, fps, ctx.cfg)
        tracks.to_csv(ctx.paths.tracks, index=False)
        ball.to_csv(ctx.paths.ball, index=False)
    teams, _ = run_team_assignment(ctx, tracks)
    sheet_path = ctx.paths.root / "team_crops.jpg"
    cv2.imwrite(str(sheet_path), crop_sheet(ctx.input_path, tracks, teams))
    print(f"Teams CSV:   {ctx.paths.teams}\nCrop sheet:  {sheet_path}")


if __name__ == "__main__":
    main()
