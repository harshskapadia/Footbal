"""File layout of one pipeline run: every stage reads and writes ``outputs/<clip>/``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    """Paths of all intermediate and final files for one input clip."""

    root: Path

    @classmethod
    def for_clip(cls, output_dir: Path, clip: Path) -> RunPaths:
        """Create ``<output_dir>/<clip stem>/`` and return its paths."""
        root = Path(output_dir) / Path(clip).stem
        root.mkdir(parents=True, exist_ok=True)
        return cls(root)

    # Stage 2: detection
    @property
    def detections(self) -> Path:
        return self.root / "detections.csv"

    @property
    def detections_video(self) -> Path:
        return self.root / "detections.mp4"

    # Stage 3: tracking
    @property
    def tracks(self) -> Path:
        return self.root / "tracks.csv"

    @property
    def ball(self) -> Path:
        return self.root / "ball.csv"

    @property
    def tracks_video(self) -> Path:
        return self.root / "tracks.mp4"

    # Stage 4: teams
    @property
    def teams(self) -> Path:
        return self.root / "teams.csv"

    @property
    def team_colours(self) -> Path:
        return self.root / "team_colours.json"

    # Stage 5: pitch mapping
    @property
    def keypoints(self) -> Path:
        return self.root / "keypoints.csv"

    @property
    def homographies(self) -> Path:
        return self.root / "homographies.csv"

    @property
    def positions(self) -> Path:
        return self.root / "positions.csv"

    @property
    def ball_positions(self) -> Path:
        return self.root / "ball_positions.csv"

    # Stage 6-7: analytics and output
    @property
    def possession(self) -> Path:
        return self.root / "possession.csv"

    @property
    def player_stats(self) -> Path:
        return self.root / "player_stats.csv"

    @property
    def passes(self) -> Path:
        return self.root / "passes.csv"

    @property
    def stats_json(self) -> Path:
        return self.root / "stats.json"

    def heatmap(self, team_id: int) -> Path:
        return self.root / f"heatmap_team{team_id}.png"

    def passing_network(self, team_id: int) -> Path:
        return self.root / f"passing_network_team{team_id}.png"

    @property
    def annotated_video(self) -> Path:
        return self.root / "annotated.mp4"

    @property
    def web_video(self) -> Path:
        return self.root / "annotated_h264.mp4"
