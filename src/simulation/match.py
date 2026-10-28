"""Simulate player and ball movement on a 105 x 68 m pitch."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.pitch.template import PitchTemplate

# 4-4-2 for a team defending x = 0, as (x, y) metres. Index 0 is the goalkeeper.
FORMATION = np.array([
    (5, 34), (22, 10), (20, 26), (20, 42), (22, 58),
    (38, 10), (36, 26), (36, 42), (38, 58), (48, 28), (48, 40),
], dtype=float)


@dataclass(frozen=True)
class Agent:
    """One person on the pitch."""

    agent_id: int
    team: int        # 0 or 1; -1 for the referee
    role: str        # player | goalkeeper | referee
    base: tuple[float, float]


@dataclass
class MatchSimulation:
    """Ground-truth trajectories of a simulated passage of play."""

    agents: list[Agent]
    positions: np.ndarray          # (frames, agents, 2) metres
    ball: np.ndarray               # (frames, 2) metres
    holder: np.ndarray             # (frames,) agent id in possession, -1 while the ball travels
    passes: list[dict] = field(default_factory=list)
    fps: float = 25.0


def make_agents(template: PitchTemplate) -> list[Agent]:
    """Two mirrored 4-4-2 teams plus a referee."""
    agents: list[Agent] = []
    for team in (0, 1):
        for k, (x, y) in enumerate(FORMATION):
            bx = x if team == 0 else template.length - x
            role = "goalkeeper" if k == 0 else "player"
            agents.append(Agent(len(agents), team, role, (bx, float(y))))
    agents.append(Agent(len(agents), -1, "referee", (template.length / 2, template.width / 2)))
    return agents


def _choose_receiver(
    rng: np.random.Generator, agents: list[Agent], pos: np.ndarray, holder: int, same_team_p: float
) -> int | None:
    """Pick a pass target 8-35 m away: usually a teammate, sometimes an opponent (turnover)."""
    team = agents[holder].team
    want = team if rng.random() < same_team_p else 1 - team
    cands = [a.agent_id for a in agents if a.team == want and a.agent_id != holder and a.role == "player"]
    d = np.linalg.norm(pos[cands] - pos[holder], axis=1)
    ok = [(c, dist) for c, dist in zip(cands, d, strict=True) if 8 <= dist <= 35]
    if not ok:
        return None
    ok.sort(key=lambda t: t[1])
    return int(rng.choice([c for c, _ in ok[:4]]))


def simulate_match(
    n_frames: int = 250, fps: float = 25.0, seed: int = 0, template: PitchTemplate = PitchTemplate()
) -> MatchSimulation:
    """Simulate ``n_frames`` of play: formations shift with the ball, players dribble and pass.

    Args:
        n_frames: Number of frames to simulate.
        fps: Frames per second.
        seed: Random seed (results are deterministic for a given seed).
        template: Pitch dimensions.
    """
    rng = np.random.default_rng(seed)
    agents = make_agents(template)
    n = len(agents)
    dt = 1.0 / fps
    base = np.array([a.base for a in agents])
    pos = base.copy()
    vel = np.zeros_like(pos)
    phase = rng.uniform(0, 2 * np.pi, size=(n, 2))
    freq = rng.uniform(0.05, 0.15, size=(n, 2))
    attack_dir = np.array([1.0 if a.team == 0 else -1.0 for a in agents])
    max_speed = np.array([4.5 if a.role == "goalkeeper" else 7.5 for a in agents])
    centre = np.array([template.length / 2, template.width / 2])

    holder = 9                     # a team 0 forward kicks off
    hold_left = int(rng.integers(15, 40))
    flight: dict | None = None
    ball = pos[holder].copy()

    positions = np.zeros((n_frames, n, 2))
    balls = np.zeros((n_frames, 2))
    holders = np.full(n_frames, -1, dtype=int)
    passes: list[dict] = []

    for f in range(n_frames):
        t = f * dt
        # Team shape follows the ball; each player also wanders a little.
        shift = np.column_stack([
            np.where([a.role == "goalkeeper" for a in agents], 0.05, 0.35) * (ball[0] - centre[0]),
            np.where([a.role == "goalkeeper" for a in agents], 0.15, 0.25) * (ball[1] - centre[1]),
        ])
        wander = 3.0 * np.sin(2 * np.pi * freq * t + phase)
        target = base + shift + wander
        ref = agents[-1].agent_id
        target[ref] = ball + np.array([-8.0, 10.0 if ball[1] < centre[1] else -10.0])
        if holder >= 0:
            target[holder] = pos[holder] + np.array([attack_dir[holder] * 4.0, 0.0])
        if flight is not None:
            target[flight["to"]] = flight["end"]

        desired = target - pos
        speed = np.linalg.norm(desired, axis=1, keepdims=True)
        desired = desired / np.maximum(speed, 1e-9) * np.minimum(speed * 1.5, max_speed[:, None])
        vel = 0.85 * vel + 0.15 * desired
        pos = pos + vel * dt
        pos[:, 0] = np.clip(pos[:, 0], 0.5, template.length - 0.5)
        pos[:, 1] = np.clip(pos[:, 1], 0.5, template.width - 0.5)

        # Ball: carried by the holder, or travelling along a pass.
        if flight is not None:
            k = (f - flight["start"]) / flight["frames"]
            ball = flight["origin"] + k * (flight["end"] - flight["origin"])
            if f - flight["start"] >= flight["frames"]:
                holder = flight["to"]
                hold_left = int(rng.integers(15, 40))
                passes[-1]["frame_end"] = f
                flight = None
        if flight is None and holder >= 0:
            ball = pos[holder] + np.array([attack_dir[holder] * 0.6, 0.0])
            hold_left -= 1
            if hold_left <= 0:
                to = _choose_receiver(rng, agents, pos, holder, same_team_p=0.8)
                if to is None:
                    hold_left = 5
                else:
                    end = pos[to] + vel[to] * 0.5
                    frames = max(4, int(np.ceil(np.linalg.norm(end - ball) / 16.0 * fps)))
                    flight = {"start": f, "frames": frames, "origin": ball.copy(), "end": end, "to": to}
                    passes.append({"frame_start": f, "frame_end": -1, "from_id": holder, "to_id": to,
                                   "from_team": agents[holder].team, "to_team": agents[to].team})
                    holder = -1

        positions[f] = pos
        balls[f] = ball
        holders[f] = holder

    passes = [p for p in passes if p["frame_end"] >= 0]
    return MatchSimulation(agents, positions, balls, holders, passes, fps)
