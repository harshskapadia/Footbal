"""Render a simulated match as broadcast-style video with a panning, zooming camera."""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np

from src.pitch.draw import PitchCanvas, draw_pitch
from src.pitch.template import PitchTemplate
from src.simulation.match import Agent, MatchSimulation

# BGR kit colours: (shirt, shorts)
KITS: dict[tuple[int, str], tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    (0, "player"): ((40, 40, 200), (240, 240, 240)),       # red shirt, white shorts
    (1, "player"): ((221, 171, 108), (90, 40, 20)),        # sky-blue shirt, navy shorts
    (0, "goalkeeper"): ((0, 220, 230), (20, 20, 20)),      # yellow
    (1, "goalkeeper"): ((0, 140, 255), (20, 20, 20)),      # orange
    (-1, "referee"): ((25, 25, 25), (25, 25, 25)),         # black
}
SKIN = (120, 160, 210)
SOCKS = (30, 30, 30)
STANDS = (70, 72, 80)


def apply_h(h: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to ``(N, 2)`` points."""
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, h).reshape(-1, 2)


def camera_homographies(
    ball: np.ndarray, fps: float, image_size: tuple[int, int] = (1280, 720),
    template: PitchTemplate = PitchTemplate(),
) -> np.ndarray:
    """Pitch-to-image homographies for a camera on the near touchline following the ball.

    The camera pans smoothly towards the ball and slowly zooms in and out, so the
    homography changes every frame, as in real broadcast footage.
    """
    w, h = image_size
    image_quad = np.array([[0, 0.12 * h], [w, 0.12 * h], [w, h], [0, h]], dtype=np.float32)
    out = np.zeros((len(ball), 3, 3))
    cx = float(np.clip(ball[0, 0], 30, template.length - 30))
    for f, (bx, _) in enumerate(ball):
        cx += 0.06 * (np.clip(bx, 30, template.length - 30) - cx)
        zoom = 1.0 + 0.12 * np.sin(2 * np.pi * f / (fps * 8))
        far, near = 32.0 / zoom, 21.0 / zoom
        world_quad = np.array([[cx - far, -1], [cx + far, -1], [cx + near, template.width + 3],
                               [cx - near, template.width + 3]], dtype=np.float32)
        out[f] = cv2.getPerspectiveTransform(world_quad, image_quad)
    return out


def person_box(h: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Image box ``[x1, y1, x2, y2]`` of a 1.8 m person standing at ``xy`` and local px/m."""
    foot = apply_h(h, xy[None])[0]
    side = apply_h(h, np.array([xy + [0.5, 0], xy - [0.5, 0]]))
    ppm = float(np.linalg.norm(side[0] - side[1]))
    height, width = 1.8 * ppm, 0.55 * ppm
    return np.array([foot[0] - width / 2, foot[1] - height, foot[0] + width / 2, foot[1]]), ppm


def draw_person(img: np.ndarray, box: np.ndarray, agent: Agent) -> None:
    """Draw a simple figure (legs, shorts, shirt, head) inside ``box``."""
    x1, y1, x2, y2 = box
    hgt, cx = y2 - y1, (x1 + x2) / 2
    shirt, shorts = KITS[(agent.team, agent.role)]

    def rect(top: float, bottom: float, half_w: float, colour: tuple[int, int, int]) -> None:
        cv2.rectangle(img, (round(cx - half_w), round(y1 + top * hgt)),
                      (round(cx + half_w), round(y1 + bottom * hgt)), colour, -1)

    half = (x2 - x1) / 2
    rect(0.62, 1.0, half * 0.55, SOCKS)        # legs
    rect(0.48, 0.64, half * 0.75, shorts)      # shorts
    rect(0.16, 0.50, half, shirt)              # shirt
    cv2.circle(img, (round(cx), round(y1 + 0.08 * hgt)), max(1, round(0.08 * hgt)), SKIN, -1, cv2.LINE_AA)


def render_frame(
    background: np.ndarray, h: np.ndarray, positions: np.ndarray, agents: list[Agent],
    ball_xy: np.ndarray, image_size: tuple[int, int], canvas: PitchCanvas,
) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray]:
    """Render one frame.

    Returns:
        The BGR frame, ``{agent_id: box}`` for every agent, and the ball box.
    """
    img_to_canvas = h @ np.linalg.inv(canvas.metres_to_pixels_matrix())
    frame = cv2.warpPerspective(background, img_to_canvas, image_size, borderValue=STANDS)
    boxes = {a.agent_id: person_box(h, positions[a.agent_id])[0] for a in agents}
    for a in sorted(agents, key=lambda a: boxes[a.agent_id][3]):   # far to near
        draw_person(frame, boxes[a.agent_id], a)
    centre = apply_h(h, ball_xy[None])[0]
    _, ppm = person_box(h, ball_xy)
    r = max(2.0, 0.15 * ppm)
    cv2.circle(frame, (round(centre[0]), round(centre[1] - r)), round(r), (250, 250, 250), -1, cv2.LINE_AA)
    ball_box = np.array([centre[0] - r, centre[1] - 2 * r, centre[0] + r, centre[1]])
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    return frame, boxes, ball_box


def render_match(
    sim: MatchSimulation, homographies: np.ndarray, image_size: tuple[int, int] = (1280, 720),
    template: PitchTemplate = PitchTemplate(),
) -> Iterator[tuple[np.ndarray, dict[int, np.ndarray], np.ndarray]]:
    """Yield ``(frame, boxes, ball_box)`` for every simulated frame."""
    canvas = PitchCanvas(template, scale=12.0, padding=10.0)
    background = draw_pitch(canvas, line_px=3)
    for f in range(len(sim.ball)):
        yield render_frame(background, homographies[f], sim.positions[f], sim.agents, sim.ball[f],
                           image_size, canvas)
