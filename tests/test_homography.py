"""Tests for the pitch template, homography fitting, smoothing and projection."""

import cv2
import numpy as np
import pandas as pd

from src.pitch.homography import apply_homography, fit_homography, reprojection_error, smooth_homographies
from src.pitch.mapping import frame_homography, project_tracks
from src.pitch.template import PitchTemplate
from src.utils.config import load_config

TEMPLATE = PitchTemplate()
SIZE = (1280, 720)


def camera(cx: float = 52.5, zoom: float = 1.0) -> np.ndarray:
    """Pitch-to-image homography of a broadcast-style camera centred on x = cx."""
    far, near = 32.0 / zoom, 21.0 / zoom
    world = np.float32([[cx - far, -1], [cx + far, -1], [cx + near, 71], [cx - near, 71]])
    image = np.float32([[0, 86], [1280, 86], [1280, 720], [0, 720]])
    return cv2.getPerspectiveTransform(world, image)


def visible_keypoints(pitch_to_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    img = apply_homography(pitch_to_image, TEMPLATE.vertices)
    inside = (img[:, 0] >= 0) & (img[:, 0] < SIZE[0]) & (img[:, 1] >= 0) & (img[:, 1] < SIZE[1])
    return img[inside], TEMPLATE.vertices[inside]


def test_template_geometry() -> None:
    v = TEMPLATE.vertices
    assert v.shape == (32, 2)
    np.testing.assert_allclose(v[0], [0, 0])
    np.testing.assert_allclose(v[29], [105, 68])
    np.testing.assert_allclose(v[8], [11, 34])                      # penalty spot
    np.testing.assert_allclose(v[[30, 31]], [[52.5 - 9.15, 34], [52.5 + 9.15, 34]])
    mirrored = np.column_stack([105 - v[:, 0], v[:, 1]])            # pitch is symmetric
    assert all(np.isclose(mirrored, p).all(axis=1).any() for p in v)
    assert all(1 <= a <= 32 and 1 <= b <= 32 for a, b in TEMPLATE.edges)
    assert TEMPLATE.contains(np.array([[-1, 10], [50, 30], [110, 30]]), margin=2).tolist() == [True, True, False]


def test_fit_recovers_exact_homography() -> None:
    truth = camera(cx=40)
    img, pitch = visible_keypoints(truth)
    assert len(img) >= 4
    h, inliers = fit_homography(img, pitch)
    assert inliers.all()
    probe_pitch = np.array([[30.0, 20.0], [45.0, 50.0], [60.0, 10.0]])
    probe_img = apply_homography(truth, probe_pitch)
    np.testing.assert_allclose(apply_homography(h, probe_img), probe_pitch, atol=1e-6)


def test_fit_is_robust_to_noise_and_outliers() -> None:
    rng = np.random.default_rng(0)
    img, pitch = visible_keypoints(camera(cx=52.5))
    noisy = img + rng.normal(0, 1.5, img.shape)
    noisy[0] += [150, -80]                                          # two badly placed keypoints
    noisy[1] += [-120, 90]
    h, inliers = fit_homography(noisy, pitch, ransac_threshold=1.0)
    assert not inliers[0] and not inliers[1]
    assert reprojection_error(h, noisy[inliers], pitch[inliers]) < 0.3


def test_fit_needs_four_points() -> None:
    img, pitch = visible_keypoints(camera())
    assert fit_homography(img[:3], pitch[:3]) is None


def test_frame_homography_ignores_low_confidence_keypoints() -> None:
    img, _ = visible_keypoints(camera())
    xy = apply_homography(camera(), TEMPLATE.vertices)
    conf = np.zeros(32)
    conf[:3] = 0.9                                                  # only 3 confident points
    h, _, _ = frame_homography(xy, conf, TEMPLATE, load_config())
    assert h is None


def _grid_error(h_est: np.ndarray, pitch_to_image: np.ndarray) -> float:
    probe = np.array([[x, y] for x in (35.0, 52.5, 70.0) for y in (15.0, 34.0, 55.0)])
    return float(np.linalg.norm(apply_homography(h_est, apply_homography(pitch_to_image, probe)) - probe, axis=1).mean())


def test_smoothing_reduces_jitter_fills_gaps_and_rejects_glitches() -> None:
    rng = np.random.default_rng(1)
    n = 60
    cams = [camera(cx=40 + 0.2 * f) for f in range(n)]              # slow pan
    raw = []
    for f, cam in enumerate(cams):
        img, pitch = visible_keypoints(cam)
        h, _ = fit_homography(img + rng.normal(0, 2.0, img.shape), pitch)
        raw.append(h)
    raw[20:25] = [None] * 5                                         # short gap
    raw[40] = cv2.getPerspectiveTransform(                          # one wildly wrong frame
        np.float32([[0, 0], [1280, 0], [1280, 720], [0, 720]]),
        np.float32([[0, 0], [10, 0], [10, 10], [0, 10]]))
    smoothed = smooth_homographies(raw, SIZE, window=7, max_gap=10)

    assert all(h is not None for h in smoothed)
    raw_err = np.mean([_grid_error(raw[f], cams[f]) for f in range(5, 18)])
    smooth_err = np.mean([_grid_error(smoothed[f], cams[f]) for f in range(5, 18)])
    assert smooth_err < raw_err
    assert _grid_error(smoothed[22], cams[22]) < 0.5                # gap interpolated
    assert _grid_error(smoothed[40], cams[40]) < 0.5                # glitch removed by the median


def test_smoothing_leaves_long_gaps_empty() -> None:
    img, pitch = visible_keypoints(camera())
    h, _ = fit_homography(img, pitch)
    raw = [h] * 5 + [None] * 30 + [h] * 5
    smoothed = smooth_homographies(raw, SIZE, window=3, max_gap=10)
    assert smoothed[20] is None and smoothed[2] is not None and smoothed[-1] is not None


def test_project_tracks_uses_feet_and_drops_off_pitch_points() -> None:
    cam = camera()
    h = np.linalg.inv(cam)
    feet = apply_homography(cam, np.array([[50.0, 30.0], [52.5, 75.0]]))   # second is 7 m off the pitch
    tracks = pd.DataFrame({
        "frame": [0, 0], "tracker_id": [1, 2], "class_name": ["player", "player"],
        "x1": feet[:, 0] - 10, "x2": feet[:, 0] + 10, "y1": feet[:, 1] - 60, "y2": feet[:, 1],
    })
    out = project_tracks(tracks, None, [h], TEMPLATE, margin=5.0)
    np.testing.assert_allclose(out.loc[0, ["x_m", "y_m"]].to_numpy(float), [50.0, 30.0], atol=0.01)
    assert out.loc[1, ["x_m", "y_m"]].isna().all()
