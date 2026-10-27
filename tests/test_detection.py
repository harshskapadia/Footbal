"""Tests for class mapping and detection post-processing (no model weights needed)."""

import numpy as np
import supervision as sv

from src.detect import detect_frame, detections_to_rows
from src.detection.classes import BALL, GOALKEEPER, PLAYER, REFEREE, build_id_map, remap_detections
from src.detection.detector import keep_best_ball
from src.utils.config import load_config

COCO_NAMES = {0: "person", 1: "bicycle", 32: "sports ball"}
FOOTBALL_NAMES = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}


def make_detections(class_ids: list[int], confs: list[float]) -> sv.Detections:
    n = len(class_ids)
    xyxy = np.array([[10 * i, 10 * i, 10 * i + 5, 10 * i + 5] for i in range(n)], dtype=float).reshape(-1, 4)
    return sv.Detections(xyxy=xyxy, confidence=np.array(confs, dtype=float), class_id=np.array(class_ids, dtype=int))


def test_build_id_map_football_and_coco() -> None:
    class_map = load_config()["detection"]["class_map"]
    assert build_id_map(FOOTBALL_NAMES, class_map) == {0: BALL, 1: GOALKEEPER, 2: PLAYER, 3: REFEREE}
    assert build_id_map(COCO_NAMES, class_map) == {0: PLAYER, 32: BALL}


def test_remap_drops_unmapped_classes() -> None:
    dets = make_detections([0, 1, 32], [0.9, 0.8, 0.7])
    out = remap_detections(dets, {0: PLAYER, 32: BALL})
    assert out.class_id.tolist() == [PLAYER, BALL]
    assert out.data["class_name"].tolist() == ["player", "ball"]


def test_keep_best_ball_keeps_one_ball_and_all_people() -> None:
    dets = make_detections([BALL, PLAYER, BALL, REFEREE], [0.3, 0.9, 0.6, 0.8])
    out = keep_best_ball(dets)
    assert sorted(out.class_id.tolist()) == sorted([BALL, PLAYER, REFEREE])
    assert out.confidence[out.class_id == BALL].tolist() == [0.6]


def test_detect_frame_prefers_sliced_ball_but_falls_back() -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    main = lambda _: make_detections([PLAYER, BALL], [0.9, 0.4])  # noqa: E731
    sliced = lambda _: make_detections([BALL], [0.7])  # noqa: E731
    empty = lambda _: make_detections([], [])  # noqa: E731

    out = detect_frame(frame, main, sliced)
    assert out.confidence[out.class_id == BALL].tolist() == [0.7]
    out = detect_frame(frame, main, empty)
    assert out.confidence[out.class_id == BALL].tolist() == [0.4]


def test_detections_to_rows() -> None:
    rows = detections_to_rows(5, make_detections([PLAYER], [0.87654]))
    assert rows == [{"frame": 5, "class_id": PLAYER, "class_name": "player", "confidence": 0.8765,
                     "x1": 0.0, "y1": 0.0, "x2": 5.0, "y2": 5.0}]


class _FakeModel:
    """Stands in for a YOLO model: only ``names`` is read by the detector."""

    names = {0: "person", 32: "sports ball"}


def test_sliced_ball_detector_maps_tiles_to_frame(monkeypatch) -> None:
    from src.detection import slicing

    def fake_predict(model, tile, conf, iou, imgsz, device):
        # "Detect" a ball wherever the tile has a white pixel, in tile coordinates.
        ys, xs = np.nonzero(tile[..., 0] == 255)
        if len(xs) == 0:
            return make_detections([], [])
        box = np.array([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], dtype=float)
        return sv.Detections(xyxy=box, confidence=np.array([0.9]), class_id=np.array([32]))

    monkeypatch.setattr(slicing, "predict", fake_predict)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[700:710, 1500:1510] = 255  # 10 px ball far from the top-left tile
    detector = slicing.SlicedBallDetector(_FakeModel(), {"sports ball": "ball"})

    out = detector(frame)
    assert out.class_id.tolist() == [BALL]
    np.testing.assert_allclose(out.xyxy[0], [1500, 700, 1510, 710])
