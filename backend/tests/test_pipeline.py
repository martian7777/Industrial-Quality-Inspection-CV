"""Smoke + behaviour tests for the inspection pipeline."""

import numpy as np
import pytest

from app.camera import CameraSimulator, Frame
from app.config import InspectionConfig
from app.inference import InspectionEngine, _iou, Detection


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #
def test_camera_clean_frame_shape():
    cam = CameraSimulator(defect_rate=0.0)
    frame = cam.grab_frame()
    assert isinstance(frame, Frame)
    assert frame.image.shape == (480, 640, 3)
    assert frame.image.dtype == np.uint8
    assert frame.ground_truth == []  # defect_rate 0 -> no defects


def test_camera_injects_defects():
    cam = CameraSimulator(defect_rate=1.0)
    # Over several frames at least one defect type appears.
    seen = set()
    for _ in range(20):
        f = cam.grab_frame()
        for gt in f.ground_truth:
            seen.add(gt.label)
    assert seen.issubset(set(CameraSimulator.DEFECT_TYPES))
    assert len(seen) > 0


def test_unit_ids_unique():
    cam = CameraSimulator()
    ids = {cam.grab_frame().unit_id for _ in range(10)}
    assert len(ids) == 10


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def test_clean_unit_passes():
    cam = CameraSimulator(defect_rate=0.0)
    engine = InspectionEngine()
    cfg = InspectionConfig()
    passes = 0
    for _ in range(15):
        result = engine.inspect(cam.grab_frame(), cfg)
        if result.passed:
            passes += 1
    # A clean part should overwhelmingly pass. Allow rare spurious edges.
    assert passes >= 12


def test_defective_unit_detected():
    cam = CameraSimulator(defect_rate=1.0)
    engine = InspectionEngine()
    cfg = InspectionConfig()
    rejected = 0
    for _ in range(25):
        result = engine.inspect(cam.grab_frame(), cfg)
        if not result.passed:
            rejected += 1
    # Most defective parts should be caught by the hybrid pipeline.
    assert rejected >= 15


def test_annotation_output():
    cam = CameraSimulator(defect_rate=1.0)
    engine = InspectionEngine()
    result = engine.inspect(cam.grab_frame(), InspectionConfig())
    assert result.annotated.shape == (480, 640, 3)
    assert 0.0 <= result.confidence <= 1.0
    assert result.cycle_time_ms >= 0.0


def test_opencv_only_model():
    cam = CameraSimulator(defect_rate=1.0)
    engine = InspectionEngine()
    cfg = InspectionConfig(active_model="opencv_only")
    result = engine.inspect(cam.grab_frame(), cfg)
    # All detections must come from OpenCV when YOLO is disabled.
    assert all(d["source"] == "opencv" for d in result.detections)


def test_iou_geometry():
    a = Detection(0, 0, 10, 10, "scratch", 0.9, "opencv")
    b = Detection(0, 0, 10, 10, "scratch", 0.8, "yolo")
    assert _iou(a, b) == pytest.approx(1.0)
    c = Detection(100, 100, 10, 10, "scratch", 0.8, "yolo")
    assert _iou(a, c) == 0.0
