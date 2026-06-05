"""Tests for persistence, stats aggregation and config validation."""

import pytest
from pydantic import ValidationError

from app import database as db
from app.config import InspectionConfig


def test_save_and_query_inspection():
    db.init_db()
    rid = db.save_inspection(
        unit_id="TEST-0001",
        passed=False,
        defects=[{"x": 1, "y": 2, "w": 3, "h": 4, "label": "scratch",
                  "confidence": 0.8, "source": "opencv"}],
        confidence=0.8,
        cycle_time_ms=5.5,
        image_path=None,
    )
    assert rid > 0
    rows = db.recent_inspections(10)
    assert any(r.unit_id == "TEST-0001" for r in rows)
    row = next(r for r in rows if r.unit_id == "TEST-0001")
    assert row.defect_types == ["scratch"]
    assert row.passed is False


def test_stats_aggregation():
    db.init_db()
    db.save_inspection(unit_id="P1", passed=True, defects=[],
                       confidence=0.99, cycle_time_ms=4.0, image_path=None)
    db.save_inspection(unit_id="P2", passed=False,
                       defects=[{"label": "stain"}],
                       confidence=0.7, cycle_time_ms=6.0, image_path=None)
    stats = db.compute_stats()
    assert stats["total_inspected"] >= 2
    assert stats["total_passed"] >= 1
    assert stats["total_failed"] >= 1
    assert 0.0 <= stats["yield_rate"] <= 100.0
    assert "stain" in stats["defect_breakdown"]


def test_config_bounds_enforced():
    # Confidence above the allowed ceiling must raise.
    with pytest.raises(ValidationError):
        InspectionConfig(yolo_confidence=1.5)
    with pytest.raises(ValidationError):
        InspectionConfig(min_defect_area=1)  # below ge=10


def test_config_defaults():
    cfg = InspectionConfig()
    assert cfg.active_model == "hybrid"
    assert cfg.plc_reject_enabled is True
    assert cfg.max_defects_to_pass == 0
