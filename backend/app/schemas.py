"""Pydantic schemas for API requests and responses."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class BoundingBox(BaseModel):
    x: int
    y: int
    w: int
    h: int
    label: str
    confidence: float
    source: str  # "yolo" or "opencv"


class InspectionResultOut(BaseModel):
    id: int
    timestamp: datetime
    unit_id: str
    passed: bool
    defect_count: int
    defect_types: List[str]
    confidence: float
    cycle_time_ms: float
    image_path: Optional[str] = None

    class Config:
        from_attributes = True


class LiveFrame(BaseModel):
    """The latest annotated frame plus its inspection verdict."""

    unit_id: str
    timestamp: datetime
    passed: bool
    defect_count: int
    defects: List[BoundingBox]
    confidence: float
    cycle_time_ms: float
    image_b64: str  # JPEG, base64-encoded annotated frame


class StatsOut(BaseModel):
    total_inspected: int
    total_passed: int
    total_failed: int
    yield_rate: float
    avg_cycle_time_ms: float
    defect_breakdown: dict
    last_reject_unit: Optional[str] = None


class ConfigUpdate(BaseModel):
    """Partial update of the live inspection config (all fields optional)."""

    yolo_confidence: Optional[float] = None
    min_defect_area: Optional[int] = None
    stain_saturation_thresh: Optional[int] = None
    scratch_canny_low: Optional[int] = None
    scratch_canny_high: Optional[int] = None
    dimensional_tolerance: Optional[float] = None
    max_defects_to_pass: Optional[int] = None
    plc_reject_enabled: Optional[bool] = None
    active_model: Optional[str] = None


class PLCStatus(BaseModel):
    online: bool
    last_command: Optional[str] = None
    last_reject_unit: Optional[str] = None
    last_reject_at: Optional[datetime] = None
    total_rejects: int = 0
