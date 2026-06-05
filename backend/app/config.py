"""Central system configuration.

Values are driven by environment variables (so they can be overridden by
docker-compose) but every field has a sane default for bare-metal local runs.

The ``InspectionConfig`` block holds the *live* tuning parameters that the
operator can change at runtime from the Streamlit control panel. These are kept
separate from the static deployment settings because they are mutated through
the API while the process is running.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Resolve a data directory that works both in Docker (/data) and on Windows.
_DEFAULT_DATA_DIR = os.getenv("QI_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data"))
DATA_DIR = Path(_DEFAULT_DATA_DIR)
ARCHIVE_DIR = DATA_DIR / "archive"          # defect image archive (for retraining)
MODEL_DIR = DATA_DIR / "models"             # YOLO weights live here
DB_PATH = DATA_DIR / "inspections.db"

for _p in (DATA_DIR, ARCHIVE_DIR, MODEL_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Live inspection tuning — mutable at runtime via the API
# --------------------------------------------------------------------------- #
class InspectionConfig(BaseModel):
    """Operator-tunable inspection parameters.

    Defect classes handled by the system: scratch, dent, missing_component,
    stain, misalignment.
    """

    # Deep-learning gate
    yolo_confidence: float = Field(0.45, ge=0.05, le=0.95, description="Min YOLO confidence")

    # Classical CV gates
    min_defect_area: int = Field(120, ge=10, le=5000, description="Min contour area (px^2) to count as a defect")
    stain_saturation_thresh: int = Field(90, ge=0, le=255, description="HSV saturation cutoff for stain detection")
    scratch_canny_low: int = Field(40, ge=0, le=255)
    scratch_canny_high: int = Field(130, ge=0, le=255)

    # Dimensional tolerance (misalignment / dent) as a fraction of nominal
    dimensional_tolerance: float = Field(0.08, ge=0.0, le=0.5, description="Allowed +/- size deviation fraction")

    # Decision policy
    max_defects_to_pass: int = Field(0, ge=0, le=10, description="Defects allowed before unit is rejected")

    # Industrial outputs
    plc_reject_enabled: bool = Field(True, description="Publish reject commands to the PLC")

    # Active model selection
    active_model: str = Field("hybrid", description="One of: hybrid, yolo_only, opencv_only")


class Settings(BaseModel):
    """Static deployment settings."""

    app_name: str = "Industrial Quality Inspection"
    version: str = "1.0.0"

    # MQTT broker
    mqtt_host: str = os.getenv("MQTT_HOST", "localhost")
    mqtt_port: int = int(os.getenv("MQTT_PORT", "1883"))
    mqtt_topic_results: str = os.getenv("MQTT_TOPIC_RESULTS", "factory/inspection/results")
    mqtt_topic_reject: str = os.getenv("MQTT_TOPIC_REJECT", "factory/plc/reject")

    # Camera simulator
    camera_fps: float = float(os.getenv("CAMERA_FPS", "2.0"))
    defect_rate: float = float(os.getenv("DEFECT_RATE", "0.35"))  # fraction of frames with a defect

    # YOLO
    yolo_weights: str = os.getenv("YOLO_WEIGHTS", "yolo11n.pt")
    yolo_enabled: bool = os.getenv("YOLO_ENABLED", "false").lower() in ("1", "true", "yes")

    # Paths (str copies for serialization)
    db_url: str = f"sqlite:///{DB_PATH.as_posix()}"
    archive_dir: str = str(ARCHIVE_DIR)


settings = Settings()

# The single, process-wide live config instance. Mutated by the API.
live_config = InspectionConfig()
