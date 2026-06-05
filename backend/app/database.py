"""SQLite persistence layer via SQLAlchemy.

Stores one row per inspected unit. Defect details are kept as JSON so the
schema stays stable even as the defect taxonomy evolves.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator, List

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import settings

Base = declarative_base()

# check_same_thread=False because background tasks and request handlers share
# the engine across threads in the asyncio/uvicorn worker model.
engine = create_engine(
    settings.db_url,
    connect_args={"check_same_thread": False},
    future=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


class InspectionResult(Base):
    __tablename__ = "inspection_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    unit_id = Column(String(64), index=True)
    passed = Column(Boolean, index=True)
    defect_count = Column(Integer, default=0)
    defect_details = Column(Text, default="[]")  # JSON list of bounding boxes
    confidence = Column(Float, default=0.0)
    cycle_time_ms = Column(Float, default=0.0)
    image_path = Column(String(512), nullable=True)

    # ---- helpers -------------------------------------------------------- #
    @property
    def defects(self) -> List[dict]:
        try:
            return json.loads(self.defect_details or "[]")
        except json.JSONDecodeError:
            return []

    @property
    def defect_types(self) -> List[str]:
        return sorted({d.get("label", "unknown") for d in self.defects})


def init_db() -> None:
    """Create tables if they do not yet exist."""
    Base.metadata.create_all(bind=engine)


@contextmanager
def session_scope() -> Iterator:
    """Transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def save_inspection(
    *,
    unit_id: str,
    passed: bool,
    defects: List[dict],
    confidence: float,
    cycle_time_ms: float,
    image_path: str | None,
) -> int:
    """Persist a single inspection and return its row id."""
    with session_scope() as s:
        row = InspectionResult(
            unit_id=unit_id,
            passed=passed,
            defect_count=len(defects),
            defect_details=json.dumps(defects),
            confidence=confidence,
            cycle_time_ms=cycle_time_ms,
            image_path=image_path,
        )
        s.add(row)
        s.flush()
        return row.id


def recent_inspections(limit: int = 100) -> List[InspectionResult]:
    with session_scope() as s:
        rows = (
            s.query(InspectionResult)
            .order_by(InspectionResult.timestamp.desc())
            .limit(limit)
            .all()
        )
        return rows


def compute_stats() -> dict:
    """Aggregate KPI statistics across all stored inspections."""
    with session_scope() as s:
        total = s.query(func.count(InspectionResult.id)).scalar() or 0
        passed = (
            s.query(func.count(InspectionResult.id))
            .filter(InspectionResult.passed.is_(True))
            .scalar()
            or 0
        )
        avg_cycle = s.query(func.avg(InspectionResult.cycle_time_ms)).scalar() or 0.0

        # Defect breakdown across all rows.
        breakdown: dict[str, int] = {}
        for (details,) in s.query(InspectionResult.defect_details).all():
            try:
                for d in json.loads(details or "[]"):
                    label = d.get("label", "unknown")
                    breakdown[label] = breakdown.get(label, 0) + 1
            except json.JSONDecodeError:
                continue

    failed = total - passed
    yield_rate = (passed / total * 100.0) if total else 100.0
    return {
        "total_inspected": total,
        "total_passed": passed,
        "total_failed": failed,
        "yield_rate": round(yield_rate, 2),
        "avg_cycle_time_ms": round(float(avg_cycle), 2),
        "defect_breakdown": breakdown,
    }
