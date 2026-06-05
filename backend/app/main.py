"""FastAPI entrypoint for the Industrial Quality Inspection backend.

Responsibilities:
  * Run the inspection loop as a background task: grab frame -> inspect ->
    persist -> publish to MQTT -> cache latest annotated frame.
  * Serve REST endpoints for the latest frame, KPI stats, inspection logs, the
    live config (GET/PATCH), and PLC status.
  * Serve a WebSocket that streams every new verdict to connected dashboards.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

import cv2
import numpy as np
from fastapi import (
    Body,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware

from . import database as db
from .camera import CameraSimulator
from .config import live_config, settings
from .inference import InspectionEngine
from .mqtt_client import InspectionPublisher, VirtualPLC
from .schemas import ConfigUpdate, InspectionResultOut, PLCStatus, StatsOut

# --------------------------------------------------------------------------- #
# Shared runtime state
# --------------------------------------------------------------------------- #
camera = CameraSimulator(defect_rate=settings.defect_rate)
engine = InspectionEngine()
publisher = InspectionPublisher()
plc = VirtualPLC()


class RuntimeState:
    """Process-wide mutable state shared between the loop and the API."""

    def __init__(self) -> None:
        self.latest_frame: Optional[dict] = None
        self.running: bool = True
        self.ws_clients: List[WebSocket] = []
        self.lock = asyncio.Lock()


state = RuntimeState()


def _encode_jpeg_b64(image) -> str:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _archive_defect_image(unit_id: str, annotated) -> Optional[str]:
    """Save failed-unit images grouped by date for later model retraining."""
    day = datetime.utcnow().strftime("%Y-%m-%d")
    folder = db.settings.archive_dir if hasattr(db.settings, "archive_dir") else settings.archive_dir
    import os

    out_dir = os.path.join(folder, day)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{unit_id}.jpg")
    cv2.imwrite(path, annotated)
    return path


# --------------------------------------------------------------------------- #
# Inspection loop
# --------------------------------------------------------------------------- #
async def inspection_loop() -> None:
    interval = 1.0 / max(0.1, settings.camera_fps)
    while True:
        if not state.running:
            await asyncio.sleep(0.2)
            continue

        # CPU-bound work off the event loop.
        frame = await asyncio.to_thread(camera.grab_frame)
        result = await asyncio.to_thread(engine.inspect, frame, live_config)

        image_path = None
        if not result.passed:
            image_path = await asyncio.to_thread(
                _archive_defect_image, result.unit_id, result.annotated
            )

        defect_types = sorted({d["label"] for d in result.detections})

        # Persist.
        await asyncio.to_thread(
            db.save_inspection,
            unit_id=result.unit_id,
            passed=result.passed,
            defects=list(result.detections),
            confidence=result.confidence,
            cycle_time_ms=result.cycle_time_ms,
            image_path=image_path,
        )

        # Publish to MQTT (+ reject command if applicable).
        payload = {
            "unit_id": result.unit_id,
            "passed": result.passed,
            "defect_count": len(result.detections),
            "defect_types": defect_types,
            "confidence": result.confidence,
            "cycle_time_ms": round(result.cycle_time_ms, 2),
            "timestamp": datetime.utcnow().isoformat(),
        }
        await asyncio.to_thread(publisher.publish_result, payload)
        if not result.passed and live_config.plc_reject_enabled:
            await asyncio.to_thread(publisher.publish_reject, result.unit_id, defect_types)

        # Cache latest frame for HTTP pollers + push to WS subscribers.
        frame_msg = {
            "unit_id": result.unit_id,
            "timestamp": payload["timestamp"],
            "passed": result.passed,
            "defect_count": len(result.detections),
            "defects": list(result.detections),
            "confidence": result.confidence,
            "cycle_time_ms": round(result.cycle_time_ms, 2),
            "image_b64": _encode_jpeg_b64(result.annotated),
        }
        async with state.lock:
            state.latest_frame = frame_msg
        await _broadcast(frame_msg)

        await asyncio.sleep(interval)


async def _broadcast(message: dict) -> None:
    if not state.ws_clients:
        return
    dead: List[WebSocket] = []
    data = json.dumps(message, default=str)
    for ws in state.ws_clients:
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in state.ws_clients:
            state.ws_clients.remove(ws)


# --------------------------------------------------------------------------- #
# Lifespan
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    publisher.connect()
    plc.connect()
    task = asyncio.create_task(inspection_loop())
    print("[main] inspection loop started")
    try:
        yield
    finally:
        state.running = False
        task.cancel()
        publisher.disconnect()
        plc.disconnect()


app = FastAPI(title=settings.app_name, version=settings.version, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# REST endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "running": state.running,
        "yolo_active": engine.yolo_active,
        "mqtt_connected": publisher.connected,
        "plc_online": plc.online,
    }


@app.get("/api/frame/latest")
async def latest_frame():
    async with state.lock:
        if state.latest_frame is None:
            raise HTTPException(status_code=404, detail="No frame yet")
        return state.latest_frame


@app.get("/api/stats", response_model=StatsOut)
async def stats():
    s = await asyncio.to_thread(db.compute_stats)
    s["last_reject_unit"] = plc.last_reject_unit
    return s


@app.get("/api/logs", response_model=List[InspectionResultOut])
async def logs(limit: int = 100):
    rows = await asyncio.to_thread(db.recent_inspections, limit)
    return [
        InspectionResultOut(
            id=r.id,
            timestamp=r.timestamp,
            unit_id=r.unit_id,
            passed=r.passed,
            defect_count=r.defect_count,
            defect_types=r.defect_types,
            confidence=r.confidence,
            cycle_time_ms=r.cycle_time_ms,
            image_path=r.image_path,
        )
        for r in rows
    ]


@app.get("/api/config")
async def get_config():
    return live_config.model_dump()


@app.patch("/api/config")
async def update_config(update: ConfigUpdate):
    changed = update.model_dump(exclude_none=True)
    for key, value in changed.items():
        setattr(live_config, key, value)
    # Re-validate the mutated config.
    validated = live_config.__class__(**live_config.model_dump())
    for key in changed:
        setattr(live_config, key, getattr(validated, key))
    return {"updated": changed, "config": live_config.model_dump()}


@app.get("/api/plc/status", response_model=PLCStatus)
async def plc_status():
    return plc.status()


@app.post("/api/control/{action}")
async def control(action: str):
    if action == "pause":
        state.running = False
    elif action == "resume":
        state.running = True
    else:
        raise HTTPException(status_code=400, detail="action must be pause|resume")
    return {"running": state.running}


# --------------------------------------------------------------------------- #
# Inspection source — let operators feed their own image / video / live stream
# --------------------------------------------------------------------------- #
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


@app.get("/api/source")
async def get_source():
    return camera.source_info()


@app.post("/api/source/image")
async def set_source_image(file: UploadFile = File(...)):
    """Upload a still image — inspected on a loop as the live feed."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    camera.set_image(img, label=file.filename or "uploaded image")
    state.running = True
    return camera.source_info()


@app.post("/api/source/video")
async def set_source_video(file: UploadFile = File(...)):
    """Upload a video file — streamed frame-by-frame (looped) as the live feed."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    suffix = os.path.splitext(file.filename or "")[1].lower() or ".mp4"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="qi_src_")
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    ok = await asyncio.to_thread(
        camera.set_video, path, "video", file.filename or "uploaded video"
    )
    if not ok:
        os.unlink(path)
        raise HTTPException(status_code=400, detail="Could not open video file")
    state.running = True
    return camera.source_info()


@app.post("/api/source/stream")
async def set_source_stream(payload: dict = Body(...)):
    """Point the feed at a live stream URL (RTSP / HTTP-MJPEG) or webcam index."""
    url = str(payload.get("url", "")).strip()
    if not url:
        raise HTTPException(status_code=400, detail="Missing 'url'")
    target = int(url) if url.isdigit() else url  # allow "0" => local webcam
    ok = await asyncio.to_thread(camera.set_video, target, "stream", url)
    if not ok:
        raise HTTPException(status_code=400, detail=f"Could not open stream: {url}")
    state.running = True
    return camera.source_info()


@app.post("/api/source/reset")
async def reset_source():
    """Revert to the built-in synthetic conveyor simulator."""
    camera.reset_source()
    return camera.source_info()


# --------------------------------------------------------------------------- #
# WebSocket — live verdict stream
# --------------------------------------------------------------------------- #
@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    state.ws_clients.append(websocket)
    try:
        # Send the current frame immediately so the client isn't blank.
        async with state.lock:
            if state.latest_frame is not None:
                await websocket.send_text(json.dumps(state.latest_frame, default=str))
        while True:
            await websocket.receive_text()  # keepalive / client pings
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in state.ws_clients:
            state.ws_clients.remove(websocket)
