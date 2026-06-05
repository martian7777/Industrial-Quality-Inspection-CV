# Wiki — Setup, Tuning & API Reference

A practical guide to running, configuring and extending the Quality Inspection
system.

---

## 1. Installation

### 1.1 With Docker (recommended)

```bash
docker compose up --build      # builds backend + frontend, starts Mosquitto
docker compose logs -f backend # follow the inspection loop
docker compose down            # stop (add -v to wipe volumes/data)
```

Services & ports:

| Service | URL / Port | Notes |
| --- | --- | --- |
| Operator HMI | http://localhost:8501 | Streamlit dashboard |
| Backend API | http://localhost:8000/docs | Swagger UI |
| MQTT broker | localhost:1883 | + WebSocket on 9001 |

### 1.2 Local (bare metal)

```bash
# Backend
cd backend && pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend
cd frontend && pip install -r requirements.txt
streamlit run app.py
```

Optional local broker:

```bash
docker run -it -p 1883:1883 eclipse-mosquitto:2
# or install mosquitto natively, then:
export MQTT_HOST=localhost
```

---

## 2. Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `MQTT_HOST` | `localhost` | Broker hostname (`mqtt` in compose) |
| `MQTT_PORT` | `1883` | Broker port |
| `MQTT_TOPIC_RESULTS` | `factory/inspection/results` | Verdict topic |
| `MQTT_TOPIC_REJECT` | `factory/plc/reject` | Reject-command topic |
| `CAMERA_FPS` | `2.0` | Inspection cadence |
| `DEFECT_RATE` | `0.35` | Fraction of synthetic frames given a defect |
| `YOLO_ENABLED` | `false` | Load real Ultralytics YOLO11 |
| `YOLO_WEIGHTS` | `yolo11n.pt` | Weights path/name |
| `QI_DATA_DIR` | `./data` (`/data` in Docker) | DB + archive + models root |
| `QI_FRAME_DIR` | — | If set, stream images from this dir instead of synthesising |
| `BACKEND_URL` | `http://localhost:8000` | HMI → backend base URL |
| `REFRESH_MS` | `1200` | HMI auto-refresh interval |

---

## 3. Classical CV Tuning Guide

All parameters below are live-editable from the **sidebar control panel** and
map 1:1 to `InspectionConfig` fields.

### Stains — HSV saturation
- **`stain_saturation_thresh`** (0–255): pixels above this saturation become
  stain candidates. Brushed metal is near-grey (low saturation), so a value of
  ~90 isolates coloured oil/rust blobs. **Lower** to catch faint stains (risk:
  false positives from coloured reflections); **raise** for only vivid stains.

### Scratches — Canny + Hough
- **`scratch_canny_low` / `scratch_canny_high`**: hysteresis edge thresholds.
  Widen the band (lower low, higher high) to capture faint scratches; narrow it
  to suppress surface-texture noise. The image is masked to the eroded part
  contour first, so belt seams don't register.
- Detection requires a Hough line ≥ 35 px; tune `minLineLength` in code for
  finer/coarser scratches.

### Stains & general blobs — minimum area
- **`min_defect_area`** (px²): contours smaller than this are discarded as
  noise. Raise to ignore speckle; lower to catch pinhole defects.

### Dents & misalignment — dimensional gate
- **`dimensional_tolerance`** (fraction): allowed ± deviation of the part's
  enclosing-circle radius from nominal before flagging a **dent**. Misalignment
  trips when the centroid is offset from the frame centre by more than
  `2.5 ×` tolerance. Tighten for precision parts; loosen for hand-placed units.

### Decision policy
- **`max_defects_to_pass`**: defects tolerated before rejection (0 = zero-defect
  policy).
- **`active_model`**: `hybrid` (both), `yolo_only`, or `opencv_only` — useful
  for A/B comparing the detector families.
- **`plc_reject_enabled`**: gate the physical reject output without stopping
  inspection/logging.

---

## 4. Enabling Real YOLO11

1. Add `ultralytics==8.3.40` to `backend/requirements.txt` (uncomment) and
   reinstall.
2. Place weights in `data/models/` (or let `yolo11n.pt` auto-download).
3. Set `YOLO_ENABLED=true` and `YOLO_WEIGHTS=/data/models/best.pt`.

The engine maps YOLO class names into the defect taxonomy and merges them with
OpenCV detections through the same NMS path used by the mock — no other change
required.

### Training a custom defect model (outline)
- Collect images from `data/archive/<date>/` (the system auto-archives every
  reject — your dataset grows itself).
- Label with the five classes: `scratch, dent, missing_component, stain,
  misalignment`.
- Train: `yolo detect train data=defects.yaml model=yolo11n.pt epochs=100 imgsz=640`.
- Drop `runs/detect/train/weights/best.pt` into `data/models/` and point
  `YOLO_WEIGHTS` at it.

---

## 5. API Reference

Base URL: `http://localhost:8000`

### `GET /health`
```json
{ "status":"ok","running":true,"yolo_active":false,"mqtt_connected":true,"plc_online":true }
```

### `GET /api/frame/latest`
Latest annotated frame and verdict. `image_b64` is a base64 JPEG.
```json
{ "unit_id":"U01234-0007","passed":false,"defect_count":1,
  "defects":[{"x":..,"y":..,"w":..,"h":..,"label":"scratch","confidence":0.81,"source":"opencv"}],
  "confidence":0.81,"cycle_time_ms":7.4,"image_b64":"/9j/4AAQ..." }
```

### `GET /api/stats`
```json
{ "total_inspected":420,"total_passed":273,"total_failed":147,
  "yield_rate":65.0,"avg_cycle_time_ms":6.9,
  "defect_breakdown":{"scratch":61,"stain":40,"dent":22,"misalignment":14,"missing_component":10},
  "last_reject_unit":"U01234-0007" }
```

### `GET /api/logs?limit=100`
Array of inspection rows (newest first): `id, timestamp, unit_id, passed,
defect_count, defect_types[], confidence, cycle_time_ms, image_path`.

### `GET /api/config` · `PATCH /api/config`
Read or partially update `InspectionConfig`. PATCH accepts any subset:
```bash
curl -X PATCH localhost:8000/api/config \
  -H 'Content-Type: application/json' \
  -d '{"yolo_confidence":0.6,"max_defects_to_pass":1}'
```

### `GET /api/plc/status`
```json
{ "online":true,"last_command":"REJECT","last_reject_unit":"U01234-0007",
  "last_reject_at":"2026-06-05T10:22:31","total_rejects":147 }
```

### `POST /api/control/{pause|resume}`
Stop/start the conveyor loop.

### `WS /ws/stream`
On connect, receives the current frame, then one JSON message per inspection
(same shape as `/api/frame/latest`). Send any text as a keepalive ping.

---

## 6. MQTT Topics

| Topic | Direction | Payload |
| --- | --- | --- |
| `factory/inspection/results` | controller → bus | `{unit_id, passed, defect_count, defect_types[], confidence, cycle_time_ms, timestamp}` |
| `factory/plc/reject` | controller → PLC | `{command:"REJECT", unit_id, defects[], timestamp}` |

Subscribe from a shell to watch the line:
```bash
mosquitto_sub -h localhost -t 'factory/#' -v
```

---

## 7. Data & Storage Layout

```
data/
├── inspections.db          # SQLite audit log (inspection_results table)
├── archive/
│   └── 2026-06-05/         # annotated images of rejected units, by date
│       └── U01234-0007.jpg
└── models/                 # YOLO weights (when YOLO_ENABLED=true)
```

The `inspection_results` table stores the verdict, a JSON list of bounding-box
detections, confidence, cycle time and the archived image path.

---

## 8. Testing

```bash
cd backend && pip install pytest && pytest -q
```

Covers the camera synthesizer, the OpenCV + mock-YOLO inference verdicts, the
SQLite persistence/stats path, and config validation. See
[backend/tests/](backend/tests/).

---

## 9. Troubleshooting

| Symptom | Cause / Fix |
| --- | --- |
| HMI: "Cannot reach backend" | Backend not up, or `BACKEND_URL` wrong. Check `GET /health`. |
| PLC LED off / no rejects actuated | Broker unreachable. Confirm Mosquitto is running and `MQTT_HOST` is correct. Inspection still works in degraded mode. |
| `yolo_active:false` | Expected unless `YOLO_ENABLED=true` and Ultralytics + weights are installed. The mock keeps the pipeline live. |
| Live feed stuck on "Waiting for first frame" | Line paused (`POST /api/control/resume`) or first iteration not complete yet. |
| Too many / too few defects flagged | Tune thresholds in §3 from the sidebar. |
