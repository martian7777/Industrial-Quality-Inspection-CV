"""Operator HMI — Streamlit dashboard for the Quality Inspection line.

A dark, industrial control-room style panel that polls the FastAPI backend for
the latest annotated frame, KPI statistics, inspection logs and PLC state, and
lets the operator tune the inspection config live.

Design goals:
  * High-contrast dark theme with metric tiles + status LEDs.
  * Near real-time live feed (auto-refresh).
  * Everything driven by the backend REST API (no business logic here).
"""

from __future__ import annotations

import base64
import os
from datetime import datetime
from io import BytesIO

import pandas as pd
import requests
import streamlit as st

API = os.getenv("BACKEND_URL", "http://localhost:8000")
REFRESH_MS = int(os.getenv("REFRESH_MS", "1200"))

st.set_page_config(
    page_title="Quality Inspection HMI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Custom industrial dark theme
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    :root { --accent:#00e5ff; --ok:#23d18b; --bad:#ff4d5e; --warn:#ffb02e; }
    .stApp { background: radial-gradient(1200px 600px at 20% -10%, #16202b 0%, #0b1016 55%, #070a0e 100%); }
    section[data-testid="stSidebar"] { background:#0d141c; border-right:1px solid #1c2733; }
    h1,h2,h3,h4 { color:#e8eef5 !important; font-family:'Segoe UI',sans-serif; letter-spacing:.3px; }
    .block-container { padding-top:1.4rem; }
    .tile {
        background:linear-gradient(160deg,#13202c 0%,#0e1822 100%);
        border:1px solid #1f2e3c; border-radius:14px; padding:16px 18px;
        box-shadow:0 6px 18px rgba(0,0,0,.35); height:100%;
    }
    .tile .label { color:#7d93a8; font-size:.72rem; text-transform:uppercase; letter-spacing:1.4px; }
    .tile .value { color:#eaf2fa; font-size:2.0rem; font-weight:700; line-height:1.1; margin-top:4px; }
    .tile .sub { color:#5f7488; font-size:.74rem; margin-top:2px; }
    .led { display:inline-block; width:12px; height:12px; border-radius:50%; margin-right:8px; vertical-align:middle; }
    .led.on  { background:var(--ok);  box-shadow:0 0 10px var(--ok); }
    .led.off { background:#445; }
    .led.bad { background:var(--bad); box-shadow:0 0 12px var(--bad); }
    .led.warn{ background:var(--warn);box-shadow:0 0 10px var(--warn); }
    .verdict-pass { color:var(--ok);  font-weight:800; font-size:1.5rem; }
    .verdict-fail { color:var(--bad); font-weight:800; font-size:1.5rem; }
    .reject-banner {
        background:linear-gradient(90deg,#3a0c12,#7a1422); border:1px solid #ff4d5e;
        border-radius:12px; padding:14px 18px; color:#ffd9dd; font-weight:700;
        animation:pulse 1s infinite; text-align:center; font-size:1.1rem;
    }
    .idle-banner {
        background:#10202a; border:1px solid #1f3a47; border-radius:12px;
        padding:14px 18px; color:#7fb6c9; text-align:center;
    }
    @keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(255,77,94,.5);} 70%{box-shadow:0 0 0 14px rgba(255,77,94,0);} 100%{box-shadow:0 0 0 0 rgba(255,77,94,0);} }
    .statusbar { color:#8ca3b8; font-size:.8rem; }
    .stDataFrame { border:1px solid #1f2e3c; border-radius:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Auto-refresh
try:
    from streamlit_autorefresh import st_autorefresh

    st_autorefresh(interval=REFRESH_MS, key="auto")
except Exception:
    st.markdown(
        f"<meta http-equiv='refresh' content='{max(1, REFRESH_MS // 1000)}'>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# API helpers
# --------------------------------------------------------------------------- #
def api_get(path: str, default=None, **params):
    try:
        r = requests.get(f"{API}{path}", params=params, timeout=4)
        r.raise_for_status()
        return r.json()
    except Exception:
        return default


def api_patch(path: str, payload: dict):
    try:
        r = requests.patch(f"{API}{path}", json=payload, timeout=4)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.sidebar.error(f"Update failed: {exc}")
        return None


def tile(col, label, value, sub=""):
    col.markdown(
        f"<div class='tile'><div class='label'>{label}</div>"
        f"<div class='value'>{value}</div><div class='sub'>{sub}</div></div>",
        unsafe_allow_html=True,
    )


def led(on: bool, kind_on: str = "on") -> str:
    return f"<span class='led {kind_on if on else 'off'}'></span>"


# --------------------------------------------------------------------------- #
# Header + connection status
# --------------------------------------------------------------------------- #
health = api_get("/health", default={}) or {}
online = bool(health)

hc1, hc2 = st.columns([0.7, 0.3])
hc1.markdown("## 🏭 Industrial Quality Inspection — Operator Panel")
hc2.markdown(
    f"<div style='text-align:right;padding-top:14px' class='statusbar'>"
    f"{led(online)} Backend "
    f"{led(health.get('mqtt_connected', False))} MQTT "
    f"{led(health.get('plc_online', False))} PLC "
    f"{led(health.get('yolo_active', False), 'warn')} YOLO"
    f"</div>",
    unsafe_allow_html=True,
)

if not online:
    st.error(f"Cannot reach backend at {API}. Is the FastAPI service running?")
    st.stop()

# --------------------------------------------------------------------------- #
# Sidebar — control panel
# --------------------------------------------------------------------------- #
st.sidebar.markdown("### ⚙️ Inspection Control Panel")
cfg = api_get("/api/config", default={}) or {}

if cfg:
    with st.sidebar.form("config_form"):
        active_model = st.selectbox(
            "Active model", ["hybrid", "yolo_only", "opencv_only"],
            index=["hybrid", "yolo_only", "opencv_only"].index(cfg.get("active_model", "hybrid")),
        )
        yolo_conf = st.slider("YOLO confidence", 0.05, 0.95, float(cfg.get("yolo_confidence", 0.45)), 0.01)
        min_area = st.slider("Min defect area (px²)", 10, 5000, int(cfg.get("min_defect_area", 120)), 10)
        sat = st.slider("Stain saturation cutoff", 0, 255, int(cfg.get("stain_saturation_thresh", 90)), 1)
        c_low = st.slider("Scratch Canny low", 0, 255, int(cfg.get("scratch_canny_low", 40)), 1)
        c_high = st.slider("Scratch Canny high", 0, 255, int(cfg.get("scratch_canny_high", 130)), 1)
        tol = st.slider("Dimensional tolerance", 0.0, 0.5, float(cfg.get("dimensional_tolerance", 0.08)), 0.01)
        max_pass = st.slider("Max defects to pass", 0, 10, int(cfg.get("max_defects_to_pass", 0)), 1)
        plc_on = st.toggle("PLC reject output", value=bool(cfg.get("plc_reject_enabled", True)))
        submitted = st.form_submit_button("Apply configuration", use_container_width=True)

    if submitted:
        api_patch("/api/config", {
            "active_model": active_model,
            "yolo_confidence": yolo_conf,
            "min_defect_area": min_area,
            "stain_saturation_thresh": sat,
            "scratch_canny_low": c_low,
            "scratch_canny_high": c_high,
            "dimensional_tolerance": tol,
            "max_defects_to_pass": max_pass,
            "plc_reject_enabled": plc_on,
        })
        st.sidebar.success("Configuration applied")

st.sidebar.markdown("---")
cc1, cc2 = st.sidebar.columns(2)
if cc1.button("⏸ Pause line", use_container_width=True):
    requests.post(f"{API}/api/control/pause", timeout=4)
if cc2.button("▶ Resume line", use_container_width=True):
    requests.post(f"{API}/api/control/resume", timeout=4)

# --------------------------------------------------------------------------- #
# Inspection source — feed your own image / video / live stream
# --------------------------------------------------------------------------- #
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎞️ Inspection Source")
src = api_get("/api/source", default={}) or {}
src_kind = src.get("kind", "simulator")
src_label = src.get("label", "")
if src_kind == "simulator":
    st.sidebar.caption("Source: built-in conveyor simulator")
else:
    st.sidebar.caption(f"Source: **{src_kind}** — {src_label}")

with st.sidebar.expander("Use my own image / video / stream", expanded=False):
    up = st.file_uploader(
        "Upload image or video",
        type=["png", "jpg", "jpeg", "bmp", "webp", "mp4", "avi", "mov", "mkv", "webm"],
        key="source_upload",
    )
    if up is not None and st.button("▶ Run on upload", use_container_width=True):
        is_video = up.name.lower().rsplit(".", 1)[-1] in (
            "mp4", "avi", "mov", "mkv", "webm",
        )
        endpoint = "/api/source/video" if is_video else "/api/source/image"
        try:
            r = requests.post(
                f"{API}{endpoint}",
                files={"file": (up.name, up.getvalue(), up.type or "application/octet-stream")},
                timeout=30,
            )
            if r.ok:
                st.success(f"Now inspecting: {up.name}")
            else:
                st.error(r.json().get("detail", "Upload failed"))
        except Exception as exc:
            st.error(f"Upload failed: {exc}")

    stream_url = st.text_input(
        "…or live stream URL",
        placeholder="rtsp://…  ·  http://…/mjpg  ·  0 (webcam)",
        key="stream_url",
    )
    if stream_url and st.button("▶ Connect stream", use_container_width=True):
        try:
            r = requests.post(f"{API}/api/source/stream", json={"url": stream_url}, timeout=15)
            if r.ok:
                st.success(f"Streaming from: {stream_url}")
            else:
                st.error(r.json().get("detail", "Could not open stream"))
        except Exception as exc:
            st.error(f"Stream failed: {exc}")

if src_kind != "simulator" and st.sidebar.button("↺ Back to simulator", use_container_width=True):
    requests.post(f"{API}/api/source/reset", timeout=4)
    st.rerun()

# --------------------------------------------------------------------------- #
# KPI tiles
# --------------------------------------------------------------------------- #
stats = api_get("/api/stats", default={}) or {}
k1, k2, k3, k4, k5 = st.columns(5)
tile(k1, "Yield Rate", f"{stats.get('yield_rate', 0):.1f}%", "pass / total")
tile(k2, "Total Inspected", f"{stats.get('total_inspected', 0):,}", "units")
tile(k3, "Passed", f"{stats.get('total_passed', 0):,}", "OK units")
tile(k4, "Rejected", f"{stats.get('total_failed', 0):,}", "defective")
tile(k5, "Avg Cycle", f"{stats.get('avg_cycle_time_ms', 0):.0f} ms", "per unit")

st.markdown("")

# --------------------------------------------------------------------------- #
# Live feed + PLC monitor
# --------------------------------------------------------------------------- #
feed_col, side_col = st.columns([0.62, 0.38])

frame = api_get("/api/frame/latest", default=None)
with feed_col:
    st.markdown("#### 📹 Live Conveyor Feed")
    if frame and frame.get("image_b64"):
        img_bytes = base64.b64decode(frame["image_b64"])
        st.image(BytesIO(img_bytes), use_container_width=True)
        verdict = "PASS" if frame["passed"] else "REJECT"
        cls = "verdict-pass" if frame["passed"] else "verdict-fail"
        st.markdown(
            f"<span class='{cls}'>{verdict}</span> &nbsp; "
            f"<span class='statusbar'>Unit <b>{frame['unit_id']}</b> · "
            f"{frame['defect_count']} defect(s) · "
            f"conf {frame['confidence']:.2f} · {frame['cycle_time_ms']:.0f} ms</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<div class='idle-banner'>Waiting for first frame…</div>", unsafe_allow_html=True)

with side_col:
    st.markdown("#### 🤖 Virtual PLC — Reject Actuator")
    plc = api_get("/api/plc/status", default={}) or {}
    just_rejected = frame is not None and not frame.get("passed", True)
    if just_rejected:
        st.markdown(
            f"<div class='reject-banner'>⛔ PUSHER ACTUATED<br>Rejecting {frame['unit_id']}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<div class='idle-banner'>✅ Actuator idle — line clear</div>", unsafe_allow_html=True)

    st.markdown(
        f"<div class='tile' style='margin-top:12px'>"
        f"<div class='label'>PLC State</div>"
        f"<div class='sub' style='font-size:.9rem;margin-top:8px'>"
        f"{led(plc.get('online', False))} {'ONLINE' if plc.get('online') else 'OFFLINE'}<br>"
        f"Total rejects: <b>{plc.get('total_rejects', 0)}</b><br>"
        f"Last unit: <b>{plc.get('last_reject_unit') or '—'}</b></div></div>",
        unsafe_allow_html=True,
    )

    # Defect breakdown
    breakdown = stats.get("defect_breakdown", {})
    if breakdown:
        st.markdown("#### 🧪 Defect Breakdown")
        df_b = pd.DataFrame(
            {"defect": list(breakdown.keys()), "count": list(breakdown.values())}
        ).sort_values("count", ascending=False)
        st.bar_chart(df_b, x="defect", y="count", color="#ff4d5e", height=220)

# --------------------------------------------------------------------------- #
# Inspection log
# --------------------------------------------------------------------------- #
st.markdown("#### 📑 Inspection Log")
logs = api_get("/api/logs", default=[], limit=200) or []
if logs:
    df = pd.DataFrame(logs)
    df["verdict"] = df["passed"].map({True: "✅ PASS", False: "⛔ REJECT"})
    df["defects"] = df["defect_types"].apply(lambda x: ", ".join(x) if x else "—")
    show = df[["timestamp", "unit_id", "verdict", "defect_count", "defects",
               "confidence", "cycle_time_ms"]].rename(columns={
        "timestamp": "Time", "unit_id": "Unit", "verdict": "Verdict",
        "defect_count": "#", "defects": "Defects",
        "confidence": "Conf", "cycle_time_ms": "Cycle (ms)",
    })
    st.dataframe(show, use_container_width=True, hide_index=True, height=320)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download log (CSV)", csv,
                       file_name=f"inspection_log_{datetime.now():%Y%m%d_%H%M%S}.csv",
                       mime="text/csv")
else:
    st.info("No inspections recorded yet.")

st.markdown(
    f"<div class='statusbar' style='text-align:center;margin-top:18px'>"
    f"Auto-refresh every {REFRESH_MS} ms · Backend {API}</div>",
    unsafe_allow_html=True,
)
