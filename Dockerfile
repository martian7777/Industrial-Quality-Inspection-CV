# Hugging Face Spaces — single-container build for the Quality Inspection demo.
#
# HF Spaces runs ONE container and exposes ONE port, so this image runs both the
# FastAPI backend (internal localhost:8000) and the Streamlit operator HMI
# (public :7860) together via start.sh. MQTT/PLC are omitted here — the pipeline
# degrades gracefully without a broker (see backend/app/mqtt_client.py).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# OpenCV-headless + ultralytics need a couple of shared libs at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Run as the non-root user (UID 1000) that HF Spaces expects.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    QI_DATA_DIR=/home/user/app/data \
    BACKEND_URL=http://localhost:8000 \
    YOLO_ENABLED=true \
    YOLO_WEIGHTS=yolo11n.pt \
    REFRESH_MS=1200 \
    CAMERA_FPS=2.0 \
    DEFECT_RATE=0.35

USER user
WORKDIR /home/user/app

# --- Python dependencies --------------------------------------------------- #
# Install CPU-only torch FIRST so ultralytics doesn't drag in the multi-GB CUDA
# build (HF free tier is CPU). Then the backend + frontend requirements.
COPY --chown=user backend/requirements.txt backend/requirements.txt
COPY --chown=user frontend/requirements.txt frontend/requirements.txt
RUN pip install --user --upgrade pip \
    && pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --user -r backend/requirements.txt \
    && pip install --user -r frontend/requirements.txt

# --- Application code ------------------------------------------------------- #
COPY --chown=user backend/app backend/app
COPY --chown=user frontend/app.py frontend/app.py
COPY --chown=user start.sh start.sh

# Pre-download the YOLO11 weights into the image so the first inference is
# instant and the deploy never depends on a runtime download succeeding.
RUN cd backend && python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

EXPOSE 7860
CMD ["bash", "start.sh"]
