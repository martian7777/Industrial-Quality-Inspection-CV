#!/usr/bin/env bash
# Launch the FastAPI backend and the Streamlit HMI together in one container.
set -euo pipefail

# Backend (FastAPI) on an internal port; the HMI talks to it over localhost.
cd /home/user/app/backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Give the backend a moment to bind before the UI starts polling /health.
sleep 4

# Frontend (Streamlit) on the public Hugging Face port.
cd /home/user/app/frontend
exec streamlit run app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
