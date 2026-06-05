"""Pytest fixtures — isolate each run in a temp data dir before app imports."""

import os
import tempfile
import sys
from pathlib import Path

# Point the app at a throwaway data dir BEFORE any app module is imported,
# so config.py creates its paths there and tests never touch real data.
_TMP = tempfile.mkdtemp(prefix="qi_test_")
os.environ.setdefault("QI_DATA_DIR", _TMP)

# Ensure the backend package root is importable (backend/ on sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
