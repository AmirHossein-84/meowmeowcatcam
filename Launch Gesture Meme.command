#!/bin/bash
# Desktop launcher (macOS/Linux): venv + gesture_meme.py
set -u
cd "$(dirname "$0")"

fail() { echo "ERROR: $1" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || fail "python3 not found. Install Python >=3.11 (python.org or Xcode CLT), then retry."

if [ ! -f .venv/bin/python3 ]; then
    echo "First run — creating .venv and installing (mediapipe + opencv, ~90MB, several minutes)..."
    python3 -m venv .venv || fail "python3 -m venv failed."
    .venv/bin/pip install --upgrade pip || fail "pip upgrade failed — see output above."
    .venv/bin/pip install -r requirements.txt || fail "pip install failed — see output above."
fi

[ -f models/hand_landmarker.task ] || echo "WARNING: models/hand_landmarker.task missing — run python3 scripts/download_models.py"
[ -f models/face_landmarker.task ] || echo "WARNING: models/face_landmarker.task missing — run python3 scripts/download_models.py"

exec .venv/bin/python3 gesture_meme.py
