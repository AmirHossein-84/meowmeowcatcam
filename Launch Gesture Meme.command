#!/bin/bash
cd "$(dirname "$0")"

if [ ! -f .venv/bin/python3 ]; then
    echo "First run — downloading and installing (mediapipe + opencv are ~90MB total, this can take several minutes depending on your internet — please don't close this window)..."
    python3 -m venv .venv
    .venv/bin/pip install --quiet --upgrade pip
    .venv/bin/pip install --quiet -r requirements.txt
fi

.venv/bin/python3 gesture_meme.py
