@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo First run - setting up, this takes a few minutes, please keep this window open...
    python -m venv .venv
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -r requirements.txt
)

echo.
echo Starting Gesture Meme web server on http://localhost:8901/
echo Your browser will open automatically. Press Ctrl+C to stop the server.
start "" /b powershell -NoProfile -Command "Start-Sleep 2; Start-Process 'http://localhost:8901/'"
.venv\Scripts\python.exe -m http.server 8901
