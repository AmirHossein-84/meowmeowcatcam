@echo off
REM Web launcher (Windows): serves this folder on http://localhost:8901/
REM Serves repo root for local use only. Do NOT expose to LAN: it serves .git/.
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 ( set PY=py -3 ) else ( set PY=python )

if not exist ".venv\Scripts\python.exe" (
    echo First run - setting up, this takes a few minutes, please keep this window open...
    %PY% -m venv .venv
    if errorlevel 1 ( echo ERROR: could not create .venv. Install Python 3.11+ and retry. & pause & exit /b 1 )
    .venv\Scripts\python.exe -m pip install --upgrade pip
)

echo.
echo Starting Gesture Meme web server on http://localhost:8901/
echo Local only - do not share this URL beyond your machine.
echo Your browser will open automatically. Press Ctrl+C to stop the server.
start "" /b powershell -NoProfile -Command "Start-Sleep 2; Start-Process 'http://localhost:8901/'"
.venv\Scripts\python.exe -m http.server 8901
