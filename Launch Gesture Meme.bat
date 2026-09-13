@echo off
REM Desktop launcher (Windows): venv + gesture_meme.py
cd /d "%~dp0"
setlocal

where py >nul 2>nul
if %errorlevel%==0 ( set PY=py -3 ) else ( set PY=python )

if not exist ".venv\Scripts\python.exe" (
    echo First run - creating .venv and installing, this takes a few minutes, please keep this window open...
    %PY% -m venv .venv
    if errorlevel 1 ( echo ERROR: could not create .venv. Install Python 3.11+ from python.org and retry. & pause & exit /b 1 )
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    if errorlevel 1 ( echo ERROR: pip install failed - see output above. & pause & exit /b 1 )
)

if not exist "models\hand_landmarker.task" echo WARNING: models\hand_landmarker.task missing - run scripts\download_models.py
.venv\Scripts\python.exe gesture_meme.py
if errorlevel 1 ( echo App exited with an error - see messages above. Check camera permissions and close Zoom/OBS. & pause )
