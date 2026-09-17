@echo off
REM Double-click launcher for the local web UI (Windows).
REM First run creates a local .venv and installs dependencies, then opens the browser.
setlocal
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo Creating virtual environment .venv ... (one time^)
  where py >nul 2>nul && ( py -3 -m venv .venv ) || ( python -m venv .venv )
  if not exist "%PY%" (
    echo.
    echo Could not create .venv. Please install Python 3.10+ from python.org and try again.
    pause
    exit /b 1
  )
  echo Installing dependencies ... (first run only, can take several minutes^)
  "%PY%" -m pip install --upgrade pip
  "%PY%" -m pip install -r requirements.txt -r requirements-webui.txt
)

echo.
echo Starting the web UI. Your browser will open at http://127.0.0.1:8000
echo Keep this window open while you use it. Close it to stop the server.
echo.
"%PY%" -m webui
pause
