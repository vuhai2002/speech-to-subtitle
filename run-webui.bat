@echo off
REM Double-click launcher for the local web UI (Windows).
REM First run creates a local .venv and installs dependencies, then opens the browser.
setlocal
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"

REM The tested CUDA torch (2.5.1) only has wheels for Python 3.10-3.13, but the default Python
REM may be newer (e.g. 3.14) with no GPU torch on the pytorch index. Pick a compatible one for
REM the venv (falls back to the default only if none of 3.10-3.13 is installed).
set "VPY="
py -3.12 --version >nul 2>nul && set "VPY=py -3.12"
if not defined VPY py -3.13 --version >nul 2>nul && set "VPY=py -3.13"
if not defined VPY py -3.11 --version >nul 2>nul && set "VPY=py -3.11"
if not defined VPY py -3.10 --version >nul 2>nul && set "VPY=py -3.10"
if not defined VPY where py >nul 2>nul && set "VPY=py -3"
if not defined VPY set "VPY=python"

if not exist "%PY%" (
  echo Creating virtual environment .venv with "%VPY%" ... (one time^)
  %VPY% -m venv .venv
  if not exist "%PY%" (
    echo.
    echo Could not create .venv. Please install Python 3.12 from python.org and try again.
    pause
    exit /b 1
  )
  echo Installing dependencies ... (first run only, can take several minutes^)
  "%PY%" -m pip install --upgrade pip
  "%PY%" -m pip install -r requirements.txt -r requirements-webui.txt
)

REM On an NVIDIA GPU, make sure torch is the CUDA build so .srt alignment (MMS) runs on the GPU.
REM Skips with no GPU, when torch already sees CUDA, or when the venv Python is too new for it.
where nvidia-smi >nul 2>nul || goto :run
"%PY%" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>nul && goto :run
"%PY%" -c "import sys; sys.exit(0 if sys.version_info[:2] <= (3,13) else 1)" >nul 2>nul || goto :pyold
echo NVIDIA GPU detected - installing the CUDA build of torch (one time, large download) ...
"%PY%" -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
goto :run
:pyold
echo Note: .venv uses a Python too new for the tested CUDA torch (needs 3.10-3.13). Delete .venv
echo       and re-run to rebuild with a compatible Python. MAI still works without a GPU.

:run
echo.
echo Starting the web UI. Your browser will open at http://127.0.0.1:8000
echo Keep this window open while you use it. Close it to stop the server.
echo.
"%PY%" -m webui
pause
