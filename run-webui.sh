#!/usr/bin/env bash
# Launcher for the local web UI (macOS / Linux).
# First run creates a local .venv and installs dependencies, then starts the server.
set -e
cd "$(dirname "$0")"
PY=".venv/bin/python"

# The tested CUDA torch (2.5.1) only has wheels for Python 3.10-3.13, but the default python3
# may be newer with no GPU torch. Pick a compatible interpreter for the venv when available.
VPY=""
for c in python3.12 python3.13 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then VPY="$c"; break; fi
done
[ -n "$VPY" ] || VPY="python3"

if [ ! -x "$PY" ]; then
  echo "Creating virtual environment .venv with $VPY ... (one time)"
  "$VPY" -m venv .venv
  echo "Installing dependencies ... (first run only, can take several minutes)"
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r requirements.txt -r requirements-webui.txt
fi

# On an NVIDIA GPU, make sure torch is the CUDA build so .srt alignment (MMS) runs on the GPU.
# Skips with no GPU (incl. macOS), when torch already sees CUDA, or when Python is too new for it.
if command -v nvidia-smi >/dev/null 2>&1 && ! "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
  if "$PY" -c "import sys; sys.exit(0 if sys.version_info[:2] <= (3,13) else 1)" >/dev/null 2>&1; then
    echo "NVIDIA GPU detected - installing the CUDA build of torch (one time, large download) ..."
    "$PY" -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
  else
    echo "Note: .venv uses a Python too new for the tested CUDA torch (needs 3.10-3.13). Delete .venv"
    echo "      and re-run to rebuild with a compatible Python. MAI still works without a GPU."
  fi
fi

echo
echo "Starting the web UI at http://127.0.0.1:8000 (Ctrl+C to stop)."
echo
exec "$PY" -m webui
