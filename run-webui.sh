#!/usr/bin/env bash
# Launcher for the local web UI (macOS / Linux).
# First run creates a local .venv and installs dependencies, then opens the browser.
# macOS: to double-click, copy this to run-webui.command and make it executable.
set -e
cd "$(dirname "$0")"
PY=".venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "Creating virtual environment .venv ... (one time)"
  python3 -m venv .venv
  echo "Installing dependencies ... (first run only, can take several minutes)"
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r requirements.txt -r requirements-webui.txt
fi

echo "Starting the web UI at http://127.0.0.1:8000 (press Ctrl+C to stop)"
"$PY" -m webui
