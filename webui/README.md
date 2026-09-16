# webui

Local browser UI for the pipeline. Pick an audio file, choose a backend, run, watch progress,
and download `.srt`/`.txt`. It orchestrates the existing CLIs as subprocesses; it does not
reimplement any pipeline logic. See `docs/webui-design.md`.

## Run

    pip install -r requirements-webui.txt
    python -m webui           # opens http://127.0.0.1:8000

Bind stays on 127.0.0.1 (local only, no auth). API keys are entered in Settings and saved to
`.env` (gitignored). Runtime state and outputs live under `out/webui/`.

## Notes
- `.srt` from the Router/Vertex backends needs an NVIDIA GPU (MMS alignment). MAI produces
  `.srt` from native timestamps without a GPU.
- Prompt override applies to the Gemini backends (Vertex, Router); MAI is a pure ASR endpoint.
