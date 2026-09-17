# Web UI - design

A local web UI that wraps the existing pipeline so jobs can be run from a browser instead of the command line. It ships in this repository so anyone can pull it and run it locally. This document is the agreed design; the implementation plan is derived from it.

## Goal

Give a friendly local front door to the two-step pipeline (transcribe -> align) without changing the pipeline itself. Pick an audio file, choose a backend, choose the output, run, watch progress, download the result.

## Scope (first version)

In scope:

- Runner: pick a local audio file (drag-and-drop upload, or pick from a folder on the machine), choose a backend, choose output (`.srt` or `.txt`), run, watch live progress, preview and download results.
- Per-backend settings with API key / credential entry and a Test button.
- Editable transcription-prompt override for the Gemini backends.
- Jobs view with live progress and run history.

Out of scope for now:

- Subtitle editor / cue editing.
- Hosted or multi-user service, authentication.

## Principles

- The UI only orchestrates existing modules; it does not reimplement any pipeline logic.
- Run the existing CLIs as subprocesses (isolation, clean cancellation, reuse of the already-tested code paths).
- Bind to `127.0.0.1` only. Single local user, no auth.
- Keep the core dependency-light: UI-only dependencies live in `requirements-webui.txt`.
- Every source file stays under ~200 lines; comments and docs in English.

## Architecture

An application shell: a left sidebar for navigation, a top bar for global status (GPU) and the theme toggle, and a content area that swaps between views. It is a single static page served by a small FastAPI server.

Package layout (each file < 200 lines):

```
webui/
  __main__.py     python -m webui: parse --host/--port/--no-open, start uvicorn, open the browser
  server.py       FastAPI app: serve the page + APIs (list files, start job, SSE, download result, test key)
  jobs.py         Job + JobManager: run the subprocess chain in a thread, capture stdout, stream via SSE, stop, persist history
  backends.py     Registry: (backend, output) -> subprocess step chain; test() probes; GPU detection
  settings.py     Load/save config (keys, defaults) to .env; return current values
  static/index.html   app shell (sidebar + top bar + the four views)
  static/app.js       vanilla JS: view routing, forms, SSE client, Test buttons
  static/style.css    light/dark theme matching the project banner
  README.md
```

If `server.py` grows past the size limit, split the route groups into `routes_*.py`.

### Execution model: subprocess chains

The UI does not import the pipeline modules and mutate their config in-process. Instead `jobs.py` builds a command such as `python -m transcribe.<backend>.run_pipeline ...` with the right environment and arguments, runs it, and reads its stdout to stream progress. This gives isolation, makes Stop a clean process kill, and reuses the exact tested CLI paths.

## Views

### Run (default)

Pick a source: drag-and-drop (or click) to upload a file, or expand "pick from a folder on this machine" to select a server-side path without copying (good for very large files). Choose a backend, with a format hint for the accepted audio/video extensions. Choose output (`.srt` or `.txt`), gated by GPU (see the matrix). Optionally edit the prompt override, which is prefilled with the default (per run). Press Run to create a job and jump to the Jobs view.

### Jobs

A table of jobs (status, backend, output, file). Selecting a job opens its detail: a stage indicator derived from the log markers, the live log (SSE), and a Stop button while it runs. When finished, the results appear as file tabs - click a tab to preview that output in the browser (read-only), with Copy and Download for the selected file, and the Stop button is hidden. One job runs at a time, but history is recorded.

### Settings

This is where keys live, entered once and reused. One tab per backend (Router / MAI / Vertex), each with its fields, a masked API key, a Test button, and Save (persisted to `.env`). The light/dark theme toggle lives in the top bar.

### Docs

An in-app "How to use" page: configure a backend, run a job, watch and download, plus the GPU notes - so the basics are available without leaving the UI.

## Backend integration

Configuration mapping to the existing modules:

- Key / endpoint / model / workers / language: passed via the existing environment variables (`OPENROUTER_API_KEY`, `ROUTER_*`, `GOOGLE_*`) and CLI arguments (`--model`, `--workers`, `--input`, `--out-dir`). No code change.
- Prompt override: the single change to existing code. Add an optional `--prompt-file` argument to the two Gemini entrypoints (`transcribe/batch_transcribe_vertex.py` and `transcribe/chunked_transcribe/run_pipeline.py`); when absent, behaviour is unchanged (Vertex keeps its constant, chunked keeps `load_prompt()`). MAI has no prompt.

Backend x output matrix (a job chains these steps):

| Backend | `.txt` | `.srt` |
|---|---|---|
| MAI | run_pipeline | run_pipeline -> `mai_transcribe.build_srt` (no GPU; native timestamps) |
| Router | run_pipeline | run_pipeline -> `chunked_transcribe.build_srt` (GPU; per-chunk MMS align) |
| Vertex | transcribe | transcribe -> `realign.run_batch` (GPU) |

GPU detection and gating: detect once at startup with a small subprocess (so the web server never loads torch), cache the result, and show it in the top bar. `.srt` for Router/Vertex requires a GPU; without one the UI disables that combination with a tooltip and suggests `.txt` or the MAI backend.

Test buttons: light probes per backend (MAI/Router: a minimal request or a models call; Vertex: a service-account auth check), run through `backends.py`, returning ok/fail plus a short message.

## Job model, progress, history

- `Job`: id, params, status (queued / running / done / error), current stage (inferred from existing log markers such as `[1/4]`, `[3/4] transcribe...`, `align...`), a log buffer, and results (file paths + stats).
- Runs in a background thread, reads the subprocess stdout line by line, appends to the log and updates the stage, and pushes to SSE subscribers at `/jobs/{id}/events`.
- Stop: kill the current step's subprocess.
- History lives under the already-gitignored `out/`:
  - `out/webui/jobs.json` - the run history index.
  - `out/webui/<job-id>/` - that job's output files and log (or the output folder chosen in the Run view).
  No database; a JSON file is enough for a single local user.

## Dependencies and running

- New dependencies: `fastapi`, `uvicorn`, and `python-multipart` (drag-and-drop upload), kept in a separate `requirements-webui.txt` so the core pipeline stays light. SSE uses `StreamingResponse` (no extra dependency).
- Run: `python -m webui` opens `http://127.0.0.1:8000`. No build step; static files are served as-is.
- Environment: no new keys (reuse the existing ones). Add optional `WEBUI_HOST` / `WEBUI_PORT` to `.env.example`.

## Security

Loopback only, no auth (sufficient for a single local user). Keys are stored in `.env` (gitignored). The server never binds to a public interface.

## Testing

Pure-logic unit tests (no GPU, no network), following the repo's pytest convention:

- `backends.py`: the command-chain builder produces the correct command for each (backend, output).
- `settings.py`: `.env` read/write round-trip against a temp file.
- `jobs.py`: the stage-from-log-marker function; and a job lifecycle driven by a fake subprocess (a `python -c` that prints fake markers) to exercise capture, streaming, and stop without the real pipeline.
- Network Test probes are not unit-tested (or are mocked).

Front end: no JS test framework; manual smoke test.

## Change to existing code

Only one, backward-compatible: add an optional `--prompt-file` to `transcribe/batch_transcribe_vertex.py` and `transcribe/chunked_transcribe/run_pipeline.py`. Everything else is pure orchestration in the new `webui/` package.

## Future / not now

A real job queue with depth > 1, a subtitle editor, and single-executable packaging.
