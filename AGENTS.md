# AGENTS.md

Rules for AI agents working in this repo. Read `README.md` to understand the flow and how to run things. Read `docs/system-architecture.md` before changing anything in `realign/`.

## Context

- The repo generates Vietnamese `.srt` subtitles for a lecture-video streaming platform. It does not write to any database. Importing the output into production happens in a separate downstream system that is not part of this repo.
- Two steps: `transcribe/` (mp3 -> txt, Gemini Vertex) then `realign/` (txt/srt + mp3 -> srt, MMS forced alignment).
- A local web UI in `webui/` wraps both steps (start it with `run-webui.bat` / `run-webui.sh`, or `.venv/Scripts/python.exe -m webui`). It orchestrates the existing CLIs as subprocesses and does not reimplement pipeline logic. See `webui/README.md` and `docs/webui-design.md`.
- The content is Vietnamese lectures. Preserve Vietnamese diacritics in code, comments, and data. Output srt filenames must preserve every character of the original transcript name, because the downstream system matches on the filename during import.

## Three transcription paths (step 1)

- **`transcribe/batch_transcribe_vertex.py` (Vertex) is the official path.** Gemini via Vertex, files sent by GCS link, billed to Google Cloud. Use this for real batches.
- **`transcribe/chunked_transcribe/` (9router) is an experimental path.** Gemini via 9router/Antigravity over OAuth (free). Run `-m transcribe.chunked_transcribe.run_pipeline`. Read `docs/chunked-transcribe-notes.md` and the package README before changing it. Key rules:
  - You must **split into roughly 10-minute chunks**. Sending a whole long file makes the model **silently drop** segments (verified) and exceeds Cloudflare's 100 MB body limit.
  - After transcribing, verify with `quality_check` (MMS + star token, catches omissions/hallucinations) and `compare_passes` (two-pass diff, catches wrong words). The MMS alignment score alone does NOT catch omissions.
  - **Do not use OpenRouter chat/completions for audio**: it bills base64 as text tokens, roughly 40x more expensive. (OpenRouter's dedicated transcription endpoint bills per second - see the MAI path.)
  - This is OAuth over a proxy: there is a risk Google bans the account, and Pro has a weekly quota plus bans of up to 7 days. Don't pile on many parallel workers (default 3). Large batches must account for the weekly quota; see the notes.
- **`transcribe/mai_transcribe/` (MAI-Transcribe-2 via OpenRouter) is an experimental/fallback path.** Microsoft's dedicated ASR, billed per second of audio (~$0.10/hour, legitimate, **no account-ban risk**). Run `-m transcribe.mai_transcribe.run_pipeline`. Read `docs/mai-transcribe-notes.md` and the package README before changing it. Key rules:
  - Use the **transcription endpoint** `/audio/transcriptions` (multipart, verbose_json, word timestamps), **not** chat/completions. Limit of 25 MB per request.
  - Still **split into 10-minute chunks** (shared `chunked_transcribe.audio_utils`), to guard against omissions and the 25 MB limit.
  - MAI returns **per-word timestamps with punctuation**, so `build_srt` builds subtitles directly, **skipping the MMS step, no GPU needed**. The SRT was verified equivalent to Gemini+MMS (97% word agreement with Gemini).
  - Requires the `OPENROUTER_API_KEY` environment variable (do not hardcode). Do not commit secrets.
- **To get subtitle timing via the Gemini/9router path: align PER CHUNK and add the offset, do NOT use `realign.window_align` on the whole file** (it assigns wrong times when the end of the file has a long non-speech segment). The MAI path already has timing and needs no alignment.

## Environment

- All Python commands for `realign/`, `tests/`, and the web UI: `.venv/Scripts/python.exe` (Python 3.10-3.13 with a CUDA build of `torch==2.5.1` + `torchaudio==2.5.1`). The `run-webui.bat` / `run-webui.sh` launcher creates it: it picks a compatible Python and installs the CUDA torch when an NVIDIA GPU is present. The tested torch has no wheels for Python 3.14+, so the venv must use 3.10-3.13.
- Run modules with `-m` from the repo root, e.g. `.venv/Scripts/python.exe -m realign.run_batch ...`. Calling by file path (`python realign/run_batch.py`) fails with `ModuleNotFoundError: realign`.
- `transcribe/batch_transcribe_vertex.py` runs separately (usually on a Linux VM), needs `.env` and a service account key, and calls a billed API. `.venv` does not have the Google libraries installed.

## Do not

- Do not rename or move `out/` or the files in it. Downstream tooling reads hardcoded absolute paths to this repo's `out/srt` and `out/unmatched-srt-reconcile.md`.
- Do not move `realign/`: downstream tooling and every `-m realign.*` command depend on this location.
- Do not run a full-directory align batch on your own. A batch runs for tens of hours, and an agent session's background process gets stopped partway by the system. Hand the command to the user to run in their own terminal.
- Do not run `transcribe/batch_transcribe_vertex.py` or the scripts under `tools/drive-batch-prep/` on your own (they cost money, or actually delete files on Google Drive) unless explicitly asked.
- Do not bulk-delete `out/words/*.json`: that is the alignment cache, and each file takes several minutes of GPU to regenerate.
- Do not commit `out/`, `test-files/`, `.env`, `service-account-key.json`, or `*.log`.
- Do not write to production. Any import into the downstream system must be run as a dry run first and reviewed by the user.

## Testing

- pytest, configured in `pytest.ini` (`pythonpath = .`, `testpaths = tests`). Install the test deps once: `.venv/Scripts/python.exe -m pip install -r requirements-dev.txt` (pytest is not in the app venv by default).
- When you change a module, run only its corresponding test file, e.g. `.venv/Scripts/python.exe -m pytest tests/realign/test_cue_builder.py`. Run the full suite only once, before reporting completion.
- Import `torch`, `torchaudio`, `silero_vad` only inside the functions that need them, not at the top of the module. This lets pure-logic tests run without a GPU. `test_window_align.py` is the exception - it needs torch.
- Verify GPU-dependent parts (align, VAD) with a smoke test on one short file in `test-files/` and `--max-sec`; do not run a full batch.

## Code conventions

- Python uses snake_case for filenames and modules (so they're importable); shell scripts use kebab-case. Keep each file under 200 lines, with type hints on public functions.
- Every subtitle-rule threshold lives in `realign/config.py` and is passed through the `cfg` parameter so tests can override it. Do not hardcode numbers in the logic.
- The data contracts between steps (Word, words.json, Cue) are described in `docs/system-architecture.md`. If you change the words.json contract, state clearly that the old cache must be re-aligned (`--force`).
- Changing cue rules only requires rerunning the segment step (cheap), not re-aligning.
- Comments and docstrings are written in Vietnamese with diacritics, using ASCII punctuation.
- Do not put plan IDs, phase numbers, or finding codes in filenames, comments, test names, or commit messages.
