"""Transcribe long audio with MAI-Transcribe-2 (Microsoft) via OpenRouter's transcription endpoint.

EXPERIMENTAL/FALLBACK route, distinct from the official step 1 (batch_transcribe_vertex.py via Vertex).
Unlike chunked_transcribe (9router/Antigravity): MAI is a dedicated ASR billed per second of audio
(~$0.10/hour, legitimate, no account-lock risk), and it returns PER-WORD TIMESTAMPS -> the MMS step can be skipped.

Modules:
  run_pipeline   audio -> ~10-minute chunks -> transcribe (parallel) -> raw_transcript.txt + mai_words.json
  build_srt      .srt from MAI's native timestamps (no MMS, no GPU)

Run from the repo root, e.g.:
  .venv-realign/Scripts/python.exe -m transcribe.mai_transcribe.run_pipeline --input <audio> --out-dir <dir>

Secrets are read from the environment variable OPENROUTER_API_KEY (see config.py). Notes + experimental
findings: docs/mai-transcribe-notes.md.
"""
