"""Transcribe audio in ~10-minute chunks via an OpenAI-compatible router (9router/Antigravity).

EXPERIMENTAL route, distinct from the official step 1 (batch_transcribe_vertex.py via Vertex). See README.md.

Modules:
  run_pipeline   audio -> chunks -> transcribe -> merge -> raw_transcript.txt + manifest.json
  quality_check  audio-grounded self-check (MMS + star token) -> qc_report.json
  compare_passes compare 2 independent transcripts (2 passes / 2 models) -> list of differing spots
  render_compare_html  export a side-by-side comparison HTML, highlighting differences

Run from the repo root, e.g.:
  .venv/Scripts/python.exe -m transcribe.chunked_transcribe.run_pipeline --input <audio> --out-dir <dir>

Secrets are read from the environment variables ROUTER_BASE_URL / ROUTER_API_KEY (see config.py).
"""
