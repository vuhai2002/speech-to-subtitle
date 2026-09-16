"""Parameters for the MAI-Transcribe-2 transcription route (OpenRouter). Pure code, no AI in the loop.

Secrets are read from the environment, NOT hardcoded:
  OPENROUTER_API_KEY   OpenRouter api key (in the form sk-or-v1-...)

The chunk-cutting stage (mono16k, VAD, cut plan) REUSES chunked_transcribe.audio_utils, so the chunk-cutting
constants (SAMPLE_RATE, CHUNK_TARGET_SEC...) come from transcribe.chunked_transcribe.config - a single source.
"""
import os
from pathlib import Path

# --- endpoint / model ---
# OpenRouter's DEDICATED transcription endpoint (billed per second of audio), NOT chat/completions
# (chat counts base64 as text tokens, ~40x more expensive - see docs/mai-transcribe-notes.md).
API_URL = os.getenv("OPENROUTER_TRANSCRIBE_URL", "https://openrouter.ai/api/v1/audio/transcriptions")
API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = os.getenv("MAI_MODEL", "microsoft/mai-transcribe-2")
LANGUAGE = os.getenv("MAI_LANGUAGE", "vi")           # language hint; leave empty to let the model auto-detect
HTTP_TIMEOUT_SEC = 600

# --- parallel sending + retry ---
# Number of chunks sent in parallel. OpenRouter is a stable paid endpoint; 3 threads is enough and safe.
CONCURRENCY = int(os.getenv("MAI_WORKERS", "3"))
MAX_ATTEMPTS = 3                # retries per chunk on network error / empty / HTTP != 200
RETRY_BACKOFF_SEC = 5.0         # pause between retries

# PROJECT_ROOT derived from the file location (mai_transcribe -> transcribe -> repo root), not hardcoded
PROJECT_ROOT = Path(__file__).resolve().parents[2]
