"""Parameters + constants for the chunked transcription pipeline (pure code, no AI in the loop).

Secrets (endpoint + api key) are read from the environment, NOT hardcoded:
  ROUTER_BASE_URL  e.g. https://<router-host>/v1 (self-hosted OpenAI-compatible router instance, e.g. 9router)
  ROUTER_API_KEY   the router api key
"""
import ast
import os
from pathlib import Path

# --- router / model ---
BASE_URL = os.getenv("ROUTER_BASE_URL", "")  # must be set via env, no default endpoint shipped
API_KEY = os.getenv("ROUTER_API_KEY", "")
MODEL = os.getenv("TRANSCRIBE_MODEL", "ag/gemini-3.8-flash")
# Browser UA: the router sits behind Cloudflare, and default HTTP-library UAs are blocked (error 1010)
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/140.0 Safari/537.36"
HTTP_TIMEOUT_SEC = 900
MAX_TOKENS = 65536

# --- chunk cutting ---
SAMPLE_RATE = 16000
CHUNK_TARGET_SEC = 600.0        # desired chunk length (~10 minutes)
CUT_SEARCH_SEC = 90.0           # search for a silence to cut within +- this window around the target mark
MIN_CHUNK_SEC = 300.0           # two consecutive cut points are at least this far apart

# --- parallel sending + guard (retry / drop chunk) ---
# Number of chunks sent in parallel. Antigravity Pro handles 3-5; free tier 2-3. Override via ROUTER_WORKERS.
CONCURRENCY = int(os.getenv("ROUTER_WORKERS", "3"))
MAX_ATTEMPTS = 3                # retries per chunk on error/empty/finish!=stop/low density
RETRY_BACKOFF_SEC = 8.0         # pause between retries (gentle on rate limits)
MIN_SPEECH_CHUNK_SEC = 30.0     # chunk with less speech than this -> not merged (avoids hallucinating over music/silence)
MIN_WORDS_PER_SPEECH_MIN = 60.0 # lower word density -> treated as an error, retry (catches a status message instead of a transcript)

# --- self-check (quality_check, pure code, grounded in the audio) ---
STAR_EVERY_WORD = False         # insert a '*' token after punctuation (False) or after every word (True)
OMISSION_FLAG_SEC = 5.0         # a star slot holding >= this many seconds of speech -> FLAG OMISSION
OMISSION_REVIEW_SEC = 2.0       # a star slot holding >= this threshold (but < FLAG) -> needs re-listening
HALLUCINATION_MIN_RUN = 10      # >= this many consecutive words in a VAD-silent region -> suspected HALLUCINATION
SILENCE_PAD_SEC = 0.3           # padding around a word when checking whether it falls into silence

# --- prompt: taken from the project's own Vertex script to keep a single source of truth ---
# PROJECT_ROOT derived from the file location (chunked_transcribe -> transcribe -> repo root), not hardcoded
PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERTEX_SCRIPT = PROJECT_ROOT / "transcribe" / "batch_transcribe_vertex.py"


def load_prompt(prompt_file: str | None = None) -> str:
    """Return the transcription prompt. If prompt_file is given, read it; otherwise read
    TRANSCRIBE_PROMPT from the Vertex script (single source of truth)."""
    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8")
    src = VERTEX_SCRIPT.read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "TRANSCRIBE_PROMPT" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"TRANSCRIBE_PROMPT not found in {VERTEX_SCRIPT}")
