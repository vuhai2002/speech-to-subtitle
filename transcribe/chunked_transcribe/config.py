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
RETRY_BACKOFF_SEC = 8.0         # base pause between retries; doubles per attempt with jitter (transcribe/retry_wait.py)
MIN_SPEECH_CHUNK_SEC = 30.0     # below this (and MIN_SPEECH_RATIO) VAD calls a chunk silent: its text is kept only where MMS scores it well
MIN_SPEECH_RATIO = 0.5          # ...unless speech fills at least this fraction of the chunk (keeps short, speech-dominated clips)
MIN_WORDS_PER_SPEECH_MIN = 60.0 # lower word density -> treated as an error, retry (catches a status message instead of a transcript)


# Mean MMS word score a sentence from a chunk VAD hears as silent needs to be kept as a subtitle line; lower
# sentences are dropped. 0.5 is the admin's choice (2026-10-01): in 41 passages the admin heard on
# 2026-09-29, every one scoring >= 0.457 was right and the highest wrong one scored 0.397.
def _silent_min_score(raw: str | None) -> float:
    """Parse SILENT_MIN_SCORE: strip whitespace, accept a comma decimal ("0,5"); empty or unset keeps 0.5.
    Anything that is not a number in (0, 1] is rejected loudly: a silently ignored bad value would misjudge
    every silent chunk of the run."""
    raw = (raw or "").strip()
    if not raw:
        return 0.5
    try:
        value = float(raw.replace(",", ".", 1))
    except ValueError:
        value = None
    if value is None or not (0 < value <= 1):
        raise ValueError(f"SILENT_MIN_SCORE must be a number greater than 0 and at most 1, got '{raw}'")
    return value


SILENT_MIN_SCORE = _silent_min_score(os.getenv("SILENT_MIN_SCORE"))

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
