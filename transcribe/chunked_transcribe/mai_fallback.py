"""Transcribe a chunk that Gemini's safety filter refused, with MAI-Transcribe-2 (OpenRouter).

Used ONLY for filter blocks (transcribe_client.is_filter_block); every other failure keeps its old
handling. Needs OPENROUTER_API_KEY in the environment: without it the chunk stays failed and the run is
reported incomplete (exit 3), exactly as before this fallback existed. MAI's text is held to the same bar
as Gemini's (transcribe_client.text_reasons) and is later timed by MMS like any other chunk.
"""
from transcribe.mai_transcribe import config as mai_config
from transcribe.mai_transcribe import transcribe_client as mai_client

from . import audio_utils, transcribe_client

NO_KEY = "no OpenRouter key for the MAI fallback"
FAILED = "MAI fallback failed: "


def _fail(reason: str) -> dict:
    """Build a failed fallback result. Whitespace in the reason is collapsed to one line first: curl/JSON
    errors from the MAI client can carry raw newlines (e.g. a non-JSON or empty HTTP body), which would
    otherwise split the manifest note and the printed chunk-progress line across lines."""
    return {"ok": False, "text": "", "why": FAILED + " ".join(reason.split())}


def transcribe_blocked_chunk(mp3_path: str, raw_path: str, chunk: dict) -> dict:
    """Returns {"ok", "text", "why"}: ok=True with MAI's text, or ok=False with a short reason for the manifest note."""
    if not mai_config.API_KEY:
        return {"ok": False, "text": "", "why": NO_KEY}
    try:
        r = mai_client.transcribe_with_retry(mp3_path, raw_path, expect_speech=audio_utils.chunk_has_speech(chunk))
    except Exception as e:
        # Report only the exception type, never str(e): subprocess failures (TimeoutExpired,
        # CalledProcessError) embed the full curl command line, including "Authorization: Bearer <key>".
        # An uncaught exception here would otherwise propagate out of _process_chunk and crash the
        # whole run instead of leaving this one chunk failed like every other bad-transcript case.
        return _fail(type(e).__name__)
    if r["reasons"]:
        return _fail("; ".join(r["reasons"]))
    text = r["res"].get("text", "")
    weak = transcribe_client.text_reasons(text, chunk["speech_sec"])
    if weak:
        return _fail("; ".join(weak))
    return {"ok": True, "text": text, "why": ""}
