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


def transcribe_blocked_chunk(mp3_path: str, raw_path: str, chunk: dict) -> dict:
    """Returns {"ok", "text", "why"}: ok=True with MAI's text, or ok=False with a short reason for the manifest note."""
    if not mai_config.API_KEY:
        return {"ok": False, "text": "", "why": NO_KEY}
    r = mai_client.transcribe_with_retry(mp3_path, raw_path, expect_speech=audio_utils.chunk_has_speech(chunk))
    if r["reasons"]:
        return {"ok": False, "text": "", "why": FAILED + "; ".join(r["reasons"])}
    text = r["res"].get("text", "")
    weak = transcribe_client.text_reasons(text, chunk["speech_sec"])
    if weak:
        return {"ok": False, "text": "", "why": FAILED + "; ".join(weak)}
    return {"ok": True, "text": text, "why": ""}
