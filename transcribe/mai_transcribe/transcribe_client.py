"""Call OpenRouter's transcription endpoint (MAI-Transcribe-2) to transcribe one audio chunk.

Use curl multipart (file @mp3), response_format=verbose_json + timestamp_granularities[]=word
to get the transcript + PER-WORD timestamps (with punctuation). Retry on network error / empty / HTTP != 200.
"""
import json
import os
import subprocess
import time

from . import config
from transcribe.retry_wait import retry_after_from_headers, wait_seconds


def transcribe_once(mp3_path: str, raw_path: str) -> dict:
    """One call. Save the raw JSON to raw_path. Returns a parsed dict (does not raise).

    {text, words, duration, language, cost, seconds, http, error, retry_after}. words = [{word,start,end}].
    """
    if not config.API_KEY:
        raise RuntimeError("missing OPENROUTER_API_KEY in the environment")
    hdr_path = raw_path + ".headers"
    if os.path.exists(hdr_path):        # never read the previous attempt's Retry-After
        try:
            os.remove(hdr_path)
        except OSError:
            pass  # e.g. a Windows sharing violation; worst case one stale Retry-After wait
    out = subprocess.run(
        ["curl", "-s", "-m", str(config.HTTP_TIMEOUT_SEC), config.API_URL,
         "-H", "Authorization: Bearer " + config.API_KEY,
         "-F", "file=@" + mp3_path,
         "-F", "model=" + config.MODEL,
         "-F", "response_format=verbose_json",
         "-F", "timestamp_granularities[]=word",
         "-F", "language=" + config.LANGUAGE,
         "-D", hdr_path,
         "-w", "\n|HTTP%{http_code}"],
        capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    body, _, http = out.rpartition("|HTTP")
    retry_after = retry_after_from_headers(hdr_path)
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(body)
    try:
        j = json.loads(body)
    except Exception:
        return {"text": "", "words": [], "http": http.strip(), "error": "body is not JSON: " + body[:200],
                "retry_after": retry_after}
    if isinstance(j, dict) and j.get("error"):
        return {"text": "", "words": [], "http": http.strip(),
                "error": json.dumps(j["error"], ensure_ascii=False)[:300], "retry_after": retry_after}
    usage = j.get("usage") or {}
    return {"text": (j.get("text") or "").strip(), "words": j.get("words") or [],
            "duration": j.get("duration"), "language": j.get("language"),
            "cost": usage.get("cost"), "seconds": usage.get("seconds"),
            "http": http.strip(), "error": None, "retry_after": retry_after}


def transcribe_with_retry(mp3_path: str, raw_path: str, expect_speech: bool) -> dict:
    """Call + retry up to MAX_ATTEMPTS. Considered BAD when: there is an error, HTTP != 200, or empty despite speech present.

    Returns: {res, attempts, reasons, cost}. res is the dict from transcribe_once (last attempt); cost is the summed
    USD cost OpenRouter reported over every attempt (each answered attempt is billed), or None when none reported one.
    """
    res, reasons, cost = {}, [], None
    for att in range(1, config.MAX_ATTEMPTS + 1):
        res = transcribe_once(mp3_path, raw_path)
        if isinstance(res.get("cost"), (int, float)):
            cost = round((cost or 0.0) + res["cost"], 6)
        reasons = []
        if res.get("error"):
            reasons.append("error: " + res["error"][:80])
        if res.get("http") != "200":
            reasons.append("http=" + str(res.get("http")))
        if expect_speech and not res.get("text"):
            reasons.append("empty despite speech present")
        if not reasons:
            return {"res": res, "attempts": att, "reasons": [], "cost": cost}
        if att < config.MAX_ATTEMPTS:
            time.sleep(wait_seconds(att, config.RETRY_BACKOFF_SEC, res.get("http"), res.get("retry_after")))
    return {"res": res, "attempts": config.MAX_ATTEMPTS, "reasons": reasons, "cost": cost}
