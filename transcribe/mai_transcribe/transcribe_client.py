"""Call OpenRouter's transcription endpoint (MAI-Transcribe-2) to transcribe one audio chunk.

Use curl multipart (file @mp3), response_format=verbose_json + timestamp_granularities[]=word
to get the transcript + PER-WORD timestamps (with punctuation). Retry on network error / empty / HTTP != 200.
"""
import json
import subprocess
import time

from . import config


def transcribe_once(mp3_path: str, raw_path: str) -> dict:
    """One call. Save the raw JSON to raw_path. Returns a parsed dict (does not raise).

    {text, words, duration, language, cost, seconds, http, error}. words = [{word,start,end}].
    """
    if not config.API_KEY:
        raise RuntimeError("missing OPENROUTER_API_KEY in the environment")
    out = subprocess.run(
        ["curl", "-s", "-m", str(config.HTTP_TIMEOUT_SEC), config.API_URL,
         "-H", "Authorization: Bearer " + config.API_KEY,
         "-F", "file=@" + mp3_path,
         "-F", "model=" + config.MODEL,
         "-F", "response_format=verbose_json",
         "-F", "timestamp_granularities[]=word",
         "-F", "language=" + config.LANGUAGE,
         "-w", "\n|HTTP%{http_code}"],
        capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    body, _, http = out.rpartition("|HTTP")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(body)
    try:
        j = json.loads(body)
    except Exception:
        return {"text": "", "words": [], "http": http.strip(), "error": "body is not JSON: " + body[:200]}
    if isinstance(j, dict) and j.get("error"):
        return {"text": "", "words": [], "http": http.strip(),
                "error": json.dumps(j["error"], ensure_ascii=False)[:300]}
    usage = j.get("usage") or {}
    return {"text": (j.get("text") or "").strip(), "words": j.get("words") or [],
            "duration": j.get("duration"), "language": j.get("language"),
            "cost": usage.get("cost"), "seconds": usage.get("seconds"),
            "http": http.strip(), "error": None}


def transcribe_with_retry(mp3_path: str, raw_path: str, expect_speech: bool) -> dict:
    """Call + retry up to MAX_ATTEMPTS. Considered BAD when: there is an error, HTTP != 200, or empty despite speech present.

    Returns: {res, attempts, reasons}. res is the dict from transcribe_once (last attempt).
    """
    res, reasons = {}, []
    for att in range(1, config.MAX_ATTEMPTS + 1):
        res = transcribe_once(mp3_path, raw_path)
        reasons = []
        if res.get("error"):
            reasons.append("error: " + res["error"][:80])
        if res.get("http") != "200":
            reasons.append("http=" + str(res.get("http")))
        if expect_speech and not res.get("text"):
            reasons.append("empty despite speech present")
        if not reasons:
            return {"res": res, "attempts": att, "reasons": []}
        if att < config.MAX_ATTEMPTS:
            time.sleep(config.RETRY_BACKOFF_SEC)
    return {"res": res, "attempts": config.MAX_ATTEMPTS, "reasons": reasons}
