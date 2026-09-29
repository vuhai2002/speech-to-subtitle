"""Call the router (OpenAI-compatible) to transcribe one chunk. Send audio as base64 via input_audio.

Use curl via subprocess: the router sits behind Cloudflare and default HTTP libraries are blocked (error 1010).
Read the SSE stream, accumulate content. Guard + retry live in this module.
"""
import base64
import json
import os
import subprocess
import time

from . import config
from transcribe.retry_wait import retry_after_from_headers, wait_seconds


def _payload(mp3_path: str, prompt: str) -> str:
    b64 = base64.b64encode(open(mp3_path, "rb").read()).decode()
    return json.dumps({
        "model": config.MODEL, "stream": True, "max_tokens": config.MAX_TOKENS,
        "messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": b64, "format": "mp3"}},
            {"type": "text", "text": prompt}]}]})


def _parse_sse(raw: str) -> dict:
    text = model = finish = ""
    usage = None
    errs = []
    reasoning_chunks = 0
    for ln in raw.splitlines():
        if not ln.startswith("data:"):
            continue
        p = ln[5:].strip()
        if p in ("", "[DONE]"):
            continue
        try:
            j = json.loads(p)
        except json.JSONDecodeError:
            continue
        if "error" in j:
            errs.append(j["error"])
            continue
        model = j.get("model", model)
        usage = j.get("usage") or usage
        for ch in j.get("choices", []):
            d = ch.get("delta", {}) or {}
            text += d.get("content") or ""
            reasoning_chunks += 1 if d.get("reasoning_content") else 0
            finish = ch.get("finish_reason") or finish
    if raw.strip() and not raw.lstrip().startswith("data:"):
        errs.append({"non_sse_body": raw[:300]})
    return {"text": text, "served_model": model, "finish": finish, "usage": usage,
            "errors": errs, "reasoning_chunks": reasoning_chunks}


def _is_403(res: dict) -> bool:
    return any("403" in json.dumps(e) for e in res["errors"])


def _transport_fail(detail: str, code: str) -> dict:
    """Shape a transport failure like a parsed SSE result so the retry/guard logic treats it
    as a retryable error, instead of the caller crashing on the missing response body."""
    return {"text": "", "served_model": "", "finish": "", "usage": None,
            "errors": [{"transport": detail}], "reasoning_chunks": 0, "http": code}


def transcribe_once(mp3_path: str, prompt: str, req_path: str, raw_path: str) -> dict:
    """One call. Returns the parsed result dict + http_code (does not raise)."""
    if not config.BASE_URL:
        raise RuntimeError("missing ROUTER_BASE_URL in the environment")
    if not config.API_KEY:
        raise RuntimeError("missing ROUTER_API_KEY in the environment")
    with open(req_path, "w") as f:
        f.write(_payload(mp3_path, prompt))
    hdr_path = raw_path + ".headers"
    if os.path.exists(hdr_path):        # never read the previous attempt's Retry-After
        try:
            os.remove(hdr_path)
        except OSError:
            pass  # e.g. a Windows sharing violation; worst case one stale Retry-After wait
    try:
        proc = subprocess.run(
            ["curl", "-s", "-N", "-m", str(config.HTTP_TIMEOUT_SEC), config.BASE_URL + "/chat/completions",
             "-H", "Authorization: Bearer " + config.API_KEY, "-H", "Content-Type: application/json",
             "-H", "User-Agent: " + config.USER_AGENT, "--data-binary", "@" + req_path,
             "-D", hdr_path, "-o", raw_path, "-w", "%{http_code}"], capture_output=True, text=True)
    except FileNotFoundError:
        return _transport_fail("curl not found on PATH", "000")
    code = proc.stdout.strip()
    if not os.path.exists(raw_path):
        # A connection-level failure (DNS/TLS/refused/timeout, e.g. the router host is down)
        # makes curl write no -o file, so reading it would raise FileNotFoundError. Report a
        # clear, retryable transport error instead.
        detail = (proc.stderr or "").strip()[:160] or f"router unreachable (http {code or '000'})"
        return _transport_fail(detail, code or "000")
    res = _parse_sse(open(raw_path, encoding="utf-8", errors="replace").read())
    res["http"] = code
    res["retry_after"] = retry_after_from_headers(hdr_path)
    return res


# Gemini's safety filter answers some chunks with a fixed refusal streamed as ordinary content and no
# finish_reason ("This request was blocked by Gemini's filters. ..."). It is deterministic for the same
# audio, so retrying the same chunk never helps.
FILTER_BLOCK_MARKER = "blocked by gemini's filters"
BLOCKED_REASON = "blocked by Gemini filters"


def is_filter_block(res: dict) -> bool:
    """True when the reply is Gemini's filter refusal rather than a transcript.

    Both conditions are required, so a real transcript that happens to contain these words (and ends
    normally with finish_reason=stop) is never mistaken for a block.
    """
    text = (res.get("text") or "").lower().replace("\u2019", "'")
    return not res.get("finish") and FILTER_BLOCK_MARKER in text


def text_reasons(text: str, speech_sec: float) -> list[str]:
    """Reasons a transcript text is unusable for a chunk: empty, or too sparse for the speech it holds.

    Shared by the Gemini guard and the MAI fallback so both engines are held to the same bar.
    """
    if speech_sec < config.MIN_SPEECH_CHUNK_SEC:
        return []
    if not text.strip():
        return ["empty despite speech present"]
    density = len(text.split()) / (speech_sec / 60)
    if density < config.MIN_WORDS_PER_SPEECH_MIN:
        return [f"low word density ({density:.0f}/min)"]
    return []


def guard_reasons(res: dict, speech_sec: float) -> list[str]:
    """Reasons a call is considered BAD (needs retry): error / finish!=stop / empty / low word density."""
    reasons = []
    if res["errors"]:
        reasons.append("error")
    if res["finish"] != "stop":
        reasons.append(f"finish={res['finish'] or 'empty'}")
    return reasons + text_reasons(res["text"], speech_sec)


def transcribe_with_retry(mp3_path: str, prompt: str, speech_sec: float, req_path: str, raw_path: str) -> dict:
    """Call + retry up to MAX_ATTEMPTS on failure. On 403 (account locked) or a Gemini filter block, STOP
    immediately, no retry: both are deterministic.

    Returns: {res, attempts, reasons, locked, blocked}. locked=True means the account got a 403 -> the whole
    file should stop; blocked=True means Gemini's safety filter refused this chunk (see is_filter_block).
    """
    res, reasons = {}, []
    for att in range(1, config.MAX_ATTEMPTS + 1):
        res = transcribe_once(mp3_path, prompt, req_path, raw_path)
        if _is_403(res):
            return {"res": res, "attempts": att, "reasons": ["403 account locked"], "locked": True, "blocked": False}
        if is_filter_block(res):
            return {"res": res, "attempts": att, "reasons": [BLOCKED_REASON], "locked": False, "blocked": True}
        reasons = guard_reasons(res, speech_sec)
        if not reasons:
            return {"res": res, "attempts": att, "reasons": [], "locked": False, "blocked": False}
        if att < config.MAX_ATTEMPTS:
            time.sleep(wait_seconds(att, config.RETRY_BACKOFF_SEC, res.get("http"), res.get("retry_after")))
    return {"res": res, "attempts": config.MAX_ATTEMPTS, "reasons": reasons, "locked": False, "blocked": False}
