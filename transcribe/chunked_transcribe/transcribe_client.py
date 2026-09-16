"""Call the router (OpenAI-compatible) to transcribe one chunk. Send audio as base64 via input_audio.

Use curl via subprocess: the router sits behind Cloudflare and default HTTP libraries are blocked (error 1010).
Read the SSE stream, accumulate content. Guard + retry live in this module.
"""
import base64
import json
import subprocess
import time

from . import config


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


def transcribe_once(mp3_path: str, prompt: str, req_path: str, raw_path: str) -> dict:
    """One call. Returns the parsed result dict + http_code (does not raise)."""
    if not config.BASE_URL:
        raise RuntimeError("missing ROUTER_BASE_URL in the environment")
    if not config.API_KEY:
        raise RuntimeError("missing ROUTER_API_KEY in the environment")
    with open(req_path, "w") as f:
        f.write(_payload(mp3_path, prompt))
    code = subprocess.run(
        ["curl", "-s", "-N", "-m", str(config.HTTP_TIMEOUT_SEC), config.BASE_URL + "/chat/completions",
         "-H", "Authorization: Bearer " + config.API_KEY, "-H", "Content-Type: application/json",
         "-H", "User-Agent: " + config.USER_AGENT, "--data-binary", "@" + req_path,
         "-o", raw_path, "-w", "%{http_code}"], capture_output=True, text=True).stdout.strip()
    res = _parse_sse(open(raw_path, encoding="utf-8", errors="replace").read())
    res["http"] = code
    return res


def guard_reasons(res: dict, speech_sec: float) -> list[str]:
    """Reasons a call is considered BAD (needs retry): error / finish!=stop / empty / low word density."""
    words = len(res["text"].split())
    density = words / (speech_sec / 60) if speech_sec > 0 else None
    reasons = []
    if res["errors"]:
        reasons.append("error")
    if res["finish"] != "stop":
        reasons.append(f"finish={res['finish'] or 'empty'}")
    if speech_sec >= config.MIN_SPEECH_CHUNK_SEC and not res["text"].strip():
        reasons.append("empty despite speech present")
    if (speech_sec >= config.MIN_SPEECH_CHUNK_SEC and res["text"].strip()
            and density is not None and density < config.MIN_WORDS_PER_SPEECH_MIN):
        reasons.append(f"low word density ({density:.0f}/min)")
    return reasons


def transcribe_with_retry(mp3_path: str, prompt: str, speech_sec: float, req_path: str, raw_path: str) -> dict:
    """Call + retry up to MAX_ATTEMPTS on failure. On 403 (account locked), STOP immediately, no retry.

    Returns: {res, attempts, reasons, locked}. locked=True means the account got a 403 -> the whole file should stop.
    """
    res, reasons = {}, []
    for att in range(1, config.MAX_ATTEMPTS + 1):
        res = transcribe_once(mp3_path, prompt, req_path, raw_path)
        if _is_403(res):
            return {"res": res, "attempts": att, "reasons": ["403 account locked"], "locked": True}
        reasons = guard_reasons(res, speech_sec)
        if not reasons:
            return {"res": res, "attempts": att, "reasons": [], "locked": False}
        if att < config.MAX_ATTEMPTS:
            time.sleep(config.RETRY_BACKOFF_SEC)
    return {"res": res, "attempts": config.MAX_ATTEMPTS, "reasons": reasons, "locked": False}
