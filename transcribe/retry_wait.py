"""How long to wait before the next transcription attempt. Shared by the router and MAI clients.

A fixed pause made every job that hit a rate limit together retry together and fail together.
Doubling with jitter spreads them out; a Retry-After the server sends on 429/503 is honoured
when it asks for longer.
"""
import random

WAIT_CAP_SEC = 120.0


def wait_seconds(attempt: int, base: float, http: str | None, retry_after: str | None,
                 rand=random.random, cap: float = WAIT_CAP_SEC) -> float:
    """Seconds to sleep after failed attempt number `attempt` (1-based)."""
    wait = min(cap, base * 2 ** (attempt - 1) * (0.5 + rand()))
    if http in ("429", "503") and retry_after:
        try:
            asked = float(retry_after.strip())
        except ValueError:          # HTTP-date form: keep the backoff
            asked = None
        if asked is not None and asked > wait:
            wait = min(cap, asked)
    return wait


def retry_after_from_headers(headers_path: str) -> str | None:
    """Last Retry-After value in a curl -D header dump, or None.

    A dump can hold more than one header block: curl runs without -L (no redirects followed),
    so the multi-block case is 1xx interim responses (e.g. 100 Continue on a request body over
    1 MB), not a redirect.
    """
    try:
        with open(headers_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return None
    value = None
    for line in text.splitlines():
        name, sep, rest = line.partition(":")
        if sep and name.strip().lower() == "retry-after":
            value = rest.strip()
    return value
