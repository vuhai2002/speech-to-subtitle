"""Both transcription clients wait by retry_wait and read Retry-After from curl's header dump."""
from types import SimpleNamespace

from transcribe.chunked_transcribe import transcribe_client as router
from transcribe.mai_transcribe import transcribe_client as mai


def _router_res(http, text="", finish="", retry_after=None, errors=None):
    return {"text": text, "served_model": "", "finish": finish, "usage": None,
            "errors": errors or [], "reasoning_chunks": 0, "http": http, "retry_after": retry_after}


def test_router_sleeps_retry_after_on_429(monkeypatch):
    calls = iter([_router_res("429", errors=[{"code": 429}], retry_after="30"),
                  _router_res("200", text="xin chao " * 200, finish="stop")])
    monkeypatch.setattr(router, "transcribe_once", lambda *a, **k: next(calls))
    slept = []
    monkeypatch.setattr(router.time, "sleep", slept.append)
    r = router.transcribe_with_retry("x.mp3", "prompt", 100.0, "req.json", "raw.sse")
    assert r["reasons"] == [] and r["attempts"] == 2
    assert slept == [30.0]


def test_router_plain_failure_uses_jittered_backoff(monkeypatch):
    calls = iter([_router_res("500", errors=[{"code": 500}]),
                  _router_res("200", text="xin chao " * 200, finish="stop")])
    monkeypatch.setattr(router, "transcribe_once", lambda *a, **k: next(calls))
    slept = []
    monkeypatch.setattr(router.time, "sleep", slept.append)
    router.transcribe_with_retry("x.mp3", "prompt", 100.0, "req.json", "raw.sse")
    base = router.config.RETRY_BACKOFF_SEC
    assert len(slept) == 1 and 0.5 * base <= slept[0] < 1.5 * base


def test_router_once_dumps_headers_and_reads_retry_after(monkeypatch, tmp_path):
    mp3 = tmp_path / "00.mp3"
    mp3.write_bytes(b"ID3")
    monkeypatch.setattr(router.config, "BASE_URL", "https://router.invalid/v1")
    monkeypatch.setattr(router.config, "API_KEY", "k")
    seen = {}

    def fake_run(argv, capture_output, text):
        seen["argv"] = argv
        with open(argv[argv.index("-D") + 1], "w", encoding="utf-8") as f:
            f.write("HTTP/2 429\r\nRetry-After: 17\r\n\r\n")
        with open(argv[argv.index("-o") + 1], "w", encoding="utf-8") as f:
            f.write('{"error": {"message": "rate limited"}}')
        return SimpleNamespace(stdout="429", stderr="")

    monkeypatch.setattr(router.subprocess, "run", fake_run)
    res = router.transcribe_once(str(mp3), "prompt", str(tmp_path / "req.json"), str(tmp_path / "raw.sse"))
    assert "-D" in seen["argv"]
    assert res["http"] == "429" and res["retry_after"] == "17"


def test_router_once_drops_a_stale_header_dump(monkeypatch, tmp_path):
    # A previous attempt's 429 headers must not leak into an attempt whose curl wrote a body
    # but no header dump: the old file is removed before curl runs.
    mp3 = tmp_path / "00.mp3"
    mp3.write_bytes(b"ID3")
    raw = tmp_path / "raw.sse"
    (tmp_path / "raw.sse.headers").write_text("HTTP/2 429\r\nRetry-After: 99\r\n\r\n", encoding="utf-8")
    monkeypatch.setattr(router.config, "BASE_URL", "https://router.invalid/v1")
    monkeypatch.setattr(router.config, "API_KEY", "k")

    def fake_run(argv, capture_output, text):
        with open(argv[argv.index("-o") + 1], "w", encoding="utf-8") as f:
            f.write('data: {"choices": [{"delta": {"content": "xin chao"}, "finish_reason": "stop"}]}\n')
        return SimpleNamespace(stdout="200", stderr="")

    monkeypatch.setattr(router.subprocess, "run", fake_run)
    res = router.transcribe_once(str(mp3), "prompt", str(tmp_path / "req.json"), str(raw))
    assert res["http"] == "200"
    assert res.get("retry_after") is None


def _mai_res(http, text="", error=None, retry_after=None):
    return {"text": text, "words": [], "http": http, "error": error, "retry_after": retry_after}


def test_mai_sleeps_retry_after_on_429(monkeypatch):
    calls = iter([_mai_res("429", error="rate limited", retry_after="25"), _mai_res("200", text="xin chao")])
    monkeypatch.setattr(mai, "transcribe_once", lambda *a, **k: next(calls))
    slept = []
    monkeypatch.setattr(mai.time, "sleep", slept.append)
    r = mai.transcribe_with_retry("x.mp3", "raw.json", True)
    assert r["reasons"] == [] and r["attempts"] == 2
    assert slept == [25.0]


def test_mai_once_dumps_headers_and_reads_retry_after(monkeypatch, tmp_path):
    monkeypatch.setattr(mai.config, "API_KEY", "k")
    seen = {}

    def fake_run(argv, capture_output, text, encoding, errors):
        seen["argv"] = argv
        with open(argv[argv.index("-D") + 1], "w", encoding="utf-8") as f:
            f.write("HTTP/2 429\r\nRetry-After: 12\r\n\r\n")
        return SimpleNamespace(stdout='{"error": {"message": "slow down"}}\n|HTTP429')

    monkeypatch.setattr(mai.subprocess, "run", fake_run)
    res = mai.transcribe_once(str(tmp_path / "00.mp3"), str(tmp_path / "00.json"))
    assert "-D" in seen["argv"]
    assert res["http"] == "429" and res["retry_after"] == "12"
