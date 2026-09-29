"""A chunk Gemini's filter refuses is transcribed by MAI when a key is set; otherwise it fails like before."""
import threading

from transcribe.chunked_transcribe import mai_fallback, run_pipeline

BLOCKED = {"res": {"text": "This request was blocked by Gemini's filters.", "finish": "", "http": "200",
                   "served_model": "gemini-3.8-flash", "usage": None, "errors": []},
           "attempts": 1, "reasons": ["blocked by Gemini filters"], "locked": False, "blocked": True}
NOT_BLOCKED_FAILURE = {"res": {"text": "", "finish": "", "http": "500", "served_model": "m", "usage": None,
                               "errors": []},
                        "attempts": 3, "reasons": ["http=500"], "locked": False, "blocked": False}
CHUNK = {"idx": 7, "start": 3600.0, "end": 4200.0, "speech_sec": 400.0, "cut_gap_sec": 1.5}
MAI_TEXT = "ngay cả những người xấu cũng cần tấm gương trong sạch " * 60


def _setup(monkeypatch, tmp_path, key="k", mai_result=None, router_result=None):
    (tmp_path / "chunks").mkdir()
    monkeypatch.setattr(run_pipeline.audio_utils, "cut_chunk", lambda *a, **k: None)
    monkeypatch.setattr(run_pipeline.transcribe_client, "transcribe_with_retry",
                         lambda *a, **k: router_result if router_result is not None else BLOCKED)
    monkeypatch.setattr(mai_fallback.mai_config, "API_KEY", key)
    calls = []

    def fake_mai(mp3, raw, expect_speech):
        calls.append((mp3, raw, expect_speech))
        return mai_result or {"res": {"text": MAI_TEXT}, "attempts": 1, "reasons": []}

    monkeypatch.setattr(mai_fallback.mai_client, "transcribe_with_retry", fake_mai)
    return calls


def _process(tmp_path, chunk=CHUNK):
    return run_pipeline._process_chunk(chunk, tmp_path, str(tmp_path / "mono16k.mp3"), "prompt", threading.Lock())


def test_blocked_chunk_is_transcribed_by_mai(monkeypatch, tmp_path, capsys):
    calls = _setup(monkeypatch, tmp_path)
    rec = _process(tmp_path)
    assert rec["included"] is True and rec["engine"] == "mai"
    assert rec["note"] == "mai fallback: gemini blocked by filters"
    assert (tmp_path / "chunks" / "07.txt").read_text(encoding="utf-8") == MAI_TEXT
    assert calls == [(str(tmp_path / "chunks" / "07.mp3"), str(tmp_path / "chunks" / "07_mai.json"), True)]
    line = capsys.readouterr().out
    assert line.startswith("    chunk 07 ")
    assert line.rstrip().endswith("MERGED | MAI (gemini blocked)")


def test_blocked_chunk_without_key_fails_and_never_calls_mai(monkeypatch, tmp_path):
    calls = _setup(monkeypatch, tmp_path, key="")
    rec = _process(tmp_path)
    assert rec["included"] is False and "engine" not in rec
    assert rec["note"] == "error after retry: blocked by Gemini filters; no OpenRouter key for the MAI fallback"
    assert calls == []


def test_mai_error_keeps_the_chunk_failed(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, mai_result={"res": {"text": ""}, "attempts": 3, "reasons": ["http=402"]})
    rec = _process(tmp_path)
    assert rec["included"] is False and "engine" not in rec
    assert rec["note"] == "error after retry: blocked by Gemini filters; MAI fallback failed: http=402"


def test_mai_exception_keeps_the_chunk_failed_instead_of_crashing_the_run(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)

    def raising_mai(mp3, raw, expect_speech):
        raise RuntimeError("boom")

    monkeypatch.setattr(mai_fallback.mai_client, "transcribe_with_retry", raising_mai)
    rec = _process(tmp_path)
    assert rec["included"] is False and "engine" not in rec
    assert rec["note"] == "error after retry: blocked by Gemini filters; MAI fallback failed: RuntimeError"


def test_mai_failure_reason_whitespace_is_collapsed_to_one_line(monkeypatch, tmp_path, capsys):
    _setup(monkeypatch, tmp_path, mai_result={"res": {"text": ""}, "attempts": 3,
                                               "reasons": ["error: body is not JSON: \n", "http=000"]})
    rec = _process(tmp_path)
    assert rec["included"] is False and "engine" not in rec
    assert "\n" not in rec["note"]
    assert capsys.readouterr().out.count("\n") == 1     # one printed chunk line, not split by the raw error body


def test_mai_text_too_sparse_for_the_speech_keeps_the_chunk_failed(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, mai_result={"res": {"text": "xin chào"}, "attempts": 1, "reasons": []})
    rec = _process(tmp_path)
    assert rec["included"] is False
    assert rec["note"].startswith("error after retry: blocked by Gemini filters; MAI fallback failed: low word density")


def test_blocked_chunk_without_speech_follows_the_speech_rule(monkeypatch, tmp_path):
    calls = _setup(monkeypatch, tmp_path)
    rec = _process(tmp_path, {**CHUNK, "speech_sec": 0.0})
    assert rec["included"] is False and rec["engine"] == "mai"
    assert rec["note"] == "fabricated (chunk has no speech but returned words)"
    assert calls[0][2] is False           # MAI is not asked to find speech in a chunk VAD calls silent


def test_mai_expect_speech_uses_geminis_short_chunk_exemption(monkeypatch, tmp_path):
    # A 50s chunk with 28s speech (56%) is "has speech" by chunk_has_speech's ratio rule, but Gemini's own
    # text guard exempts anything under MIN_SPEECH_CHUNK_SEC (30s) from the empty/low-density check. MAI
    # must be held to that same floor, or a legitimately-short empty reply gets billed for 3 retries and
    # fails a chunk that an equivalent empty Gemini reply (finish=stop) would have accepted.
    short_chunk = {"idx": 7, "start": 0.0, "end": 50.0, "speech_sec": 28.0, "cut_gap_sec": 1.5}
    calls = _setup(monkeypatch, tmp_path, mai_result={"res": {"text": ""}, "attempts": 1, "reasons": []})
    rec = _process(tmp_path, short_chunk)
    assert calls[0][2] is False
    assert rec["included"] is True and rec["engine"] == "mai"


def test_unblocked_chunk_has_no_engine_and_never_calls_mai(monkeypatch, tmp_path):
    calls = _setup(monkeypatch, tmp_path)
    ok = {"res": {"text": MAI_TEXT, "finish": "stop", "http": "200", "served_model": "m", "usage": None,
                  "errors": []},
          "attempts": 1, "reasons": [], "locked": False, "blocked": False}
    monkeypatch.setattr(run_pipeline.transcribe_client, "transcribe_with_retry", lambda *a, **k: ok)
    rec = _process(tmp_path)
    assert rec["included"] is True and "engine" not in rec and rec["note"] == ""
    assert calls == []


def test_non_blocked_failure_never_calls_mai(monkeypatch, tmp_path):
    # Pins the scope decision: MAI is a fallback ONLY for a Gemini filter block (r["blocked"]), never
    # for an ordinary post-retry failure such as a plain HTTP error.
    calls = _setup(monkeypatch, tmp_path, router_result=NOT_BLOCKED_FAILURE)
    rec = _process(tmp_path)
    assert rec["included"] is False and "engine" not in rec
    assert rec["note"] == "error after retry: http=500"
    assert calls == []
