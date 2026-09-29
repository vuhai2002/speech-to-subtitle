"""Gemini's filter refusal is recognised and never retried."""
from transcribe.chunked_transcribe import transcribe_client as router

BLOCK_TEXT = ("This request was blocked by Gemini's filters. They can occasionally trigger by mistake on safe "
              "coding, security, or biology-related queries. Please try rephrasing your prompt.")


def _res(text, finish="", errors=None, http="200"):
    return {"text": text, "served_model": "gemini-3.8-flash", "finish": finish, "usage": None,
            "errors": errors or [], "reasoning_chunks": 0, "http": http, "retry_after": None}


def test_filter_block_is_the_refusal_text_without_finish():
    assert router.is_filter_block(_res(BLOCK_TEXT))


def test_filter_block_matches_a_curly_apostrophe_and_other_casing():
    assert router.is_filter_block(_res(BLOCK_TEXT.replace("'", "\u2019").upper()))


def test_real_transcript_that_ends_normally_is_not_a_block():
    assert not router.is_filter_block(_res("thầy nói về câu " + BLOCK_TEXT, finish="stop"))


def test_ordinary_truncated_reply_is_not_a_block():
    assert not router.is_filter_block(_res("xin chào các bạn", finish=""))


def test_block_stops_retrying_after_the_first_attempt(monkeypatch):
    calls = []
    monkeypatch.setattr(router, "transcribe_once", lambda *a, **k: calls.append(1) or _res(BLOCK_TEXT))
    slept = []
    monkeypatch.setattr(router.time, "sleep", slept.append)
    r = router.transcribe_with_retry("x.mp3", "prompt", 400.0, "req.json", "raw.sse")
    assert r["blocked"] is True and r["locked"] is False
    assert r["reasons"] == ["blocked by Gemini filters"] and r["attempts"] == 1
    assert len(calls) == 1 and slept == []


def test_normal_success_reports_not_blocked(monkeypatch):
    monkeypatch.setattr(router, "transcribe_once", lambda *a, **k: _res("xin chào " * 400, finish="stop"))
    r = router.transcribe_with_retry("x.mp3", "prompt", 400.0, "req.json", "raw.sse")
    assert r["blocked"] is False and r["reasons"] == []


def test_text_reasons_flags_empty_and_sparse_text_only_when_speech_is_present():
    assert router.text_reasons("", 400.0) == ["empty despite speech present"]
    assert router.text_reasons("xin chào", 400.0) == ["low word density (0/min)"]
    assert router.text_reasons("xin chào " * 400, 400.0) == []
    assert router.text_reasons("", 10.0) == []


def test_guard_reasons_still_reports_finish_and_density_in_order():
    assert router.guard_reasons(_res("xin chào"), 400.0) == ["finish=empty", "low word density (0/min)"]
