"""Gemini's filter refusal is recognised (both forms) and retried like any other failure."""
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


def test_content_filter_stop_is_a_block():
    assert router.is_filter_block(_res("thầy nói rằng", finish="content_filter"))


def test_block_is_retried_then_reported_blocked(monkeypatch):
    calls = []
    monkeypatch.setattr(router, "transcribe_once", lambda *a, **k: calls.append(1) or _res(BLOCK_TEXT))
    slept = []
    monkeypatch.setattr(router.time, "sleep", slept.append)
    r = router.transcribe_with_retry("x.mp3", "prompt", 400.0, "req.json", "raw.sse")
    assert r["blocked"] is True and r["locked"] is False
    assert r["reasons"] == ["blocked by Gemini filters"] and r["attempts"] == 3
    assert len(calls) == 3 and len(slept) == 2
    assert [h["outcome"] for h in r["history"]] == ["blocked", "blocked", "blocked"]
    assert [h["words"] for h in r["history"]] == [0, 0, 0]


def test_block_then_transcript_passes_on_retry(monkeypatch):
    replies = iter([_res(BLOCK_TEXT), _res("xin chào " * 400, finish="stop")])
    monkeypatch.setattr(router, "transcribe_once", lambda *a, **k: next(replies))
    monkeypatch.setattr(router.time, "sleep", lambda s: None)
    r = router.transcribe_with_retry("x.mp3", "prompt", 400.0, "req.json", "raw.sse")
    assert r["blocked"] is False and r["reasons"] == [] and r["attempts"] == 2
    assert [h["outcome"] for h in r["history"]] == ["blocked", "ok"]


def test_partial_content_filter_on_the_last_try_is_still_a_block(monkeypatch):
    replies = iter([_res(BLOCK_TEXT), _res(BLOCK_TEXT), _res("một phần lời giảng", finish="content_filter")])
    monkeypatch.setattr(router, "transcribe_once", lambda *a, **k: next(replies))
    monkeypatch.setattr(router.time, "sleep", lambda s: None)
    r = router.transcribe_with_retry("x.mp3", "prompt", 400.0, "req.json", "raw.sse")
    assert r["blocked"] is True and r["reasons"] == ["blocked by Gemini filters"]
    assert r["history"][-1]["outcome"] == "blocked" and r["history"][-1]["finish"] == "content_filter"


def test_403_stops_at_once_with_one_history_entry(monkeypatch):
    calls = []
    monkeypatch.setattr(router, "transcribe_once",
                        lambda *a, **k: calls.append(1) or _res("", errors=[{"code": 403}], http="403"))
    r = router.transcribe_with_retry("x.mp3", "prompt", 400.0, "req.json", "raw.sse")
    assert r["locked"] is True and len(calls) == 1
    assert [h["outcome"] for h in r["history"]] == ["403"]


def test_history_records_http_finish_words_model_detail_and_wait(monkeypatch):
    replies = iter([_res("", http="500", errors=[{"message": "boom"}]), _res("xin chào " * 400, finish="stop")])
    monkeypatch.setattr(router, "transcribe_once", lambda *a, **k: next(replies))
    monkeypatch.setattr(router.time, "sleep", lambda s: None)
    monkeypatch.setattr(router, "wait_seconds", lambda *a, **k: 7.0)
    r = router.transcribe_with_retry("x.mp3", "prompt", 400.0, "req.json", "raw.sse")
    assert r["history"] == [
        {"n": 1, "engine": "gemini", "http": "500", "finish": "", "words": 0, "served_model": "gemini-3.8-flash",
         "outcome": "error", "detail": 'error; finish=empty; empty despite speech present | [{"message": "boom"}]',
         "wait_sec": 7.0},
        {"n": 2, "engine": "gemini", "http": "200", "finish": "stop", "words": 800, "served_model": "gemini-3.8-flash",
         "outcome": "ok", "detail": "", "wait_sec": 0.0},
    ]


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


def test_a_block_then_an_ordinary_error_on_the_last_try_still_counts_as_blocked(monkeypatch):
    # The only real answers were blocks: the chunk is still blocked (MAI for teaching, empty for silent),
    # not an ordinary failure that would fail the whole lecture.
    replies = iter([_res(BLOCK_TEXT), _res(BLOCK_TEXT), _res("", http="500", errors=[{"message": "boom"}])])
    monkeypatch.setattr(router, "transcribe_once", lambda *a, **k: next(replies))
    monkeypatch.setattr(router.time, "sleep", lambda s: None)
    r = router.transcribe_with_retry("x.mp3", "prompt", 400.0, "req.json", "raw.sse")
    assert r["blocked"] is True and r["reasons"] == ["blocked by Gemini filters"]
    assert [h["outcome"] for h in r["history"]] == ["blocked", "blocked", "error"]
