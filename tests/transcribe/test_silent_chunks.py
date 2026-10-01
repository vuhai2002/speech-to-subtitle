"""A chunk VAD hears as silent keeps its text for build_srt to score, is never sent to MAI, and is left empty
(not failed) when Gemini's filter still blocks it after the retries."""
import threading

import pytest

from transcribe.chunked_transcribe import config, mai_fallback, run_pipeline

SILENT = {"idx": 9, "start": 4572.9, "end": 4960.0, "speech_sec": 12.0, "cut_gap_sec": 1.0}
TEACHING = {"idx": 3, "start": 1200.0, "end": 1800.0, "speech_sec": 400.0, "cut_gap_sec": 1.0}
HISTORY = [{"n": 1, "engine": "gemini", "http": "200", "finish": "stop", "words": 4, "served_model": "m",
            "outcome": "ok", "detail": "", "wait_sec": 0.0}]
BLOCKED = {"res": {"text": "This request was blocked by Gemini's filters.", "finish": "", "http": "200",
                   "served_model": "m", "usage": None, "errors": []},
           "attempts": 3, "reasons": ["blocked by Gemini filters"], "locked": False, "blocked": True, "history": []}


def _ok(text):
    return {"res": {"text": text, "finish": "stop", "http": "200", "served_model": "m", "usage": None, "errors": []},
            "attempts": 1, "reasons": [], "locked": False, "blocked": False, "history": HISTORY}


def _setup(monkeypatch, tmp_path, result):
    (tmp_path / "chunks").mkdir()
    monkeypatch.setattr(run_pipeline.audio_utils, "cut_chunk", lambda *a, **k: None)
    monkeypatch.setattr(run_pipeline.transcribe_client, "transcribe_with_retry", lambda *a, **k: result)
    monkeypatch.setattr(mai_fallback.mai_config, "API_KEY", "k")
    calls = []

    def fake_mai(mp3, raw, expect_speech):
        calls.append(mp3)
        return {"res": {"text": "lời giảng " * 450, "cost": 0.015}, "attempts": 1, "reasons": []}

    monkeypatch.setattr(mai_fallback.mai_client, "transcribe_with_retry", fake_mai)
    return calls


def _process(tmp_path, chunk):
    return run_pipeline._process_chunk(chunk, tmp_path, str(tmp_path / "mono16k.mp3"), "prompt", threading.Lock())


def test_silent_chunk_with_text_is_kept_for_scoring(monkeypatch, tmp_path, capsys):
    calls = _setup(monkeypatch, tmp_path, _ok("Tính làm Bồ Tát nào? Đừng có đòi làm Bồ Tát nha!"))
    rec = _process(tmp_path, SILENT)
    assert rec["included"] is True and rec["silent"] is True
    assert rec["note"] == "silent: sentences kept when MMS score >= 0.5"
    assert (tmp_path / "chunks" / "09.txt").read_text(encoding="utf-8").startswith("Tính làm Bồ Tát nào?")
    assert rec["history"] == HISTORY and rec["mai"] is None and calls == []
    assert capsys.readouterr().out.rstrip().endswith("MERGED | silent: scored at build")


def test_silent_chunk_without_text_is_skipped(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, _ok("   "))
    rec = _process(tmp_path, SILENT)
    assert rec["included"] is False and rec["note"] == "skipped (no speech)"


def test_silent_chunk_still_blocked_is_left_empty_and_never_sent_to_mai(monkeypatch, tmp_path):
    calls = _setup(monkeypatch, tmp_path, BLOCKED)
    rec = _process(tmp_path, SILENT)
    assert rec["included"] is False
    assert rec["note"] == "silent blocked: gemini filters refused a chunk without speech"
    assert calls == [] and rec["mai"] is None
    assert (tmp_path / "chunks" / "09.txt").read_text(encoding="utf-8") == ""


def test_teaching_chunk_still_blocked_goes_to_mai_and_records_it(monkeypatch, tmp_path):
    calls = _setup(monkeypatch, tmp_path, BLOCKED)
    rec = _process(tmp_path, TEACHING)
    assert rec["included"] is True and rec["engine"] == "mai" and rec["silent"] is False
    assert rec["mai"] == {"outcome": "ok", "reason": "", "words": 900, "cost": 0.015}
    assert len(calls) == 1


def test_a_partial_content_filter_reply_is_never_merged(monkeypatch, tmp_path):
    partial = {**BLOCKED, "res": {**BLOCKED["res"], "text": "một phần lời giảng", "finish": "content_filter"}}
    _setup(monkeypatch, tmp_path, partial)
    monkeypatch.setattr(mai_fallback.mai_config, "API_KEY", "")
    rec = _process(tmp_path, TEACHING)
    assert rec["included"] is False and rec["note"].startswith("error after retry: blocked by Gemini filters")
    assert (tmp_path / "chunks" / "03.txt").read_text(encoding="utf-8") == ""


def test_silent_chunk_with_an_ordinary_error_still_fails(monkeypatch, tmp_path):
    err = {"res": {"text": "", "finish": "", "http": "500", "served_model": "m", "usage": None,
                   "errors": [{"message": "boom"}]},
           "attempts": 3, "reasons": ["error", "finish=empty"], "locked": False, "blocked": False, "history": []}
    _setup(monkeypatch, tmp_path, err)
    rec = _process(tmp_path, SILENT)
    assert rec["included"] is False and rec["note"] == "error after retry: error; finish=empty"


def test_silent_note_shows_the_configured_threshold(monkeypatch):
    monkeypatch.setattr(config, "SILENT_MIN_SCORE", 0.45)
    assert run_pipeline.silent_note() == "silent: sentences kept when MMS score >= 0.45"


@pytest.mark.parametrize("raw,expected", [(None, 0.5), ("", 0.5), ("  ", 0.5), ("0,6", 0.6), (" 0.7 ", 0.7),
                                          ("1", 1.0)])
def test_silent_min_score_parses(raw, expected):
    assert config._silent_min_score(raw) == expected


@pytest.mark.parametrize("raw", ["0", "-0.1", "1.5", "abc", "nan", "inf"])
def test_silent_min_score_rejects_bad_values(raw):
    with pytest.raises(ValueError, match="SILENT_MIN_SCORE"):
        config._silent_min_score(raw)


def test_raw_transcript_keeps_teaching_text_only(monkeypatch, tmp_path):
    out = tmp_path / "out"
    monkeypatch.setattr(run_pipeline.audio_utils, "to_mono16k", lambda src, dst: 1200.0)
    monkeypatch.setattr(run_pipeline.audio_utils, "speech_segments", lambda p: [[1.0, 500.0]])
    monkeypatch.setattr(run_pipeline.audio_utils, "build_plan", lambda p, d, s: [
        dict(TEACHING, idx=1, start=0.0, end=600.0), dict(SILENT, idx=2, start=600.0, end=1200.0)])
    monkeypatch.setattr(run_pipeline.config, "load_prompt", lambda f=None: "prompt")

    def fake_process(c, out_dir, mono, prompt, lock):
        silent = c["idx"] == 2
        (out_dir / "chunks" / f"{c['idx']:02d}.txt").write_text("câu hát." if silent else "lời giảng.",
                                                                encoding="utf-8")
        return {**c, "included": True, "silent": silent, "locked": False, "note": "", "words": 2,
                "history": [], "mai": None}

    monkeypatch.setattr(run_pipeline, "_process_chunk", fake_process)
    res = run_pipeline.run("x/bai.m4a", str(out))
    assert (out / "raw_transcript.txt").read_text(encoding="utf-8") == "lời giảng."
    assert res["words"] == 2            # the words of raw_transcript.txt, not the unscored silent text
