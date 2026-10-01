"""run_trace.json tells everything a run did: metadata, VAD, and per chunk every try, the outcome and the text."""
import hashlib
import json

import pytest

from transcribe.chunked_transcribe import config, run_pipeline, run_trace

BASE = {"idx": 1, "start": 0.0, "end": 600.0, "speech_sec": 12.0}


@pytest.mark.parametrize("rec,outcome", [
    ({"included": True, "note": ""}, "merged"),
    ({"included": True, "engine": "mai", "note": "mai fallback: gemini blocked by filters"}, "mai"),
    ({"included": True, "silent": True, "note": "silent: sentences kept when MMS score >= 0.5"}, "silent-scored"),
    ({"included": False, "silent": True, "note": "skipped (no speech)"}, "silent-empty"),
    ({"included": False, "silent": True, "note": "silent blocked: gemini filters refused a chunk without speech"},
     "silent-blocked"),
    ({"included": False, "note": "error after retry: error"}, "failed"),
    ({"included": False, "locked": True, "note": "403 account locked - not transcribed"}, "locked"),
])
def test_chunk_entry_outcome(rec, outcome):
    assert run_trace.chunk_entry({**BASE, **rec}, "chữ")["outcome"] == outcome


def test_chunk_entry_keeps_the_text_only_when_the_chunk_was_merged():
    assert run_trace.chunk_entry({**BASE, "included": True, "note": ""}, "lời")["text"] == "lời"
    assert run_trace.chunk_entry({**BASE, "included": False, "note": "error after retry: x"}, "rác")["text"] == ""


def test_chunk_entry_carries_vad_tries_and_mai():
    e = run_trace.chunk_entry({**BASE, "included": True, "silent": True, "note": "silent: x",
                               "history": [{"n": 1}], "mai": None}, "câu")
    assert e == {"idx": 1, "start": 0.0, "end": 600.0, "speech_sec": 12.0, "vad": "silent",
                 "attempts": [{"n": 1}], "mai": None, "outcome": "silent-scored", "note": "silent: x", "text": "câu"}


def test_new_trace_records_meta_vad_and_chunks(monkeypatch):
    monkeypatch.setattr(run_trace, "git_commit", lambda: "abc1234")
    monkeypatch.setattr(run_trace, "now_iso", lambda: "2026-10-01T10:12:00+07:00")
    manifest = [{**BASE, "speech_sec": 500.0, "included": True, "note": "", "history": [], "mai": None}]
    t = run_trace.new_trace(input_path="D:/x/bai.m4a", duration_sec=600.0, prompt="abc", segments=[[0.5, 4.5]],
                            manifest=manifest, texts={1: "lời"}, started_at="2026-10-01T10:00:00+07:00",
                            mai_enabled=True)
    assert t["version"] == 1
    assert t["meta"] == {
        "mode": "full", "input": "bai.m4a", "duration_sec": 600.0, "pipeline_commit": "abc1234",
        "model": config.MODEL, "prompt_sha256": hashlib.sha256(b"abc").hexdigest(),
        "thresholds": {"min_speech_chunk_sec": 30.0, "min_speech_ratio": 0.5,
                       "silent_min_score": config.SILENT_MIN_SCORE, "max_attempts": 3},
        "mai_enabled": True, "started_at": "2026-10-01T10:00:00+07:00", "finished_at": "2026-10-01T10:12:00+07:00"}
    assert t["vad"] == {"speech_sec": 4.0, "segments": [[0.5, 4.5]]}
    assert t["chunks"][0]["text"] == "lời" and t["chunks"][0]["vad"] == "speech"


def test_write_then_read_round_trips_and_a_missing_or_bad_file_reads_as_none(tmp_path):
    run_trace.write(tmp_path, {"version": 1, "chunks": [{"idx": 1, "text": "Nam mô"}]})
    assert run_trace.read(tmp_path) == {"version": 1, "chunks": [{"idx": 1, "text": "Nam mô"}]}
    assert run_trace.read(tmp_path / "nowhere") is None
    (tmp_path / "run_trace.json").write_text("{bad", encoding="utf-8")
    assert run_trace.read(tmp_path) is None


def test_a_nan_or_infinite_score_is_written_as_null_so_strict_readers_can_parse_the_trace(tmp_path):
    # A NaN MMS score (broken audio) used to be written as a bare NaN token: the desktop's JSON.parse and
    # Postgres jsonb refuse it, so the archived run lost its whole trace and kept only the app section.
    trace = {"version": 1, "chunks": [{"idx": 1, "build": {"mean_score": float("nan")},
                                       "sentences": [{"score": float("inf")}, {"score": -float("inf")},
                                                     {"score": 0.5}]}],
             "vad": {"segments": [(0.0, 1.5)]}}
    run_trace.write(tmp_path, trace)

    def refuse(token):
        raise ValueError(f"non-standard JSON number {token}")

    data = json.loads((tmp_path / run_trace.FILE).read_text(encoding="utf-8"), parse_constant=refuse)
    assert data["chunks"][0]["build"]["mean_score"] is None
    assert [s["score"] for s in data["chunks"][0]["sentences"]] == [None, None, 0.5]
    assert data["vad"]["segments"] == [[0.0, 1.5]]
    assert trace["chunks"][0]["sentences"][2]["score"] == 0.5  # the caller's trace is not modified


def test_run_writes_the_trace_even_when_a_chunk_failed_or_hit_403(monkeypatch, tmp_path):
    out = tmp_path / "out"
    monkeypatch.setattr(run_pipeline.audio_utils, "to_mono16k", lambda src, dst: 1800.0)
    monkeypatch.setattr(run_pipeline.audio_utils, "speech_segments", lambda p: [[1.0, 400.0], [700.0, 900.0]])
    monkeypatch.setattr(run_pipeline.audio_utils, "build_plan", lambda p, d, s: [
        {"idx": 1, "start": 0.0, "end": 600.0, "speech_sec": 399.0, "cut_gap_sec": None},
        {"idx": 2, "start": 600.0, "end": 1200.0, "speech_sec": 200.0, "cut_gap_sec": 1.0},
        {"idx": 3, "start": 1200.0, "end": 1800.0, "speech_sec": 0.0, "cut_gap_sec": 1.0}])
    monkeypatch.setattr(run_pipeline.config, "load_prompt", lambda f=None: "prompt")
    monkeypatch.setattr(run_trace, "git_commit", lambda: "abc1234")
    plan = {1: ("", True, False), 2: ("error after retry: error", False, False),
            3: ("403 account locked - not transcribed", False, True)}

    def fake_process(c, out_dir, mono, prompt, lock):
        note, included, locked = plan[c["idx"]]
        (out_dir / "chunks" / f"{c['idx']:02d}.txt").write_text("lời giảng." if included else "", encoding="utf-8")
        return {**c, "included": included, "locked": locked, "silent": False, "note": note, "words": 2,
                "history": [{"n": 1, "outcome": "ok"}], "mai": None}

    monkeypatch.setattr(run_pipeline, "_process_chunk", fake_process)
    res = run_pipeline.run("x/bai.m4a", str(out))
    assert res["failed"] == [2] and res["locked"] == [3]
    trace = run_trace.read(out)
    assert trace["version"] == 1 and trace["meta"]["input"] == "bai.m4a"
    assert trace["meta"]["pipeline_commit"] == "abc1234" and trace["vad"]["speech_sec"] == 599.0
    assert [c["outcome"] for c in trace["chunks"]] == ["merged", "failed", "locked"]
    assert trace["chunks"][0]["text"] == "lời giảng." and trace["chunks"][0]["attempts"] == [{"n": 1, "outcome": "ok"}]


def test_a_trace_that_cannot_be_written_never_changes_the_run_outcome(monkeypatch, tmp_path, capsys):
    out = tmp_path / "out"
    monkeypatch.setattr(run_pipeline.audio_utils, "to_mono16k", lambda src, dst: 600.0)
    monkeypatch.setattr(run_pipeline.audio_utils, "speech_segments", lambda p: [[1.0, 400.0]])
    monkeypatch.setattr(run_pipeline.audio_utils, "build_plan", lambda p, d, s: [
        {"idx": 1, "start": 0.0, "end": 600.0, "speech_sec": 399.0, "cut_gap_sec": None}])
    monkeypatch.setattr(run_pipeline.config, "load_prompt", lambda f=None: "prompt")

    def fake_process(c, out_dir, mono, prompt, lock):
        (out_dir / "chunks" / "01.txt").write_text("lời giảng.", encoding="utf-8")
        return {**c, "included": True, "locked": False, "silent": False, "note": "", "words": 2, "history": [],
                "mai": None}

    def broken_write(out_dir, trace):
        raise OSError("disk full")

    monkeypatch.setattr(run_pipeline, "_process_chunk", fake_process)
    monkeypatch.setattr(run_trace, "write", broken_write)
    res = run_pipeline.run("x/bai.m4a", str(out))
    assert res["failed"] == [] and (out / "manifest.json").exists()
    assert "run_trace.json" in capsys.readouterr().err
