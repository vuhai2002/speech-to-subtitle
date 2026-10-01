"""build_srt keeps the sentences of a silent chunk that MMS scores >= SILENT_MIN_SCORE, widens the clamp region to
cover them, and folds the alignment and sentence results into run_trace.json."""
import json

import pytest
import realign.align_words as aw
import realign.cue_clamp as cc

from transcribe import exit_codes
from transcribe.chunked_transcribe import build_srt, run_trace

LOW = {"Lạc", "đề", "hoàn", "toàn."}
SILENT_WORDS = {"Câu", "này", "rõ", "ràng.", "Lạc", "đề", "hoàn", "toàn."}
ALL_WORDS = SILENT_WORDS | {"Xin", "chào", "đại", "chúng."}


def _fake_align(calls, unaligned=frozenset()):
    def align(mp3, words, device, models=None, **kw):
        calls.append(models)
        return [{"w": w, "start": None, "end": None, "score": None} if w in unaligned else
                {"w": w, "start": 1.0 + i * 0.5, "end": 1.4 + i * 0.5, "score": 0.1 if w in LOW else 0.8}
                for i, w in enumerate(words)]
    return align


def _out_dir(tmp_path):
    (tmp_path / "chunks").mkdir()
    (tmp_path / "plan.json").write_text(json.dumps({"input": "x/bai.m4a", "segments": [[1.0, 30.0]]}),
                                        encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps([
        {"idx": 1, "start": 0.0, "end": 600.0, "speech_sec": 400.0, "included": True, "silent": False, "note": ""},
        {"idx": 2, "start": 600.0, "end": 1200.0, "speech_sec": 3.0, "included": True, "silent": True,
         "note": "silent: sentences kept when MMS score >= 0.5"}]), encoding="utf-8")
    (tmp_path / "chunks" / "01.txt").write_text("Xin chào đại chúng.", encoding="utf-8")
    (tmp_path / "chunks" / "02.txt").write_text("Câu này rõ ràng. Lạc đề hoàn toàn.", encoding="utf-8")
    run_trace.write(tmp_path, {"version": 1, "chunks": [{"idx": 1, "outcome": "merged"},
                                                        {"idx": 2, "outcome": "silent-scored"}]})
    return tmp_path


@pytest.fixture
def built(monkeypatch, tmp_path):
    calls, regions, loads = [], [], []
    monkeypatch.setattr(aw, "align", _fake_align(calls))
    monkeypatch.setattr(aw, "get_models", lambda device: loads.append(device) or "LOADED")
    real_clamp = cc.clamp_cues
    monkeypatch.setattr(cc, "clamp_cues", lambda cues, region, cfg: regions.append(region) or real_clamp(cues, region, cfg))
    out = _out_dir(tmp_path)
    path = build_srt.build(str(out), "cpu")
    return out, path, calls, regions, loads


def test_kept_sentences_reach_the_srt_and_dropped_ones_do_not(built):
    out, path, _, _, _ = built
    text = open(path, encoding="utf-8").read()
    assert "Xin chào đại chúng." in text and "Câu này rõ ràng." in text
    assert "Lạc" not in text


def test_the_trace_gets_the_scored_sentences_and_the_build_summary(built):
    out, _, _, _, _ = built
    trace = run_trace.read(out)
    silent = trace["chunks"][1]
    assert [(s["text"], s["score"], s["kept"]) for s in silent["sentences"]] == [
        ("Câu này rõ ràng.", 0.8, True), ("Lạc đề hoàn toàn.", 0.1, False)]
    assert silent["sentences"][0]["start"] == pytest.approx(601.0)   # times are from the start of the file
    assert silent["build"] == {"total_words": 8, "aligned_words": 8, "mean_score": 0.45}
    assert trace["chunks"][0]["build"]["aligned_words"] == 4
    assert trace["build"]["exit"] == 0 and trace["build"]["cues"] >= 2


def test_the_clamp_region_is_widened_to_cover_kept_silent_words(built):
    _, _, _, regions, _ = built
    assert regions == [pytest.approx((1.0, 602.9))]


def test_mms_is_loaded_once_per_build_and_reused_for_every_chunk(built):
    _, _, calls, _, loads = built
    assert loads == ["cpu"] and calls == ["LOADED", "LOADED"]


def test_preloaded_models_are_used_and_not_loaded_again(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(aw, "align", _fake_align(calls))

    def no_load(device):
        raise AssertionError("MMS loaded again")

    monkeypatch.setattr(aw, "get_models", no_load)
    build_srt.build(str(_out_dir(tmp_path)), "cpu", models="PRELOADED")
    assert calls == ["PRELOADED", "PRELOADED"]


def test_a_silent_chunk_mms_cannot_align_keeps_nothing_and_does_not_crash(monkeypatch, tmp_path):
    monkeypatch.setattr(aw, "align", _fake_align([], unaligned=SILENT_WORDS))
    monkeypatch.setattr(aw, "get_models", lambda device: "LOADED")
    out = _out_dir(tmp_path)
    path = build_srt.build(str(out), "cpu")
    assert "Câu" not in open(path, encoding="utf-8").read()
    silent = run_trace.read(out)["chunks"][1]
    assert all(s["score"] is None and s["kept"] is False for s in silent["sentences"])


def test_no_aligned_word_at_all_still_records_exit_4_in_the_trace(monkeypatch, tmp_path):
    monkeypatch.setattr(aw, "align", _fake_align([], unaligned=ALL_WORDS))
    monkeypatch.setattr(aw, "get_models", lambda device: "LOADED")
    out = _out_dir(tmp_path)
    with pytest.raises(exit_codes.NoAlignedWords):
        build_srt.build(str(out), "cpu")
    assert run_trace.read(out)["build"]["exit"] == 4
    assert not (out / "bai.srt").exists()


def test_a_run_dir_without_a_trace_still_builds(monkeypatch, tmp_path):
    monkeypatch.setattr(aw, "align", _fake_align([]))
    monkeypatch.setattr(aw, "get_models", lambda device: "LOADED")
    out = _out_dir(tmp_path)
    (out / "run_trace.json").unlink()
    build_srt.build(str(out), "cpu")
    assert not (out / "run_trace.json").exists()


def _lecture(tmp_path, chunks):
    """chunks: list of (silent, text); chunk i starts at (i - 1) * 600 s; VAD speech only in the first 30 s."""
    (tmp_path / "chunks").mkdir()
    (tmp_path / "plan.json").write_text(json.dumps({"input": "x/bai.m4a", "segments": [[1.0, 30.0]]}),
                                        encoding="utf-8")
    manifest, entries = [], []
    for i, (silent, text) in enumerate(chunks, start=1):
        manifest.append({"idx": i, "start": (i - 1) * 600.0, "end": i * 600.0, "speech_sec": 3.0 if silent else 400.0,
                         "included": True, "silent": silent, "note": ""})
        entries.append({"idx": i, "outcome": "silent-scored" if silent else "merged"})
        (tmp_path / "chunks" / f"{i:02d}.txt").write_text(text, encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    run_trace.write(tmp_path, {"version": 1, "chunks": entries})
    return tmp_path


def _srt_starts(path):
    """{cue text (one line): start timestamp} from an SRT file."""
    out = {}
    for block in open(path, encoding="utf-8").read().strip().split("\n\n"):
        lines = block.split("\n")
        out[" ".join(lines[2:])] = lines[1].split(" --> ")[0]
    return out


def test_a_lecture_whose_only_words_are_kept_silent_sentences_still_builds(monkeypatch, tmp_path):
    monkeypatch.setattr(aw, "align", _fake_align([]))
    monkeypatch.setattr(aw, "get_models", lambda device: "LOADED")
    out = _lecture(tmp_path, [(True, "Câu này rõ ràng. Lạc đề hoàn toàn.")])
    path = build_srt.build(str(out), "cpu")
    assert "Câu này rõ ràng." in open(path, encoding="utf-8").read()
    assert run_trace.read(out)["build"]["exit"] == 0


def test_a_kept_sentence_starting_with_an_unalignable_token_keeps_its_own_time(monkeypatch, tmp_path):
    # "108" has no MMS token: without pinning it would borrow the end of the teaching chunk minutes earlier.
    monkeypatch.setattr(aw, "align", _fake_align([], unaligned={"108"}))
    monkeypatch.setattr(aw, "get_models", lambda device: "LOADED")
    out = _lecture(tmp_path, [(False, "Xin chào đại chúng."), (True, "108 lạy sám hối xong.")])
    starts = _srt_starts(build_srt.build(str(out), "cpu"))
    assert starts["108 lạy sám hối xong."] == "00:10:01,500"


@pytest.mark.parametrize("chunks,short_start", [
    ([(False, "Xin chào đại chúng."), (True, "Mô Phật.")], "00:10:01,000"),   # kept 2-word line at the very end
    ([(True, "Mô Phật."), (False, "Xin chào đại chúng.")], "00:00:01,000"),   # ... and at the very start
])
def test_a_short_kept_silent_line_at_the_lecture_edge_is_not_merged_into_a_distant_cue(monkeypatch, tmp_path,
                                                                                      chunks, short_start):
    monkeypatch.setattr(aw, "align", _fake_align([]))
    monkeypatch.setattr(aw, "get_models", lambda device: "LOADED")
    starts = _srt_starts(build_srt.build(str(_lecture(tmp_path, chunks)), "cpu"))
    assert starts.get("Mô Phật.") == short_start


def test_a_trace_that_cannot_be_written_never_costs_the_srt(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(aw, "align", _fake_align([]))
    monkeypatch.setattr(aw, "get_models", lambda device: "LOADED")
    out = _out_dir(tmp_path)

    def broken_write(out_dir, trace):
        raise OSError("disk full")

    monkeypatch.setattr(run_trace, "write", broken_write)
    path = build_srt.build(str(out), "cpu")
    assert open(path, encoding="utf-8").read().strip() != ""
    assert "run_trace.json" in capsys.readouterr().err


def test_a_short_kept_silent_line_is_not_pulled_back_onto_a_teaching_cue_minutes_earlier(monkeypatch, tmp_path):
    times = {"Hôm": (587.0, 587.4), "nay": (587.5, 587.9), "học.": (588.0, 588.4), "Mô": (100.0, 100.3),
             "Phật.": (100.3, 100.6), "Nam": (101.0, 101.2), "mô": (101.2, 101.4), "Bổn": (101.4, 101.6),
             "Sư.": (101.6, 101.8)}

    def align(mp3, words, device, models=None, **kw):
        return [{"w": w, "start": times[w][0], "end": times[w][1], "score": 0.8} for w in words]

    monkeypatch.setattr(aw, "align", align)
    monkeypatch.setattr(aw, "get_models", lambda device: "LOADED")
    out = _lecture(tmp_path, [(False, "Hôm nay học."), (True, "Mô Phật. Nam mô Bổn Sư.")])
    starts = _srt_starts(build_srt.build(str(out), "cpu"))
    assert [(t, st) for t, st in starts.items() if "Mô Phật." in t] == [("Mô Phật. Nam mô Bổn Sư.", "00:11:40,000")]



def test_the_trace_records_the_threshold_the_build_judged_with(monkeypatch, tmp_path):
    # A rebuild with another SILENT_MIN_SCORE (e.g. trying 0.45) must not leave the trace claiming the old one.
    monkeypatch.setattr(aw, "align", _fake_align([]))
    monkeypatch.setattr(aw, "get_models", lambda device: "LOADED")
    monkeypatch.setattr(build_srt.config, "SILENT_MIN_SCORE", 0.05)
    out = _out_dir(tmp_path)
    build_srt.build(str(out), "cpu")
    trace = run_trace.read(out)
    assert trace["meta"]["thresholds"]["silent_min_score"] == 0.05
    assert [s["kept"] for s in trace["chunks"][1]["sentences"]] == [True, True]
