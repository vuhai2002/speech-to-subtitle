"""build_srt against the REAL MMS tokenizer and aligner (a fake acoustic model and fake audio, so no weights are
downloaded): text MMS cannot align never crashes the build. In a chunk VAD hears as silent it is recorded and dropped;
a teaching chunk with no alignable word stops the lecture with exit 4 (a teaching chunk never drops out quietly),
while a teaching chunk with only SOME unalignable tokens aligns as usual."""
import json

import pytest

torch = pytest.importorskip("torch")
pipelines = pytest.importorskip("torchaudio.pipelines")

import realign.align_words as aw  # noqa: E402

from transcribe import exit_codes  # noqa: E402
from transcribe.chunked_transcribe import build_srt, run_trace  # noqa: E402

SR = 16000


class FakeModel:
    """Stands in for the wav2vec2 acoustic model: uniform log-probabilities, 50 frames per second of audio."""

    def __init__(self, vocab: int):
        self.vocab = vocab

    def __call__(self, wav):
        frames = max(1, wav.size(1) // 320)
        return torch.log_softmax(torch.zeros(1, frames, self.vocab), dim=-1), None


@pytest.fixture
def models(monkeypatch):
    # 1 s of audio = 50 frames: enough for a short teaching line, far too few for a long silent text.
    monkeypatch.setattr(aw, "load_audio", lambda path, sample_rate, max_sec=0.0: (torch.zeros(1, SR), SR))
    bundle = pipelines.MMS_FA
    return FakeModel(len(bundle.get_dict())), bundle.get_tokenizer(), bundle.get_aligner()


def _out_dir(tmp_path, chunks):
    """chunks: list of (silent, text); chunk i starts at i * 600 s."""
    (tmp_path / "chunks").mkdir()
    (tmp_path / "plan.json").write_text(json.dumps({"input": "x/bai.m4a", "segments": [[0.1, 0.9]]}),
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


def test_silent_text_with_no_alignable_token_is_dropped_and_recorded(models, tmp_path):
    out = _out_dir(tmp_path, [(False, "Xin chào."), (True, "♪ ... (...)")])
    path = build_srt.build(str(out), "cpu", models=models)
    assert "Xin chào." in open(path, encoding="utf-8").read()
    silent = run_trace.read(out)["chunks"][1]
    assert silent["build"]["align_error"] == "no alignable word"
    assert silent["sentences"] and all(s["score"] is None and not s["kept"] for s in silent["sentences"])


def test_silent_text_longer_than_its_audio_is_dropped_and_recorded(models, tmp_path):
    out = _out_dir(tmp_path, [(False, "Xin chào."), (True, " ".join(["abcdefgh"] * 60) + ".")])
    path = build_srt.build(str(out), "cpu", models=models)
    assert "Xin chào." in open(path, encoding="utf-8").read()
    silent = run_trace.read(out)["chunks"][1]
    assert silent["build"]["align_error"].startswith("align failed: ")
    assert all(not s["kept"] for s in silent["sentences"])


def test_a_lecture_with_only_unalignable_silent_text_exits_4_not_with_a_crash(models, tmp_path):
    out = _out_dir(tmp_path, [(True, "♪ ♪ ♪")])
    with pytest.raises(exit_codes.NoAlignedWords):
        build_srt.build(str(out), "cpu", models=models)
    assert run_trace.read(out)["build"]["exit"] == 4


def test_a_teaching_chunk_with_empty_text_exits_4_not_with_a_crash(models, tmp_path):
    # In real runs this is a whole file under a minute (speech fills half of it but stays under 30 s, so an empty
    # Gemini reply passes the text check). build_srt does not look at speech_sec, so the fixture's sizes do not matter.
    out = _out_dir(tmp_path, [(False, "")])
    with pytest.raises(exit_codes.NoAlignedWords, match="chunk 01"):
        build_srt.build(str(out), "cpu", models=models)
    trace = run_trace.read(out)
    assert trace["build"]["exit"] == 4
    assert trace["chunks"][0]["build"]["align_error"] == "no alignable word"


def test_a_teaching_chunk_with_only_symbols_and_digits_stops_the_lecture_with_exit_4(models, tmp_path):
    # A teaching chunk never drops out quietly: the lecture stops (nothing is uploaded) and the trace names the chunk.
    out = _out_dir(tmp_path, [(False, "Xin chào."), (False, "♪ ... 123")])
    with pytest.raises(exit_codes.NoAlignedWords, match="chunk 02"):
        build_srt.build(str(out), "cpu", models=models)
    trace = run_trace.read(out)
    assert trace["build"]["exit"] == 4
    assert trace["chunks"][0]["build"]["aligned_words"] == 2
    assert trace["chunks"][1]["build"] == {"total_words": 3, "aligned_words": 0, "mean_score": None,
                                           "align_error": "no alignable word"}
    assert not list(out.glob("*.srt"))


def test_a_teaching_chunk_with_some_unalignable_tokens_still_aligns(models, tmp_path):
    # Most lectures have a number, a dash or a "♪" somewhere: only a chunk with NO alignable word may stop the lecture.
    out = _out_dir(tmp_path, [(False, "♪ 108 lạy ... Xin chào. -")])
    path = build_srt.build(str(out), "cpu", models=models)
    assert "Xin chào." in open(path, encoding="utf-8").read()
    built = run_trace.read(out)["chunks"][0]["build"]
    assert built["aligned_words"] == 3 and "align_error" not in built
