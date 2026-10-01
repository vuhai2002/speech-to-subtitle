"""build_srt writes no .srt and exits 4 when nothing can be placed on the timeline."""
import json
import sys

import pytest

from transcribe import exit_codes
from transcribe.chunked_transcribe import build_srt as router_bs
from transcribe.mai_transcribe import build_srt as mai_bs


def test_router_no_words_raises_before_writing(monkeypatch, tmp_path):
    monkeypatch.setattr(router_bs, "_aligned_words", lambda out_dir, device, models=None: ([], None, "bai", {}))
    with pytest.raises(exit_codes.NoAlignedWords):
        router_bs.build(str(tmp_path), "cpu")
    assert not (tmp_path / "bai.srt").exists()


def test_router_words_without_timestamps_raise(monkeypatch, tmp_path):
    words = [{"w": "xin", "start": None, "end": None, "score": None}]
    monkeypatch.setattr(router_bs, "_aligned_words", lambda out_dir, device, models=None: (words, None, "bai", {}))
    with pytest.raises(exit_codes.NoAlignedWords):
        router_bs.build(str(tmp_path), "cpu")
    assert not (tmp_path / "bai.srt").exists()


def test_router_words_without_scores_still_build(monkeypatch, tmp_path):
    # The mean MMS score used to divide by zero when no word carried a score.
    words = [{"w": "xin", "start": 1.0, "end": 1.4, "score": None},
             {"w": "chào.", "start": 1.5, "end": 2.0, "score": None}]
    monkeypatch.setattr(router_bs, "_aligned_words", lambda out_dir, device, models=None: (words, None, "bai", {}))
    path = router_bs.build(str(tmp_path), "cpu")
    assert (tmp_path / "bai.srt").read_text(encoding="utf-8").strip() != ""
    assert path.endswith("bai.srt")


def test_router_main_exits_4(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(router_bs, "_aligned_words", lambda out_dir, device, models=None: ([], None, "bai", {}))
    monkeypatch.setattr(sys, "argv", ["build_srt", "--out-dir", str(tmp_path), "--device", "cpu"])
    with pytest.raises(SystemExit) as e:
        router_bs.main()
    assert e.value.code == exit_codes.NO_ALIGNED_WORDS == 4
    assert "NO_ALIGNED_WORDS" in capsys.readouterr().err


def _mai_out(tmp_path, words):
    (tmp_path / "mai_words.json").write_text(json.dumps(words), encoding="utf-8")
    (tmp_path / "plan.json").write_text(json.dumps({"input": "x/bai.mp3", "segments": []}), encoding="utf-8")


def test_mai_no_words_raises_before_writing(tmp_path):
    _mai_out(tmp_path, [])
    with pytest.raises(exit_codes.NoAlignedWords):
        mai_bs.build(str(tmp_path))
    assert not (tmp_path / "bai.srt").exists()


def test_mai_main_exits_4(monkeypatch, tmp_path, capsys):
    _mai_out(tmp_path, [])
    monkeypatch.setattr(sys, "argv", ["build_srt", "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as e:
        mai_bs.main()
    assert e.value.code == 4
    assert "NO_ALIGNED_WORDS" in capsys.readouterr().err
