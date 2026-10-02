"""run_pipeline exits 3 when any chunk has no transcript, so the desktop app never uploads a partial subtitle, and
exits 5 when the mono mix of a polarity-inverted source still cancels the voice."""
import sys

import pytest

from transcribe import exit_codes
from transcribe.chunked_transcribe import polarity, run_trace
from transcribe.chunked_transcribe import run_pipeline as router_rp
from transcribe.mai_transcribe import run_pipeline as mai_rp


def _argv(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["run_pipeline", "--input", "a.mp3", "--out-dir", str(tmp_path)])


def test_router_failed_chunk_exits_incomplete(monkeypatch, tmp_path, capsys):
    _argv(monkeypatch, tmp_path)
    monkeypatch.setattr(router_rp, "run", lambda *a, **k: {"failed": [2], "locked": []})
    with pytest.raises(SystemExit) as e:
        router_rp.main()
    assert e.value.code == exit_codes.INCOMPLETE == 3
    assert "INCOMPLETE" in capsys.readouterr().err


def test_router_locked_chunk_exits_incomplete(monkeypatch, tmp_path):
    _argv(monkeypatch, tmp_path)
    monkeypatch.setattr(router_rp, "run", lambda *a, **k: {"failed": [], "locked": [4, 5]})
    with pytest.raises(SystemExit) as e:
        router_rp.main()
    assert e.value.code == 3


def test_router_complete_run_returns_normally(monkeypatch, tmp_path):
    _argv(monkeypatch, tmp_path)
    monkeypatch.setattr(router_rp, "run", lambda *a, **k: {"failed": [], "locked": []})
    router_rp.main()        # no SystemExit -> exit code 0


def test_mai_failed_chunk_exits_incomplete(monkeypatch, tmp_path, capsys):
    _argv(monkeypatch, tmp_path)
    monkeypatch.setattr(mai_rp, "run", lambda *a, **k: {"failed": [0]})
    with pytest.raises(SystemExit) as e:
        mai_rp.main()
    assert e.value.code == 3
    assert "INCOMPLETE" in capsys.readouterr().err


def test_mai_complete_run_returns_normally(monkeypatch, tmp_path):
    _argv(monkeypatch, tmp_path)
    monkeypatch.setattr(mai_rp, "run", lambda *a, **k: {"failed": []})
    mai_rp.main()


def _raise_cancelled(*a, **k):
    raise polarity.AudioCancelled([(60, 4560)])


def test_router_cancelled_audio_exits_5(monkeypatch, tmp_path, capsys):
    _argv(monkeypatch, tmp_path)
    monkeypatch.setattr(router_rp, "run", _raise_cancelled)
    with pytest.raises(SystemExit) as e:
        router_rp.main()
    assert e.value.code == exit_codes.AUDIO_CANCELLED == 5
    assert "AUDIO_CANCELLED: the mono mix is cancelled in 00:01:00-01:16:00" in capsys.readouterr().err


def test_mai_cancelled_audio_exits_5(monkeypatch, tmp_path, capsys):
    _argv(monkeypatch, tmp_path)
    monkeypatch.setattr(mai_rp, "run", _raise_cancelled)
    with pytest.raises(SystemExit) as e:
        mai_rp.main()
    assert e.value.code == 5
    assert "AUDIO_CANCELLED" in capsys.readouterr().err


def _cancel_the_mixdown(monkeypatch, rp):
    """Make rp.run's mixdown raise AudioCancelled and fail on any VAD step after it. Returns the VAD calls made,
    which stay empty while run() stops at the mixdown."""
    vad_calls = []

    def mixdown(src, dst):
        raise polarity.AudioCancelled([(60, 4560)])

    def vad(path):
        vad_calls.append(path)
        raise AssertionError("VAD ran after the mixdown was cancelled")

    monkeypatch.setattr(rp.audio_utils, "to_mono16k", mixdown)
    monkeypatch.setattr(rp.audio_utils, "speech_segments", vad)
    return vad_calls


def test_router_cancelled_mixdown_stops_before_transcription(monkeypatch, tmp_path, capsys):
    _argv(monkeypatch, tmp_path)
    monkeypatch.setattr(router_rp.config, "load_prompt", lambda f=None: "prompt")
    vad_calls = _cancel_the_mixdown(monkeypatch, router_rp)
    with pytest.raises(SystemExit) as e:
        router_rp.main()
    assert e.value.code == 5
    assert vad_calls == []                                  # nothing ran after the mixdown
    assert not (tmp_path / run_trace.FILE).exists()         # and no trace was written
    assert "AUDIO_CANCELLED: the mono mix is cancelled in 00:01:00-01:16:00" in capsys.readouterr().err


def test_mai_cancelled_mixdown_stops_before_transcription(monkeypatch, tmp_path, capsys):
    _argv(monkeypatch, tmp_path)
    vad_calls = _cancel_the_mixdown(monkeypatch, mai_rp)
    with pytest.raises(SystemExit) as e:
        mai_rp.main()
    assert e.value.code == 5
    assert vad_calls == []
    assert not (tmp_path / run_trace.FILE).exists()
    assert "AUDIO_CANCELLED: the mono mix is cancelled in 00:01:00-01:16:00" in capsys.readouterr().err
