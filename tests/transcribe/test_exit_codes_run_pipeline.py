"""run_pipeline exits 3 when any chunk has no transcript, so the desktop app never uploads a partial subtitle."""
import sys

import pytest

from transcribe import exit_codes
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
