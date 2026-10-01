"""transcribe_once never reads a previous try's answer as this try's.

curl writes no -o file when it gets no reply body (no connection, a timeout before the reply, a connection dropped
after the headers), so an answer left by an earlier try or run in the same chunk folder must be gone before curl
runs: otherwise an unreachable router reads as the old 403 (the run stops as "account locked") or as the old text.
"""
import builtins
import errno
from types import SimpleNamespace

import pytest

from transcribe.chunked_transcribe import transcribe_client as router

OLD_403 = 'data: {"error": {"code": 403, "message": "account locked"}}\n'
OLD_TEXT = 'data: {"choices": [{"delta": {"content": "câu cũ"}, "finish_reason": "stop"}]}\n'
NEW_TEXT = 'data: {"choices": [{"delta": {"content": "câu mới"}, "finish_reason": "stop"}]}\n'


def _setup(monkeypatch, tmp_path, old_answer, curl):
    mp3 = tmp_path / "00.mp3"
    mp3.write_bytes(b"ID3")
    raw = tmp_path / "00_raw.sse"
    raw.write_text(old_answer, encoding="utf-8")
    monkeypatch.setattr(router.config, "BASE_URL", "https://router.invalid/v1")
    monkeypatch.setattr(router.config, "API_KEY", "k")
    calls = []

    def fake_run(argv, capture_output, text):
        calls.append(argv)
        return curl(argv)

    monkeypatch.setattr(router.subprocess, "run", fake_run)
    return lambda: router.transcribe_once(str(mp3), "prompt", str(tmp_path / "00_req.json"), str(raw)), calls


def _no_reply(code, exit_code=7):
    # curl -s wrote no -o file and, muted, no error text: only its exit code tells why (7 = could not connect)
    return lambda argv: SimpleNamespace(stdout=code, stderr="", returncode=exit_code)


def test_an_unreachable_router_is_not_read_as_the_previous_403(monkeypatch, tmp_path):
    once, _ = _setup(monkeypatch, tmp_path, OLD_403, _no_reply("000", 7))
    res = once()
    assert res["http"] == "000" and res["text"] == ""
    assert not router._is_403(res)                       # a router outage, not a locked account
    assert res["errors"][0]["transport"] == "no reply body from the router (http 000, curl exit 7)"


def test_a_reply_without_a_body_is_not_read_as_the_previous_text(monkeypatch, tmp_path):
    # headers came back (http 200), then the connection dropped before any body byte (curl exit 18)
    once, _ = _setup(monkeypatch, tmp_path, OLD_TEXT, _no_reply("200", 18))
    res = once()
    assert res["text"] == "" and res["errors"]           # a failed try, retried like any transport error


def test_a_new_answer_replaces_the_previous_one(monkeypatch, tmp_path):
    def answers(argv):
        with open(argv[argv.index("-o") + 1], "w", encoding="utf-8") as f:
            f.write(NEW_TEXT)
        return SimpleNamespace(stdout="200", stderr="")

    once, _ = _setup(monkeypatch, tmp_path, OLD_TEXT, answers)
    res = once()
    assert res["text"] == "câu mới" and res["http"] == "200" and not res["errors"]


def test_a_previous_answer_that_cannot_be_removed_fails_the_try_without_a_call(monkeypatch, tmp_path):
    # a Windows sharing violation (AV scanner) on the old answer: reading it would pass it off as this try's
    once, calls = _setup(monkeypatch, tmp_path, OLD_TEXT, _no_reply("000"))
    real_remove = router.os.remove

    def remove(path):
        if str(path).endswith("_raw.sse"):
            raise OSError("sharing violation")
        real_remove(path)

    monkeypatch.setattr(router.os, "remove", remove)
    res = once()
    assert res["text"] == "" and res["errors"] and calls == []
    assert "could not clear the previous answer" in res["errors"][0]["transport"]


def _request_write_raises(monkeypatch, error):
    real_open = builtins.open

    def fake_open(path, mode="r", *a, **kw):
        if str(path).endswith("_req.json") and "w" in mode:
            raise error
        return real_open(path, mode, *a, **kw)

    monkeypatch.setattr(builtins, "open", fake_open)


@pytest.mark.parametrize("error", [PermissionError(errno.EACCES, "Permission denied"),
                                   IsADirectoryError(errno.EISDIR, "Is a directory")])
def test_a_request_file_that_cannot_be_written_fails_the_try_without_a_call(monkeypatch, tmp_path, error):
    # a read-only, locked or directory leftover at the request path (a reused folder) must not crash the run
    once, calls = _setup(monkeypatch, tmp_path, OLD_TEXT, _no_reply("000"))
    _request_write_raises(monkeypatch, error)
    res = once()
    assert res["text"] == "" and calls == []
    assert res["errors"][0]["transport"] == f"could not write the request 00_req.json: {error.strerror}"


def test_an_unreadable_chunk_file_stops_the_run_and_is_not_blamed_on_the_request(monkeypatch, tmp_path):
    once, calls = _setup(monkeypatch, tmp_path, OLD_TEXT, _no_reply("000"))
    real_open = builtins.open

    def fake_open(path, mode="r", *a, **kw):
        if str(path).endswith(".mp3"):
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return real_open(path, mode, *a, **kw)

    monkeypatch.setattr(builtins, "open", fake_open)
    with pytest.raises(PermissionError, match=r"00\.mp3"):
        once()
    assert calls == []


def test_a_full_disk_still_stops_the_run_with_its_own_message(monkeypatch, tmp_path):
    # retrying cannot free the disk: the run stops loudly, as before, instead of ending as a "router" failure
    once, calls = _setup(monkeypatch, tmp_path, OLD_TEXT, _no_reply("000"))
    _request_write_raises(monkeypatch, OSError(errno.ENOSPC, "No space left on device"))
    with pytest.raises(OSError, match="No space left on device"):
        once()
    assert calls == []


def test_a_try_failed_on_an_answer_it_could_not_clear_is_retried_and_can_succeed(monkeypatch, tmp_path):
    mp3 = tmp_path / "00.mp3"
    mp3.write_bytes(b"ID3")
    raw = tmp_path / "00_raw.sse"
    raw.write_text(OLD_TEXT, encoding="utf-8")
    monkeypatch.setattr(router.config, "BASE_URL", "https://router.invalid/v1")
    monkeypatch.setattr(router.config, "API_KEY", "k")
    monkeypatch.setattr(router.time, "sleep", lambda s: None)
    real_remove, raw_removals, calls = router.os.remove, [], []

    def remove(path):   # the first removal of the old answer hits a sharing violation, later ones work
        if str(path).endswith("_raw.sse"):
            raw_removals.append(path)
            if len(raw_removals) == 1:
                raise PermissionError("sharing violation")
        real_remove(path)

    def answers(argv, **kw):
        calls.append(argv)
        with open(argv[argv.index("-o") + 1], "w", encoding="utf-8") as f:
            f.write(NEW_TEXT)
        return SimpleNamespace(stdout="200", stderr="", returncode=0)

    monkeypatch.setattr(router.os, "remove", remove)
    monkeypatch.setattr(router.subprocess, "run", answers)
    # 5 s of speech: below the text guards' floor, so a short answer passes
    r = router.transcribe_with_retry(str(mp3), "prompt", 5.0, str(tmp_path / "00_req.json"), str(raw))
    assert r["attempts"] == 2 and r["reasons"] == [] and len(calls) == 1
    assert r["history"][0]["outcome"] == "error" and r["res"]["text"] == "câu mới"
