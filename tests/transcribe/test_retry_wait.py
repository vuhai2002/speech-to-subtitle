"""Wait before the next attempt: doubling backoff with jitter, Retry-After on 429/503, hard cap."""
from transcribe.retry_wait import retry_after_from_headers, wait_seconds


def _mid():
    return 0.5          # jitter factor 0.5 + 0.5 = 1.0 -> exact base * 2**(attempt-1)


def test_backoff_doubles_per_attempt():
    assert wait_seconds(1, 8.0, "500", None, rand=_mid) == 8.0
    assert wait_seconds(2, 8.0, "500", None, rand=_mid) == 16.0
    assert wait_seconds(3, 8.0, "500", None, rand=_mid) == 32.0


def test_jitter_spreads_between_half_and_one_and_a_half():
    assert wait_seconds(1, 8.0, "500", None, rand=lambda: 0.0) == 4.0
    assert wait_seconds(1, 8.0, "500", None, rand=lambda: 0.999) < 12.0


def test_retry_after_longer_than_backoff_wins_on_429_and_503():
    assert wait_seconds(1, 8.0, "429", "30", rand=_mid) == 30.0
    assert wait_seconds(1, 8.0, "503", " 45 ", rand=_mid) == 45.0


def test_retry_after_shorter_than_backoff_is_ignored():
    assert wait_seconds(1, 8.0, "429", "2", rand=_mid) == 8.0


def test_retry_after_is_ignored_for_other_statuses():
    assert wait_seconds(1, 8.0, "500", "30", rand=_mid) == 8.0
    assert wait_seconds(1, 8.0, None, "30", rand=_mid) == 8.0


def test_http_date_retry_after_falls_back_to_backoff():
    assert wait_seconds(1, 8.0, "503", "Wed, 21 Oct 2026 07:28:00 GMT", rand=_mid) == 8.0


def test_wait_never_exceeds_the_cap():
    assert wait_seconds(1, 8.0, "429", "600", rand=_mid) == 120.0
    assert wait_seconds(10, 8.0, "500", None, rand=_mid) == 120.0


def test_headers_last_retry_after_wins_across_redirect_blocks(tmp_path):
    p = tmp_path / "raw.headers"
    p.write_text("HTTP/1.1 301 Moved\r\nRetry-After: 5\r\n\r\n"
                 "HTTP/2 429\r\nretry-after: 42\r\ncontent-type: application/json\r\n\r\n", encoding="utf-8")
    assert retry_after_from_headers(str(p)) == "42"


def test_headers_missing_file_or_header_gives_none(tmp_path):
    assert retry_after_from_headers(str(tmp_path / "none.headers")) is None
    p = tmp_path / "raw.headers"
    p.write_text("HTTP/2 200\r\ncontent-type: text/event-stream\r\n\r\n", encoding="utf-8")
    assert retry_after_from_headers(str(p)) is None
