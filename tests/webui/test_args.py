import socket

from webui.__main__ import parse_args, find_free_port


def test_defaults():
    a = parse_args([])
    assert a.host == "127.0.0.1" and a.port == 8000 and a.open is True


def test_no_open_and_port():
    a = parse_args(["--no-open", "--port", "9001"])
    assert a.open is False and a.port == 9001


def test_find_free_port_skips_busy():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    s.listen()
    busy = s.getsockname()[1]
    try:
        p = find_free_port("127.0.0.1", busy)
        assert isinstance(p, int) and p != busy
    finally:
        s.close()
