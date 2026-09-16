from webui.__main__ import parse_args


def test_defaults():
    a = parse_args([])
    assert a.host == "127.0.0.1" and a.port == 8000 and a.open is True


def test_no_open_and_port():
    a = parse_args(["--no-open", "--port", "9001"])
    assert a.open is False and a.port == 9001
