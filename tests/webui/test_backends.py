import sys
from webui.backends import (BACKENDS, output_allowed, build_steps, Step, gpu_name,
                            _classify_probe, probe_backend)


def test_registry_shape():
    assert set(BACKENDS) == {"vertex", "router", "mai"}
    assert BACKENDS["mai"]["needs_gpu_for_srt"] is False
    assert BACKENDS["router"]["needs_gpu_for_srt"] is True
    assert BACKENDS["mai"]["has_prompt"] is False


def test_output_gating():
    assert output_allowed("mai", "txt", has_gpu=False)
    assert output_allowed("mai", "srt", has_gpu=False)          # native timestamps
    assert output_allowed("router", "txt", has_gpu=False)
    assert not output_allowed("router", "srt", has_gpu=False)   # needs GPU
    assert output_allowed("router", "srt", has_gpu=True)


def test_build_steps_mai_srt():
    steps = build_steps("mai", "srt", "a.m4a", "out/webui/j1", model="microsoft/mai-transcribe-2", workers=3)
    assert [s.label for s in steps] == ["transcribe", "build srt"]
    assert steps[0].argv == [sys.executable, "-m", "transcribe.mai_transcribe.run_pipeline",
                             "--input", "a.m4a", "--out-dir", "out/webui/j1",
                             "--model", "microsoft/mai-transcribe-2", "--workers", "3"]
    assert steps[1].argv == [sys.executable, "-m", "transcribe.mai_transcribe.build_srt",
                             "--out-dir", "out/webui/j1"]


def test_build_steps_router_txt_with_prompt():
    steps = build_steps("router", "txt", "a.m4a", "o", prompt_file="/tmp/p.txt")
    assert len(steps) == 1
    assert "--prompt-file" in steps[0].argv
    assert steps[0].argv[steps[0].argv.index("--prompt-file") + 1] == "/tmp/p.txt"


def test_gpu_name_returns_str():
    assert isinstance(gpu_name(), str)


def test_classify_probe_success_and_bad_key():
    assert _classify_probe("router", 200, "HTTP 200")[0] is True
    assert _classify_probe("mai", 204, "HTTP 204")[0] is True
    # 401 is a rejected key for every backend.
    assert _classify_probe("router", 401, "HTTP 401")[0] is False
    assert _classify_probe("mai", 401, "HTTP 401")[0] is False


def test_classify_probe_router_403_is_accepted():
    # 9router authenticates the key but forbids listing models -> still valid.
    ok, msg = _classify_probe("router", 403, "HTTP 403")
    assert ok is True and "transcription" in msg
    # For other backends a 403 is not a known "auth ok" case.
    assert _classify_probe("mai", 403, "HTTP 403")[0] is False


def test_classify_probe_transport_error():
    ok, msg = _classify_probe("router", 0, "getaddrinfo failed")
    assert ok is False and "getaddrinfo failed" in msg


def test_probe_backend_requires_key_without_network():
    # Missing credentials must fail fast, before any network call.
    assert probe_backend("mai", {}) == (False, "No API key set.")
    assert probe_backend("router", {"ROUTER_API_KEY": "k"}) == (False, "No Base URL set.")
    assert probe_backend("router", {"ROUTER_BASE_URL": "https://x/v1"}) == (False, "No API key set.")
    assert probe_backend("nope", {}) == (False, "unknown backend")
