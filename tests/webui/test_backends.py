import sys
from webui.backends import BACKENDS, output_allowed, build_steps, Step


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
