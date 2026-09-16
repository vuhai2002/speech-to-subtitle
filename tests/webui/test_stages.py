from webui.stages import stage_from_line


def test_markers():
    assert stage_from_line("10:46:19 [1/4] ffmpeg -> mono 16kHz") == "prepare"
    assert stage_from_line("10:46:42 [2/4] VAD + plan") == "plan"
    assert stage_from_line("10:47:39 [3/4] transcribe 11 chunks") == "transcribe"
    assert stage_from_line("10:49:02 [4/4] done: 11/11") == "assemble"
    assert stage_from_line("10:50:55 align per chunk (MMS, cuda)") == "align"
    assert stage_from_line("[stage] a.m4a -> out/_in") == "stage"
    assert stage_from_line("some unrelated line") is None
