import realign.config as cfg
from realign.cue_clamp import clamp_cues


def C(t, s, e):
    return {"text": t, "start": s, "end": e}


def test_overlap_removed():
    out = clamp_cues([C("a", 0.0, 3.0), C("b", 2.5, 5.0)], None, cfg)
    assert out[0]["end"] <= out[1]["start"] - cfg.GAP_MIN + 1e-9


def test_small_gap_made_contiguous():
    out = clamp_cues([C("a", 0.0, 2.0), C("b", 2.3, 4.0)], None, cfg)  # gap 0.3 < 0.5
    assert abs(out[0]["end"] - (2.3 - cfg.GAP_MIN)) < 1e-6


def test_real_pause_preserved():
    out = clamp_cues([C("a", 0.0, 2.0), C("b", 3.0, 5.0)], None, cfg)  # gap 1.0 >= 0.5
    assert abs(out[0]["end"] - 2.0) < 1e-6


def test_dur_max_capped():
    out = clamp_cues([C("a", 0.0, 10.0)], None, cfg)
    assert abs(out[0]["end"] - cfg.DUR_MAX) < 1e-6


def test_floor_enforced():
    out = clamp_cues([C("a", 1.0, 1.0)], None, cfg)
    assert out[0]["end"] >= 1.0 + cfg.FLOOR - 1e-9


def test_vad_trailing_trim_but_keeps_min_display():
    out = clamp_cues([C("a", 80.0, 90.0)], (0.0, 85.0), cfg)  # last_speech 85
    assert out[0]["end"] <= 85.0 + cfg.VAD_PAD + 1e-9
    assert (out[0]["end"] - out[0]["start"]) >= cfg.DUR_MIN - 1e-9


def test_vad_leading_trim_pushes_start_to_speech_when_room():
    out = clamp_cues([C("a", 0.0, 6.0)], (2.0, 6.0), cfg)
    assert abs(out[0]["start"] - 2.0) < 1e-9


def test_vad_leading_trim_never_moves_start_backward():
    # first cue is short (end-start < DUR_MIN), first_speech > start -> start must NOT move backward/negative
    out = clamp_cues([C("a", 0.0, 1.0), C("b", 5.0, 7.0)], (0.3, 7.0), cfg)
    assert out[0]["start"] >= 0.0 and abs(out[0]["start"] - 0.0) < 1e-9
