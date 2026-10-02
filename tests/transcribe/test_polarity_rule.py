"""The polarity rule of spec section 3 and the output check of section 4, on synthetic per-second statistics."""
import numpy as np

from transcribe.chunked_transcribe import polarity

NORMAL = (-20, -20, -20, -60)        # L, R, (L+R)/2, (L-R)/2 in dB: identical channels
INVERTED = (-20, -20, -80, -20)      # R = -L
SILENT = (-90, -90, -90, -90)
MISMATCHED_INVERTED = (-20, -23, -36.7, -21.4)   # R = -0.7 L (Xây Dựng 0:02-0:05): D - S = +15 dB
TRUE_STEREO = (-20, -20, -24.0, -22.2)           # correlation -0.2 (chanting): D - S = +1.8 dB
ONE_CHANNEL = (-20, -90, -26.0, -26.0)           # right channel silent: D - S = 0


def stats(seconds, quiet_blocks=()):
    """Stats from per-second dB tuples; 20 blocks per second at the louder channel's energy, except quiet_blocks."""
    a = np.array(seconds, dtype=float)
    e = lambda v: 10 ** (v / 10)   # noqa: E731
    blocks = np.repeat(np.maximum(e(a[:, 0]), e(a[:, 1])), 20)
    for i in quiet_blocks:
        blocks[i] = 1e-12
    return polarity.Stats(left=e(a[:, 0]), right=e(a[:, 1]), mix=e(a[:, 2]), diff=e(a[:, 3]), blocks=blocks)


def test_normal_source_has_no_section():
    assert polarity.find_sections(stats([NORMAL] * 30)) == []


def test_fully_inverted_source_is_one_section_to_the_end():
    assert polarity.find_sections(stats([INVERTED] * 30)) == [(0.0, None)]


def test_inverted_then_normal_flips_up_to_the_silent_seam():
    secs = [INVERTED] * 60 + [SILENT] * 3 + [NORMAL] * 57
    (start, end), = polarity.find_sections(stats(secs))
    assert start == 0.0
    assert 60.0 <= end <= 62.0          # inside the silence, never in speech


def test_seam_moves_to_the_quietest_block():
    secs = [INVERTED] * 40 + [NORMAL] * 40
    quiet = 39 * 20 + 7                 # block 7 of second 39 -> centre 39.375 s
    assert polarity.find_sections(stats(secs, quiet_blocks=[quiet])) == [(0.0, 39.375)]


def test_inverted_with_mismatched_channel_levels_is_flipped():
    assert polarity.find_sections(stats([MISMATCHED_INVERTED] * 20)) == [(0.0, None)]


def test_true_stereo_and_one_silent_channel_are_left_alone():
    assert polarity.find_sections(stats([TRUE_STEREO] * 20)) == []
    assert polarity.find_sections(stats([ONE_CHANNEL] * 20)) == []


def test_silence_only_has_no_section():
    assert polarity.find_sections(stats([SILENT] * 10)) == []


def test_short_inverted_blip_is_dropped_but_five_seconds_count():
    assert polarity.find_sections(stats([NORMAL] * 10 + [INVERTED] * 3 + [NORMAL] * 10)) == []
    (start, end), = polarity.find_sections(stats([NORMAL] * 10 + [INVERTED] * 5 + [NORMAL] * 10))
    assert 9.0 <= start <= 11.0 and 14.0 <= end <= 16.0


def test_short_normal_blip_inside_an_inverted_section_is_merged():
    assert polarity.find_sections(stats([INVERTED] * 20 + [NORMAL] * 2 + [INVERTED] * 20)) == [(0.0, None)]
    assert len(polarity.find_sections(stats([INVERTED] * 20 + [NORMAL] * 3 + [INVERTED] * 20))) == 2


def test_flip_filter_ramps_only_inside_the_file():
    assert polarity.flip_filter([(0.0, None)]) == \
        "aeval=exprs='val(0)|val(1)*(1-2*(1*1))',pan=stereo|c0=c0|c1=c1"
    assert polarity.flip_filter([(10.0, 20.0)]) == (
        "aeval=exprs='val(0)|val(1)*(1-2*(clip((t-(9.990))/0.02,0,1)*clip(((20.010)-t)/0.02,0,1)))',"
        "pan=stereo|c0=c0|c1=c1")


def test_check_flags_five_cancelled_seconds_only():
    louder = np.full(12, -20.0)
    mix = np.full(12, -20.0)
    mix[3:8] = -40.0
    assert polarity.cancelled_stretches(mix, louder) == [(3, 8)]
    mix[3:8] = -20.0
    mix[3:7] = -40.0
    assert polarity.cancelled_stretches(mix, louder) == []


def test_check_skips_silent_seconds():
    louder = np.array([-20, -20, -90, -20, -20, -90, -20, -20.0])
    mix = np.array([-40, -40, -90, -40, -40, -90, -40, -20.0])
    assert polarity.cancelled_stretches(mix, louder) == [(0, 7)]


def test_audio_cancelled_names_the_stretches():
    e = polarity.AudioCancelled([(3, 8), (3600, 3725)])
    assert str(e) == "the mono mix is cancelled in 00:00:03-00:00:08, 01:00:00-01:02:05"
    assert e.stretches == [(3, 8), (3600, 3725)]
