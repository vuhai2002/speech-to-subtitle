"""chunk_has_speech: which chunks count as real speech (shared by both transcribe paths)."""
from transcribe.chunked_transcribe.audio_utils import chunk_has_speech


def _chunk(start, end, speech):
    return {"start": start, "end": end, "speech_sec": speech}


def test_short_speech_dominated_clip_is_kept():
    # The reported bug: a 42s clip with 28s speech (67%) was dropped as "fabricated"
    # because 28 < the 30s floor. The ratio rule now keeps it.
    assert chunk_has_speech(_chunk(0, 42, 28)) is True


def test_long_trailing_silence_chunk_is_dropped():
    # A 10-minute chunk with only 20s of speech (3%) is near-silent -> drop (hallucination guard).
    assert chunk_has_speech(_chunk(0, 600, 20)) is False


def test_long_chunk_with_ample_speech_is_kept():
    assert chunk_has_speech(_chunk(0, 600, 500)) is True         # above the absolute floor


def test_absolute_floor_alone_is_enough():
    assert chunk_has_speech(_chunk(0, 1000, 30)) is True         # 30s speech, ratio tiny, but >= floor


def test_ratio_boundary_is_inclusive():
    assert chunk_has_speech(_chunk(0, 100, 50)) is True          # exactly 0.5 ratio


def test_tiny_speech_short_clip_is_dropped():
    assert chunk_has_speech(_chunk(0, 43, 1)) is False           # 2% speech, under floor -> silence
