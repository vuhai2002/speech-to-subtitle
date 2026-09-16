from realign.vad_speech_region import region_from_timestamps


def test_region_from_timestamps_first_and_last():
    ts = [{"start": 1600, "end": 3200}, {"start": 8000, "end": 16000}]
    assert region_from_timestamps(ts, 16000) == (0.1, 1.0)


def test_region_empty_returns_none():
    assert region_from_timestamps([], 16000) is None
