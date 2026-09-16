from realign.srt_writer import fmt_ts, cues_to_srt


def test_fmt_ts():
    assert fmt_ts(0) == "00:00:00,000"
    assert fmt_ts(3661.5) == "01:01:01,500"
    assert fmt_ts(-1) == "00:00:00,000"


def test_cues_to_srt_format():
    srt = cues_to_srt([{"text": "Một.", "start": 0.0, "end": 2.0},
                       {"text": "Hai\nba.", "start": 2.1, "end": 4.0}])
    assert srt.startswith("1\n00:00:00,000 --> 00:00:02,000\nMột.\n\n")
    assert "2\n00:00:02,100 --> 00:00:04,000\nHai\nba.\n" in srt
