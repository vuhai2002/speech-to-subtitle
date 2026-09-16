import json
from realign.segment_cues import segment


def test_segment_end_to_end(tmp_path):
    wj = tmp_path / "w.json"
    json.dump(
        {"key": "k", "audio": "a.mp3", "words": [
            {"w": "Một.", "start": 0.0, "end": 2.0, "score": 0.9},
            {"w": "Hai.", "start": 2.1, "end": 4.0, "score": 0.9}]},
        open(wj, "w", encoding="utf-8"), ensure_ascii=False)
    out = tmp_path / "o.srt"
    cues = segment(str(wj), str(out))
    assert len(cues) == 2
    text = open(out, encoding="utf-8").read()
    assert text.startswith("1\n") and "-->" in text and "Một." in text
    assert all(cues[i]["start"] <= cues[i + 1]["start"] for i in range(len(cues) - 1))
