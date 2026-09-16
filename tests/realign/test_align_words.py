"""Unit tests for realign.align_words - pure helpers only, no GPU required."""
from collections import namedtuple

from realign.align_words import normalize_word, assemble_words, _chunk_bounds

TS = namedtuple("TS", ["start", "end", "score"])


def test_normalize_word_vietnamese():
    assert normalize_word("Đúng") == "dung"
    assert normalize_word("thế?") == "the"
    assert normalize_word("...") == ""


def test_assemble_words_maps_spans_and_marks_unaligned_none():
    words_raw = ["Một", "[nhạc]", "Hai."]
    align_idx = [0, 2]                       # word 1 could not be aligned
    token_spans = [[TS(0, 10, 0.9)], [TS(20, 30, 0.8)]]
    out = assemble_words(words_raw, align_idx, token_spans, ratio=1.0, sr=100)
    assert out[0]["w"] == "Một" and abs(out[0]["start"] - 0.0) < 1e-9 and abs(out[0]["end"] - 0.1) < 1e-9
    assert out[1]["start"] is None and out[1]["end"] is None and out[1]["score"] is None
    assert abs(out[2]["start"] - 0.2) < 1e-9 and abs(out[2]["score"] - 0.8) < 1e-9


def test_chunk_bounds_absorbs_tiny_tail():
    # tail under 10 (< min_tail 15) must be merged into the last chunk -> no tiny chunk created
    b = _chunk_bounds(100, 30, 15)
    assert b == [(0, 30), (30, 60), (60, 100)]
    assert all(e - s >= 15 for s, e in b)
    assert b[-1][1] == 100                                   # covers all n_samples


def test_chunk_bounds_exact_multiple_and_short_audio():
    assert _chunk_bounds(90, 30, 15) == [(0, 30), (30, 60), (60, 90)]   # divides evenly
    assert _chunk_bounds(5, 30, 15) == [(0, 5)]                          # audio shorter than 1 chunk
