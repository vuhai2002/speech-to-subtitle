"""Sentences of a chunk VAD hears as silent are scored by their mean MMS word score and kept at the threshold."""
import pytest

from transcribe.chunked_transcribe import silent_scoring as ss


def _al(words, scores):
    out = []
    for i, (w, sc) in enumerate(zip(words, scores)):
        if sc is None:
            out.append({"w": w, "start": None, "end": None, "score": None})
        else:
            out.append({"w": w, "start": float(i), "end": i + 0.8, "score": sc})
    return out


def test_split_at_sentence_punctuation_not_at_commas():
    assert ss.split_sentences("Tính làm Bồ Tát nào? Hả? Thôi, ráng tu học nha! Thế thôi; xong.") == [
        ["Tính", "làm", "Bồ", "Tát", "nào?"], ["Hả?"], ["Thôi,", "ráng", "tu", "học", "nha!"], ["Thế", "thôi;"],
        ["xong."]]


def test_a_run_on_without_punctuation_is_cut_into_pieces_of_40_words():
    words = [f"w{i}" for i in range(95)]
    parts = ss.split_sentences(" ".join(words))
    assert [len(p) for p in parts] == [40, 40, 15]
    assert [w for p in parts for w in p] == words


def test_split_keeps_every_word_including_decomposed_vietnamese():
    text = "Nam mô A Di Đà Phật.  Con xin quy y.\n"
    assert [w for s in ss.split_sentences(text) for w in s] == text.split()


def test_split_of_blank_text_is_empty():
    assert ss.split_sentences("  \n ") == []


def test_score_is_the_mean_of_aligned_word_scores_and_unaligned_words_are_ignored():
    r = ss.score_sentences([["a", "b", "c"]], _al(["a", "b", "c"], [0.6, None, 0.8]), 0.5)
    assert r[0]["score"] == 0.7 and r[0]["kept"] is True
    assert r[0]["start"] == 0.0 and r[0]["end"] == 2.8 and r[0]["text"] == "a b c"


def test_the_threshold_is_inclusive_and_judged_on_the_recorded_score():
    r = ss.score_sentences([["a"], ["b"]], _al(["a", "b"], [0.5, 0.4994]), 0.5)
    assert [(x["score"], x["kept"]) for x in r] == [(0.5, True), (0.499, False)]


def test_a_sentence_with_no_aligned_word_has_no_score_and_is_dropped():
    r = ss.score_sentences([["x", "y"]], _al(["x", "y"], [None, None]), 0.5)
    assert r[0]["score"] is None and r[0]["kept"] is False and r[0]["start"] is None and r[0]["end"] is None


def test_mismatched_word_counts_raise():
    with pytest.raises(ValueError):
        ss.score_sentences([["a", "b"]], _al(["a"], [0.9]), 0.5)


def test_each_result_carries_its_own_aligned_words_in_order():
    al = _al(["a", "b", "c"], [0.9, 0.9, 0.1])
    r = ss.score_sentences([["a", "b"], ["c"]], al, 0.5)
    assert r[0]["words"] == al[:2] and r[1]["words"] == al[2:]
    assert [x["kept"] for x in r] == [True, False]
