import realign.config as cfg
from realign.text_metrics import char_count
from realign.cue_builder import fill_word_times, build_cues


def W(w, s, e, score=0.9):
    return {"w": w, "start": s, "end": e, "score": score}


def test_fill_word_times_interpolates_none():
    out = fill_word_times([W("a", 0.0, 0.5), W("b", None, None), W("c", 1.0, 1.5)])
    assert out[1]["start"] == 0.5 and out[1]["end"] >= 0.5


def test_sentence_split_when_each_cue_meets_min_display():
    # 2 sentences, each cue long enough (>=1.5s) -> NOT merged -> 2 cues
    cues = build_cues([W("Một.", 0.0, 2.0), W("Hai.", 2.1, 4.0)], cfg)
    assert len(cues) == 2
    assert cues[0]["text"] == "Một." and cues[1]["text"] == "Hai."


def test_long_pause_splits_into_two_cues():
    # LONG pause 2.4s >= LONG_PAUSE -> hard split
    cues = build_cues([W("alpha", 0.0, 1.6), W("beta", 4.0, 5.6)], cfg)
    assert len(cues) == 2


def test_medium_pause_does_not_split_short_sentence():
    # medium pause 0.8s (>= PAUSE_SPLIT, < LONG_PAUSE) does NOT split a short sentence
    words = [W("alpha", 0.0, 0.4), W("beta", 0.5, 0.9),
             W("gamma", 1.7, 2.1), W("delta", 2.2, 2.6)]   # gap beta->gamma = 0.8s
    cues = build_cues(words, cfg)
    assert len(cues) == 1


def test_deisolate_leading_misaligned_word():
    # leading word "Nam" wrongly anchored at 8.3s, body at 54s (gap 44s) -> merged, use body timing
    body = ["Mô", "Bổn", "Sư", "Phật."]
    words = [W("Nam", 8.3, 9.8)] + [W(w, 54.0 + i * 0.2, 54.0 + i * 0.2 + 0.15)
                                    for i, w in enumerate(body)]
    cues = build_cues(words, cfg)
    assert len(cues) == 1
    assert cues[0]["text"].replace("\n", " ").startswith("Nam Mô")
    assert cues[0]["start"] >= 50.0


def test_long_run_splits_to_keep_lines_within_cpl():
    words = [W(f"w{i:02d}", i * 1.0, i * 1.0 + 0.8) for i in range(40)]  # force splits to keep <= 2 lines <= CPL
    cues = build_cues(words, cfg)
    assert len(cues) >= 2
    for c in cues:
        assert all(char_count(ln) <= cfg.CPL_MAX for ln in c["text"].split("\n"))


def test_cps_floor_extends_short_cue_into_gap():
    # 1 cue ~60 chars needs >= 60/15 = 4.0s; audio is only 1s; long gap after -> extend
    text_words = ["chu" + str(i) for i in range(12)]  # ~ 12*5+11 = 71 chars, 1 cue
    words = [W(w, 0.0 + i * 0.08, 0.0 + i * 0.08 + 0.07) for i, w in enumerate(text_words)]
    # ends ~0.96s, no internal sentence/pause -> 1 cue; no following cue -> cap = start+DUR_MAX
    cues = build_cues(words, cfg)
    assert len(cues) == 1
    dur = cues[0]["end"] - cues[0]["start"]
    chars = char_count(cues[0]["text"])
    assert dur >= chars / cfg.CPS_MAX - 1e-6   # reaches the reading-speed ceiling


def test_tiny_back_to_back_cue_merged_no_flash():
    # short sentence "Vâng." 0.4s, immediately followed by a long sentence, NO gap -> merge, do not leave a cue < 1.5s
    words = [W("Vâng.", 0.0, 0.4)] + [W(f"x{i}", 0.45 + i * 0.5, 0.45 + i * 0.5 + 0.45) for i in range(5)]
    cues = build_cues(words, cfg)
    for c in cues:
        assert (c["end"] - c["start"]) >= cfg.DUR_MIN - 1e-6   # no more flashing cue


def test_merge_never_creates_over_cpl_line():
    # tiny cue "Vâng." right before a long sentence, no gap -> must not merge into a cue > CPL
    words = [W("Vâng.", 0.0, 0.05)] + [W("abc", 0.1 + i * 0.5, 0.1 + i * 0.5 + 0.45) for i in range(20)]
    cues = build_cues(words, cfg)
    for c in cues:
        assert all(char_count(ln) <= cfg.CPL_MAX for ln in c["text"].split("\n"))


def test_single_overlong_word_does_not_crash():
    cues = build_cues([W("x" * 50, 0.0, 3.0)], cfg)
    assert len(cues) == 1


def test_no_words_returns_empty():
    assert build_cues([], cfg) == []


def test_long_sentence_splits_balanced_at_comma():
    text = "Hôm nay chúng ta nghe một đoạn pháp cú ngắn thôi, rồi sau đó chúng ta nói qua chuyện khác."
    toks = text.split()
    words = [W(t, i * 0.4, i * 0.4 + 0.35) for i, t in enumerate(toks)]   # continuous, no pause
    cues = build_cues(words, cfg)
    assert len(cues) == 2
    assert cues[0]["text"].replace("\n", " ").rstrip().endswith(",")      # split at the comma
    for c in cues:                                                        # no stubby tail
        assert len(c["text"].replace("\n", " ").split()) >= 3


def test_boundary_merge_can_be_turned_off_per_end():
    # A 1-2 word cue isolated by >= ISOLATION_GAP at either end is merged into its neighbour by default (it is
    # usually a misaligned word). A caller that knows such a cue is real text at its own time can keep it.
    words = ([W("Mô", 1.0, 1.4), W("Phật.", 1.5, 1.9)]
             + [W(w, 60.0 + i * 0.3, 60.25 + i * 0.3) for i, w in enumerate(["Hôm", "nay", "ta", "học."])]
             + [W("Nam", 120.0, 120.4), W("mô.", 120.5, 120.9)])
    assert len(build_cues(words, cfg)) == 1
    assert [c["start"] for c in build_cues(words, cfg, merge_head=False, merge_tail=False)] == [1.0, 60.0, 120.0]
    assert [c["start"] for c in build_cues(words, cfg, merge_tail=False)] == [60.0, 120.0]


def test_a_short_cue_can_be_kept_from_merging_back_across_a_long_pause():
    # "Mô Phật." cannot reach DUR_MIN before the next line, so it is merged. By default into the previous cue,
    # however far back; with max_back_gap a distant previous cue is skipped and it joins the next line instead.
    words = ([W(w, 0.0 + i * 0.4, 0.3 + i * 0.4) for i, w in enumerate(["Hôm", "nay", "ta", "học", "tới", "đây."])]
             + [W("Mô", 100.0, 100.3), W("Phật.", 100.3, 100.6)]
             + [W(w, 101.0 + i * 0.2, 101.15 + i * 0.2) for i, w in enumerate(["Nam", "mô", "Bổn", "Sư."])])
    default = build_cues(words, cfg)
    assert default[0]["text"].replace("\n", " ").endswith("Mô Phật.")      # pulled back ~100 s
    guarded = build_cues(words, cfg, max_back_gap=cfg.ISOLATION_GAP)
    assert [(c["start"], c["text"].replace("\n", " ")) for c in guarded][1:] == [(100.0, "Mô Phật. Nam mô Bổn Sư.")]
