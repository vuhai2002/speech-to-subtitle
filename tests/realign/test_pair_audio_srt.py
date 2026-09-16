from realign.pair_audio_srt import (
    normalize_key, build_pairs, format_unpaired_report, index_audio,
)


def test_normalize_key_strips_diacritics_and_lowercases():
    assert normalize_key("Đạo làm con B") == "dao lam con b"


def test_normalize_key_collapses_whitespace():
    assert normalize_key("  Tình   yêu  Tổ quốc ") == "tinh yeu to quoc"


def test_build_pairs_matches_case_and_diacritic_insensitive(tmp_path):
    srt_dir = tmp_path / "srt"; srt_dir.mkdir()
    audio = tmp_path / "a1"; audio.mkdir()
    (srt_dir / "Đạo làm con B.srt").write_text("x", encoding="utf-8")
    (srt_dir / "Khong co audio.srt").write_text("x", encoding="utf-8")
    (audio / "đạo làm con b.mp3").write_bytes(b"")     # different case + diacritics still match
    (audio / "Bai le thua.mp3").write_bytes(b"")       # no srt -> unpaired_audio

    pairs, unpaired_srt, unpaired_audio = build_pairs(str(srt_dir), [str(audio)])

    assert len(pairs) == 1
    assert pairs[0]["key"] == "dao lam con b"
    assert pairs[0]["srt_name"] == "Đạo làm con B.srt"
    assert pairs[0]["kind"] == "srt"
    assert pairs[0]["audio_path"].endswith("đạo làm con b.mp3")
    assert [u["srt_name"] for u in unpaired_srt] == ["Khong co audio.srt"]
    assert [u["audio_name"] for u in unpaired_audio] == ["Bai le thua.mp3"]


def test_build_pairs_txt_mode_pairs_txt_and_tags_kind(tmp_path):
    txt_dir = tmp_path / "txt"; txt_dir.mkdir()
    audio = tmp_path / "a1"; audio.mkdir()
    (txt_dir / "Góp nhặt cát đá.txt").write_text("Nam Mô", encoding="utf-8")
    (txt_dir / "Bỏ qua cái srt.srt").write_text("x", encoding="utf-8")   # .srt is skipped when ext=.txt
    (audio / "gop nhat cat da.mp3").write_bytes(b"")

    pairs, us, ua = build_pairs(str(txt_dir), [str(audio)], ext=".txt")

    assert len(pairs) == 1
    assert pairs[0]["srt_name"] == "Góp nhặt cát đá.txt"
    assert pairs[0]["kind"] == "txt"
    assert pairs[0]["audio_path"].endswith("gop nhat cat da.mp3")


def test_format_unpaired_report_lists_both():
    txt = format_unpaired_report(
        [{"srt_name": "A.srt", "key": "a"}],
        [{"audio_name": "B.mp3", "key": "b"}],
    )
    assert "A.srt" in txt and "B.mp3" in txt


def test_index_audio_skips_missing_dir(tmp_path):
    real = tmp_path / "a"; real.mkdir()
    (real / "Bai.mp3").write_bytes(b"")
    idx = index_audio([str(real), str(tmp_path / "khong-ton-tai")])
    assert idx == {"bai": str(real / "Bai.mp3")}


def test_index_audio_dedupes_by_normalized_key(tmp_path):
    d = tmp_path / "a"; d.mkdir()
    (d / "Bài.mp3").write_bytes(b"")   # different diacritics but same key "bai"
    (d / "Bai.mp3").write_bytes(b"")
    idx = index_audio([str(d)])
    assert list(idx.keys()) == ["bai"]
