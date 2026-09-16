from realign.srt_to_txt import parse_srt_cues, srt_to_transcript, words_from_file

SRT = """1
00:00:01,000 --> 00:00:03,000
Dung the.

2
00:00:03,200 --> 00:00:06,000
Toi nghi
ban sai roi?
"""

def test_parse_basic_two_cues():
    assert parse_srt_cues(SRT) == ["Dung the.", "Toi nghi ban sai roi?"]

def test_multiline_cue_three_lines_joined_with_space():
    srt = "1\n00:00:00,000 --> 00:00:03,000\nMot\nhai\nba bon.\n"
    assert parse_srt_cues(srt) == ["Mot hai ba bon."]

def test_preserves_vietnamese_diacritics_and_punctuation():
    srt = "1\n00:00:00,000 --> 00:00:02,000\nKhông thể chỉnh sửa: Đúng chưa?\n"
    assert parse_srt_cues(srt) == ["Không thể chỉnh sửa: Đúng chưa?"]

def test_handles_crlf_and_bom():
    srt = "﻿1\r\n00:00:00,000 --> 00:00:01,000\r\nHello\r\n"
    assert parse_srt_cues(srt) == ["Hello"]

def test_transcript_joins_cues_with_newline():
    assert srt_to_transcript(SRT) == "Dung the.\nToi nghi ban sai roi?"

def test_words_from_file_txt_reads_raw_split(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("Nam Mô\nBổn Sư  Thích Ca", encoding="utf-8")   # newline + double space -> merged
    assert words_from_file(str(p), "txt") == ["Nam", "Mô", "Bổn", "Sư", "Thích", "Ca"]

def test_words_from_file_srt_parses_cues_then_splits(tmp_path):
    p = tmp_path / "t.srt"
    p.write_text(SRT, encoding="utf-8")
    assert words_from_file(str(p), "srt") == ["Dung", "the.", "Toi", "nghi", "ban", "sai", "roi?"]
