import unicodedata
from realign.text_metrics import char_count, wrap_two_lines


def test_char_count_counts_vietnamese_syllable_as_one():
    assert char_count("Một thế") == 7  # M o t _ t h e (diacritic = 1 char)


def test_char_count_equal_for_nfc_and_nfd():
    s = "Đúng thế?"
    assert char_count(unicodedata.normalize("NFC", s)) == char_count(unicodedata.normalize("NFD", s))


def test_char_count_ignores_newline():
    assert char_count("abc\ndef") == 6


def test_wrap_short_text_stays_one_line():
    assert "\n" not in wrap_two_lines("ngan gon", 42)


def test_wrap_long_text_two_lines_bottom_heavy_within_cpl():
    text = " ".join(["tu"] * 20)  # 20*2 + 19 spaces = 59 chars, > 42 -> must wrap to 2 lines
    out = wrap_two_lines(text, 42)
    lines = out.split("\n")
    assert len(lines) == 2
    assert char_count(lines[0]) <= 42 and char_count(lines[1]) <= 42
    assert char_count(lines[1]) >= char_count(lines[0])          # bottom-heavy
    assert out.replace("\n", " ") == text                        # no words lost
