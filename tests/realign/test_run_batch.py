import json
import shutil
from unittest.mock import patch
from realign.run_batch import out_paths, should_align, select_pairs, copy_with_retry


def test_out_paths_keeps_original_srt_name():
    wj, srt = out_paths("out", "Đạo làm con B.srt")
    assert wj.replace("\\", "/") == "out/words/Đạo làm con B.json"
    assert srt.replace("\\", "/") == "out/srt/Đạo làm con B.srt"


def test_should_align_skips_when_json_exists(tmp_path):
    j = tmp_path / "w.json"
    assert should_align(str(j), force=False) is True
    j.write_text("{}", encoding="utf-8")
    assert should_align(str(j), force=False) is False
    assert should_align(str(j), force=True) is True


def test_select_pairs_sample_is_deterministic():
    pairs = [{"key": f"k{i}", "srt_name": f"{i}.srt"} for i in range(5)]
    a = select_pairs(pairs, sample=2, seed=0)
    b = select_pairs(pairs, sample=2, seed=0)
    assert len(a) == 2 and [p["key"] for p in a] == [p["key"] for p in b]


def test_select_pairs_limit_and_only():
    pairs = [{"key": "alpha", "srt_name": "a.srt"}, {"key": "beta", "srt_name": "b.srt"}]
    assert len(select_pairs(pairs, limit=1)) == 1
    assert [p["key"] for p in select_pairs(pairs, only="bet")] == ["beta"]


def test_copy_with_retry_copies(tmp_path):
    src = tmp_path / "s.bin"; src.write_bytes(b"hello")
    dst = tmp_path / "d.bin"
    copy_with_retry(str(src), str(dst))
    assert dst.read_bytes() == b"hello"


def test_copy_with_retry_retries_then_succeeds(tmp_path):
    src = tmp_path / "s.bin"; src.write_bytes(b"x")
    dst = tmp_path / "d.bin"
    real_copy = shutil.copyfile
    calls = {"n": 0}

    def flaky(a, b):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("drive blip")
        return real_copy(a, b)

    with patch("shutil.copyfile", side_effect=flaky):
        copy_with_retry(str(src), str(dst), attempts=3, sleep=0)
    assert calls["n"] == 2
    assert dst.read_bytes() == b"x"
