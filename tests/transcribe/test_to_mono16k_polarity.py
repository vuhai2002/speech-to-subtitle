"""to_mono16k end to end with real ffmpeg: inverted sections are flipped before the mixdown, the summary lands in
polarity.json, and a mix that still cancels raises AudioCancelled."""
import json
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from transcribe.chunked_transcribe import audio_utils, polarity

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
SR = 44100


def speech_like(seconds: float, seed: int = 1) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return np.random.default_rng(seed).standard_normal(t.size) * 0.1 * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))


def write(path, left, right=None):
    data = left if right is None else np.stack([left, right], axis=1)
    sf.write(str(path), data, SR, subtype="PCM_16")
    return str(path)


def mono_levels(path) -> list[float]:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    y = np.frombuffer(raw, dtype="<i2").astype(float) / 32768
    return [10 * np.log10(np.mean(y[i * 8000:(i + 1) * 8000] ** 2) + 1e-12) for i in range(len(y) // 8000)]


def summary_of(tmp_path) -> dict:
    return json.loads((tmp_path / audio_utils.POLARITY_FILE).read_text(encoding="utf-8"))


def test_partly_inverted_source_is_flipped_up_to_the_seam(tmp_path):
    x = speech_like(14)
    t = np.arange(x.size) / SR
    x[(t >= 6) & (t < 7)] = 0.0                       # silent seam
    src = write(tmp_path / "src.wav", x, np.where(t < 6.5, -x, x))
    dst = tmp_path / "mono16k.mp3"
    dur = audio_utils.to_mono16k(src, str(dst))
    assert 13.5 < dur < 14.5
    s = summary_of(tmp_path)
    assert s["channels"] == 2 and s["check"] == "passed"
    (start, end), = s["sections"]
    assert start == 0.0 and 6.0 <= end <= 7.0
    levels = mono_levels(dst)
    assert all(v > -40 for v in levels[:6]) and all(v > -40 for v in levels[7:13])


def test_fully_inverted_source_is_one_section(tmp_path):
    x = speech_like(8)
    audio_utils.to_mono16k(write(tmp_path / "src.wav", x, -x), str(tmp_path / "mono16k.mp3"))
    assert summary_of(tmp_path)["sections"] == [[0.0, None]]


def test_normal_source_has_no_section(tmp_path):
    x = speech_like(8)
    audio_utils.to_mono16k(write(tmp_path / "src.wav", x, x), str(tmp_path / "mono16k.mp3"))
    s = summary_of(tmp_path)
    assert s["sections"] == [] and s["check"] == "passed"


def test_late_starting_audio_is_flipped_at_the_right_samples(tmp_path):
    x = speech_like(14)
    t = np.arange(x.size) / SR
    x[(t >= 6) & (t < 7)] = 0.0
    wav = write(tmp_path / "src.wav", x, np.where(t < 6.5, -x, x))
    late = tmp_path / "late.mkv"   # the audio stream starts 3 s after the video stream
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=17",
                    "-itsoffset", "3", "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset",
                    "ultrafast", "-c:a", "pcm_s16le", str(late)], check=True)
    dst = tmp_path / "mono16k.mp3"
    audio_utils.to_mono16k(str(late), str(dst))
    (start, end), = summary_of(tmp_path)["sections"]
    assert start == 0.0 and 6.0 <= end <= 7.0     # seconds counted from the first audio sample
    assert all(v > -40 for v in mono_levels(dst)[:6])   # a timestamp-timed flip would leave seconds 3-5 cancelled


def test_mono_source_is_skipped(tmp_path):
    audio_utils.to_mono16k(write(tmp_path / "src.wav", speech_like(5)), str(tmp_path / "mono16k.mp3"))
    s = summary_of(tmp_path)
    assert s["channels"] == 1 and s["check"] == "skipped: mono source"


def test_silent_source_passes(tmp_path):
    z = np.zeros(5 * SR)
    audio_utils.to_mono16k(write(tmp_path / "src.wav", z, z), str(tmp_path / "mono16k.mp3"))
    assert summary_of(tmp_path)["check"] == "passed"


def test_unnamed_two_channel_layout_is_analysed(tmp_path):
    x = speech_like(8)
    wav = write(tmp_path / "src.wav", x, -x)
    raw = tmp_path / "unnamed.mka"   # PCM in matroska keeps "2 channels" unnamed (FLAC would name it stereo)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav, "-af", "aformat=channel_layouts=2c",
                    "-c:a", "pcm_s16le", str(raw)], check=True)
    layout = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                             "stream=channel_layout", "-of", "csv=p=0", str(raw)],
                            capture_output=True, text=True).stdout.strip()
    assert layout == "unknown"       # the fixture really has no named layout (checked 2026-10-02)
    audio_utils.to_mono16k(str(raw), str(tmp_path / "mono16k.mp3"))
    assert summary_of(tmp_path)["sections"] == [[0.0, None]]


def test_flip_bypass_fails_the_check(tmp_path, monkeypatch):
    x = speech_like(8)
    monkeypatch.setattr(polarity, "find_sections", lambda s: [])
    with pytest.raises(polarity.AudioCancelled) as e:
        audio_utils.to_mono16k(write(tmp_path / "src.wav", x, -x), str(tmp_path / "mono16k.mp3"))
    assert e.value.stretches and e.value.stretches[0][0] == 0
    assert summary_of(tmp_path)["check"] == "failed"


def test_ffmpeg_failure_raises_and_writes_no_summary(tmp_path):
    bad = tmp_path / "broken.wav"
    bad.write_bytes(b"RIFF....not audio")
    with pytest.raises(subprocess.CalledProcessError):
        audio_utils.to_mono16k(str(bad), str(tmp_path / "mono16k.mp3"))
    assert not (tmp_path / audio_utils.POLARITY_FILE).exists()
