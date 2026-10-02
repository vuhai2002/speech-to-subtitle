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
                    "-itsoffset", "3", "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "mpeg4",
                    "-c:a", "pcm_s16le", str(late)], check=True)
    dst = tmp_path / "mono16k.mp3"
    audio_utils.to_mono16k(str(late), str(dst))
    (start, end), = summary_of(tmp_path)["sections"]
    assert start == 0.0 and 6.0 <= end <= 7.0     # seconds counted from the first audio sample
    assert all(v > -40 for v in mono_levels(dst)[:6])   # a timestamp-timed flip would leave seconds 3-5 cancelled


def test_mono_source_is_skipped(tmp_path):
    audio_utils.to_mono16k(write(tmp_path / "src.wav", speech_like(5)), str(tmp_path / "mono16k.mp3"))
    s = summary_of(tmp_path)
    assert s["channels"] == 1 and s["check"] == "skipped: mono source"


def test_mono_first_stream_is_the_one_converted(tmp_path):
    y = speech_like(5, seed=2)
    mono, stereo = write(tmp_path / "mono.wav", speech_like(5)), write(tmp_path / "stereo.wav", y, -y)
    two = tmp_path / "two.mka"     # a:0 is mono; a:1 is stereo, flagged default (ffmpeg's own pick) and it cancels
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", mono, "-i", stereo, "-map", "0:a", "-map", "1:a",
                    "-c:a", "pcm_s16le", "-disposition:a:0", "0", "-disposition:a:1", "default", str(two)],
                   check=True)
    dst = tmp_path / "mono16k.mp3"
    audio_utils.to_mono16k(str(two), str(dst))
    s = summary_of(tmp_path)
    assert s["channels"] == 1 and s["check"] == "skipped: mono source"
    assert all(v > -40 for v in mono_levels(dst)[:4])


def test_unreadable_channel_count_is_still_analysed(tmp_path, monkeypatch):
    x = speech_like(8)
    monkeypatch.setattr(audio_utils, "_channels", lambda src: 0)     # ffprobe printed nothing (an old or failing probe)
    dst = tmp_path / "mono16k.mp3"
    audio_utils.to_mono16k(write(tmp_path / "src.wav", x, -x), str(dst))
    s = summary_of(tmp_path)
    assert (s["channels"], s["sections"], s["check"]) == (0, [[0.0, None]], "passed")   # the probed value is recorded
    levels = mono_levels(dst)
    assert all(v > -40 for v in levels), levels


def mono_gain_db(folder, peak: float) -> float:
    """Level of the mono mp3 that to_mono16k writes, in dB against one source channel, for an s16 file holding the same
    300 Hz sine of this peak in both channels. Decoded as float so a mix above full scale is not clipped here."""
    folder.mkdir()
    x = peak * np.sin(2 * np.pi * 300 * np.arange(3 * SR) / SR)
    dst = folder / "mono16k.mp3"
    audio_utils.to_mono16k(write(folder / "src.wav", x, x), str(dst))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(dst), "-ar", "16000", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    middle = np.frombuffer(raw, dtype="<f4").astype(float)[16000:32000]      # the middle second
    return 10 * np.log10(np.mean(middle ** 2)) - 20 * np.log10(peak / np.sqrt(2))


def test_integer_source_is_mixed_without_clipping(tmp_path):
    quiet = mono_gain_db(tmp_path / "quiet", 10 ** (-20 / 20))     # far from saturation: the mix path's own gain
    loud = mono_gain_db(tmp_path / "loud", 10 ** (-1 / 20))        # the two channels add up to +2 dBFS
    assert quiet > 2.0     # the +3.01 dB of the mix minus the mp3 codec's own constant loss of about 0.4 dB
    assert loud > quiet - 0.5, f"loud {loud:.2f} dB vs quiet {quiet:.2f} dB"      # saturation costs about 1 dB


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


def test_flip_that_misses_fails_the_check(tmp_path, monkeypatch):
    x = speech_like(8)
    monkeypatch.setattr(polarity, "flip_filter", lambda sections: "anull")   # the flip runs but changes nothing
    with pytest.raises(polarity.AudioCancelled):
        audio_utils.to_mono16k(write(tmp_path / "src.wav", x, -x), str(tmp_path / "mono16k.mp3"))
    assert summary_of(tmp_path)["check"] == "failed"


def test_ffmpeg_failure_raises_and_writes_no_summary(tmp_path):
    bad = tmp_path / "broken.wav"
    bad.write_bytes(b"RIFF....not audio")
    with pytest.raises(subprocess.CalledProcessError):
        audio_utils.to_mono16k(str(bad), str(tmp_path / "mono16k.mp3"))
    assert not (tmp_path / audio_utils.POLARITY_FILE).exists()


def test_stereo_ffmpeg_failure_shows_ffmpeg_reason(tmp_path, capfd):
    x = speech_like(4)
    wav = tmp_path / "src.wav"
    write(wav, x, -x)
    data = bytearray(wav.read_bytes())
    data[20:22] = (0x1234).to_bytes(2, "little")   # unknown wFormatTag: ffprobe sees 2 channels, ffmpeg has no decoder
    wav.write_bytes(data)
    with pytest.raises(subprocess.CalledProcessError) as e:
        audio_utils.to_mono16k(str(wav), str(tmp_path / "mono16k.mp3"))
    assert capfd.readouterr().err.strip() != ""    # ffmpeg's own reason reached the process's stderr
    assert e.value.cmd[0] == "ffmpeg" and "-i" in e.value.cmd
    assert not (tmp_path / audio_utils.POLARITY_FILE).exists()


def test_empty_stereo_source_writes_no_summary(tmp_path):
    sf.write(str(tmp_path / "src.wav"), np.zeros((0, 2)), SR, subtype="PCM_16")
    with pytest.raises(ValueError):                # an empty file has no duration to read
        audio_utils.to_mono16k(str(tmp_path / "src.wav"), str(tmp_path / "mono16k.mp3"))
    assert not (tmp_path / audio_utils.POLARITY_FILE).exists()
