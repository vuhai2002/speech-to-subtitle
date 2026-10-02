"""Pure-code audio utilities: decode to mono 16kHz, run VAD, plan chunk cuts at silences."""
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from . import config, polarity

POLARITY_FILE = "polarity.json"
# Explicit mono gain: what `-ac 1` gives a float-decoded source (AAC/MP3) today, now independent of the sample format.
MONO_MIX = "pan=mono|c0=0.7071*c0+0.7071*c1"
# Both graphs start from planar float. pan mixes in the stream's own sample format, so an integer source would
# saturate in 0.7071*(L+R) while the flip filter (aeval, always float) would not: one file, two different mixes.
STEREO_FLOAT = "aformat=sample_fmts=fltp:channel_layouts=stereo"


def _channels(src: str) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=channels",
                          "-of", "csv=p=0", src], capture_output=True, text=True).stdout.strip()
    return int(out.split(",")[0]) if out else 0


def _duration(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    return float(out.strip())


def _run_ffmpeg_with_pcm(args: list[str]) -> polarity.Stats:
    """Runs ffmpeg whose `pipe:1` output is s16le stereo at polarity.ANALYSIS_RATE; returns its statistics.

    ffmpeg's stderr is inherited, not piped (so it cannot block): its error text reaches the job log like every other
    ffmpeg call in this module, and CalledProcessError carries the full command."""
    acc = polarity.StatsAccumulator()
    cmd = ["ffmpeg", "-y", "-v", "error", "-nostdin", *args]
    # Popen as a context manager closes the pipe on every path, so ffmpeg never stays blocked on a pipe nobody reads.
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as p:
        while chunk := p.stdout.read(65536):
            acc.feed(chunk)
        if p.wait() != 0:
            raise subprocess.CalledProcessError(p.returncode, cmd)
    return acc.finish()


def _pcm_out() -> list[str]:
    return ["-f", "s16le", "-acodec", "pcm_s16le", "pipe:1"]


def _save_polarity(dst: str, record: dict) -> None:
    (Path(dst).parent / POLARITY_FILE).write_text(json.dumps(record), encoding="utf-8")


def to_mono16k(src: str, dst: str) -> float:
    """ffmpeg: any audio -> mp3 mono 16kHz, polarity-aware. Returns the duration (seconds).

    A 2-channel source is measured in the same pass that writes the mono file; sections where its channels are
    polarity-inverted (polarity.find_sections) are flipped in a second pass. Writes polarity.json next to dst.
    Raises polarity.AudioCancelled when the result still cancels the voice (the caller exits with
    exit_codes.AUDIO_CANCELLED); ffmpeg failures raise CalledProcessError and write no summary."""
    rate = str(config.SAMPLE_RATE)
    (Path(dst).parent / POLARITY_FILE).unlink(missing_ok=True)
    channels = _channels(src)
    if channels < 2:
        # -map 0:a:0 is the stream _channels probed; without it ffmpeg may pick another audio stream
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-i", src, "-map", "0:a:0", "-ac", "1",
                        "-ar", rate, dst], check=True)
        dur = _duration(dst)           # read before the summary: a file without a duration must leave no record
        _save_polarity(dst, polarity.summary(channels, [], "skipped: mono source"))
        return dur
    stats = _run_ffmpeg_with_pcm([
        "-i", src, "-filter_complex",
        f"[0:a:0]{STEREO_FLOAT},asplit=2[m0][a];[m0]{MONO_MIX}[m];"
        f"[a]aresample={polarity.ANALYSIS_RATE}[an]",
        "-map", "[m]", "-ar", rate, dst, "-map", "[an]", *_pcm_out()])
    sections = polarity.find_sections(stats)
    louder = np.maximum(polarity.db(stats.left), polarity.db(stats.right))
    mix = polarity.db(stats.mix)
    if sections:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-i", src, "-filter_complex",
                        f"[0:a:0]{STEREO_FLOAT},{polarity.flip_filter(sections)},{MONO_MIX}[m]",
                        "-map", "[m]", "-ar", rate, dst], check=True)
        mix = polarity.db(_run_ffmpeg_with_pcm(
            ["-i", dst, "-ac", "2", "-ar", str(polarity.ANALYSIS_RATE), *_pcm_out()]).left)
    n = min(len(mix), len(louder))
    stretches = polarity.cancelled_stretches(mix[:n], louder[:n])
    if stretches:
        _save_polarity(dst, polarity.summary(channels, sections, "failed", stretches))
        raise polarity.AudioCancelled(stretches)
    dur = _duration(dst)               # read before the summary: a file without a duration must leave no "passed"
    _save_polarity(dst, polarity.summary(channels, sections, "passed"))
    return dur


def cut_chunk(src_mono16k: str, start: float, end: float, dst: str) -> None:
    """Cut [start, end) from the mono16k file into a mono16k mp3 (re-encode, cheap)."""
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
                    "-i", src_mono16k, "-ac", "1", "-ar", str(config.SAMPLE_RATE), dst], check=True)


def speech_segments(mono16k_path: str) -> list[list[float]]:
    """silero-vad -> list of [start, end] (seconds) of speech segments."""
    from silero_vad import get_speech_timestamps, load_silero_vad
    wav, sr = sf.read(mono16k_path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    import torch
    ts = get_speech_timestamps(torch.from_numpy(wav), load_silero_vad(), sampling_rate=sr)
    return [[t["start"] / sr, t["end"] / sr] for t in ts]


def speech_in(segs: list[list[float]], a: float, b: float) -> float:
    """Total seconds of speech within [a, b]."""
    return sum(max(0.0, min(e, b) - max(s, a)) for s, e in segs)


def chunk_has_speech(chunk: dict) -> bool:
    """True if a chunk holds real speech (its text is merged as is); False if VAD hears (near-)silence there,
    where a model may hallucinate: the MAI path drops such a chunk's text, the Router path keeps only the
    sentences MMS aligns well (build_srt + silent_scoring).

    A chunk qualifies when it has at least MIN_SPEECH_CHUNK_SEC of detected speech, OR when
    speech fills at least MIN_SPEECH_RATIO of its own length. The ratio keeps short clips: a
    42s chunk with 28s of speech is 67% speech - clearly real - even though 28s is under the
    absolute floor that is meant for near-silent 10-minute chunks. Shared by both transcribe
    paths (chunked_transcribe and mai_transcribe) so they agree on what counts as speech."""
    speech = chunk.get("speech_sec", 0.0)
    length = max(1e-9, chunk["end"] - chunk["start"])
    return speech >= config.MIN_SPEECH_CHUNK_SEC or (speech / length) >= config.MIN_SPEECH_RATIO


def build_plan(mono16k_path: str, dur: float, segs: list[list[float]]) -> list[dict]:
    """Split into ~CHUNK_TARGET_SEC chunks, cutting at the LONGEST SILENCE near the target mark.

    Each chunk: {idx, start, end, speech_sec, cut_gap_sec}. speech_sec decides whether VAD calls the chunk silent
    (chunk_has_speech).
    """
    gaps = [(segs[i][1], segs[i + 1][0]) for i in range(len(segs) - 1)]
    cuts = [0.0]
    gap_len = [None]
    target = config.CHUNK_TARGET_SEC
    while target < dur - config.MIN_CHUNK_SEC / 2:
        cand = [(e - s, (s + e) / 2) for s, e in gaps
                if abs((s + e) / 2 - target) <= config.CUT_SEARCH_SEC and (s + e) / 2 > cuts[-1] + config.MIN_CHUNK_SEC]
        g, mid = max(cand) if cand else (0.0, target)
        cuts.append(round(mid, 2))
        gap_len.append(round(g, 2))
        target = mid + config.CHUNK_TARGET_SEC
    cuts.append(round(dur, 3))
    plan = []
    for i in range(len(cuts) - 1):
        s, e = cuts[i], cuts[i + 1]
        plan.append({"idx": i + 1, "start": s, "end": e, "speech_sec": round(speech_in(segs, s, e), 1),
                     "cut_gap_sec": gap_len[i + 1] if i + 1 < len(gap_len) else None})
    return plan


def loudness_dbfs(mono16k_path: str, window_sec: float = 30.0) -> list[float]:
    """Loudness per window (dBFS) - to distinguish 'silence' from 'loud music that VAD misses'."""
    wav, sr = sf.read(mono16k_path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    n = int(window_sec * sr)
    return [round(float(20 * np.log10(np.sqrt((wav[i:i + n] ** 2).mean()) + 1e-9)), 1)
            for i in range(0, len(wav), n)]
