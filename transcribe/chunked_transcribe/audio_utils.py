"""Pure-code audio utilities: decode to mono 16kHz, run VAD, plan chunk cuts at silences."""
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from . import config


def to_mono16k(src: str, dst: str) -> float:
    """ffmpeg: any audio -> mp3 mono 16kHz. Returns the duration (seconds)."""
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ac", "1", "-ar",
                    str(config.SAMPLE_RATE), dst], check=True)
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", dst], capture_output=True, text=True).stdout
    return float(out.strip())


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


def build_plan(mono16k_path: str, dur: float, segs: list[list[float]]) -> list[dict]:
    """Split into ~CHUNK_TARGET_SEC chunks, cutting at the LONGEST SILENCE near the target mark.

    Each chunk: {idx, start, end, speech_sec, cut_gap_sec}. speech_sec is used to drop chunks with no speech.
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
