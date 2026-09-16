"""silero-vad: find the speech region [first_speech, last_speech] to cut intro/outro music/chanting.

Functions:
- region_from_timestamps: pure function, unit-testable, no model needed.
- load_vad_model: load silero-vad once, reused across many files.
- speech_region: decode audio via load_audio (soundfile), run VAD, return (first, last) or None.
"""
import realign.config as config


def region_from_timestamps(ts: list[dict], sr: int) -> tuple[float, float] | None:
    """Convert a list of VAD timestamps [{start, end}] (in samples) -> (first_sec, last_sec).

    Return None if the list is empty (no speech detected).
    first_sec = start of the first segment / sr.
    last_sec  = end   of the last segment / sr.
    """
    if not ts:
        return None
    return (ts[0]["start"] / sr, ts[-1]["end"] / sr)


def load_vad_model():
    """Load the silero-vad model once to reuse across many files (avoid re-init overhead).

    Return a model ready for get_speech_timestamps.
    """
    from silero_vad import load_silero_vad
    return load_silero_vad()


def speech_region(
    audio_path: str,
    sr: int = config.SAMPLE_RATE,
    model=None,
) -> tuple[float, float] | None:
    """Find the speech region in an audio file with silero-vad.

    - Decode audio via realign.align_words.load_audio (soundfile, avoids torchcodec).
    - Run silero get_speech_timestamps on the 1D waveform.
    - Return (first_speech_sec, last_speech_sec) or None if completely silent.

    model: pass in to reuse; if None, load once internally.
    """
    from silero_vad import get_speech_timestamps
    from realign.align_words import load_audio

    model = model or load_vad_model()
    wav, _ = load_audio(audio_path, sr)
    ts = get_speech_timestamps(wav.squeeze(0), model, sampling_rate=sr)
    return region_from_timestamps(ts, sr)
