"""Forced-align for files TOO LONG that OOM at the DP step (trellis = frames x tokens).

Split the audio into K windows of ~target_window_sec, CUTTING at silence (blank-run in
the emission) so as not to cut mid-sentence, forced-align each window then join (time offset).
The emission is computed once (cheap, ~tens of MB); only forced-align uses RAM -> windowing
bounds each window's trellis ~ (frames/K)x(tokens/K) = total/K^2 -> safe.

window_align() returns a list of Word in the same format as align_words.align(), so it plugs straight into
run_batch (VAD + words.json + segment unchanged).
"""
from . import config
from .align_words import _emission, get_models, load_audio, normalize_word


def _blank_runs(emission, min_run: int) -> list[tuple[int, int]]:
    """Runs of consecutive frames with token = blank(0) (silence) of length >= min_run frames."""
    blank = (emission[0].argmax(dim=-1) == 0).tolist()
    runs: list[tuple[int, int]] = []
    i, n = 0, len(blank)
    while i < n:
        if blank[i]:
            j = i
            while j < n and blank[j]:
                j += 1
            if j - i >= min_run:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _window_bounds(emission, n_align: int, dur_sec: float, target_window_sec: float,
                   ratio: float, sr: int) -> tuple[list[int], list[int]]:
    """Split into K windows (~target_window_sec). Cut point = MIDDLE of the blank-run nearest the
    evenly-divided point (cut where speech pauses). Returns (frame_bounds, word_bounds); words split proportionally to frames."""
    total_frames = emission.size(1)
    k = max(1, round(dur_sec / target_window_sec)) if target_window_sec > 0 else 1
    if k <= 1:
        return [0, total_frames], [0, n_align]
    min_run = max(1, int(0.3 * sr / ratio))          # blank-run >= 0.3s to count as a cut point
    centers = [(s + e) // 2 for s, e in _blank_runs(emission, min_run)]
    fsplits = []
    for i in range(1, k):
        t = round(total_frames * i / k)
        fsplits.append(min(centers, key=lambda c: abs(c - t)) if centers else t)
    fsplits = sorted({f for f in fsplits if 0 < f < total_frames})
    fbounds = [0] + fsplits + [total_frames]
    wbounds = [0] + [round(n_align * f / total_frames) for f in fsplits] + [n_align]
    return fbounds, wbounds


def window_align(audio_path: str, words_raw: list[str], device: str = "cpu",
                 emit_chunk_sec: float = config.EMIT_CHUNK_SEC,
                 target_window_sec: float = 1500.0, models: tuple | None = None) -> list[dict]:
    """Like align() but forced-align per time window then join. For long files that OOM."""
    import torch
    dev = torch.device(device)
    wav, sr = load_audio(audio_path, config.SAMPLE_RATE, 0.0)
    norm = [normalize_word(w) for w in words_raw]
    align_idx = [i for i, nw in enumerate(norm) if nw]
    align_words_list = [norm[i] for i in align_idx]
    model, tokenizer, aligner = models if models is not None else get_models(dev)
    emission = _emission(model, wav, dev, emit_chunk_sec, sr)
    ratio = wav.size(1) / emission.size(1)
    fbounds, wbounds = _window_bounds(emission, len(align_words_list), wav.size(1) / sr,
                                      target_window_sec, ratio, sr)

    times: dict[int, tuple] = {}
    for wi in range(len(fbounds) - 1):
        f0, f1 = fbounds[wi], fbounds[wi + 1]
        a0, a1 = wbounds[wi], wbounds[wi + 1]
        sub = align_words_list[a0:a1]
        if not sub:
            continue
        with torch.inference_mode():
            spans = aligner(emission[0, f0:f1].to(dev), tokenizer(sub))
        offset = f0 * ratio / sr
        for k, sp in enumerate(spans):
            if not sp:
                continue
            times[align_idx[a0 + k]] = (
                sp[0].start * ratio / sr + offset,
                sp[-1].end * ratio / sr + offset,
                sum(s.score for s in sp) / len(sp),
            )
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    out = []
    for i, w in enumerate(words_raw):
        if i in times:
            s, e, sc = times[i]
            out.append({"w": w, "start": float(s), "end": float(e), "score": float(sc)})
        else:
            out.append({"w": w, "start": None, "end": None, "score": None})
    return out
