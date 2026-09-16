"""Stage 2 (align): forced-align the transcript onto the audio using torchaudio MMS_FA.
Emit word-level [{w,start,end,score}]. Reuses proven prototype logic (chunked
emissions to avoid 4GB GPU OOM).

Import torch/torchaudio only when calling GPU-dependent functions (load_audio, get_models, align).
Pure functions (normalize_word, assemble_words) work without torch.
"""
import argparse
import json
import re
import sys
import time
import unicodedata

import realign.config as config

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# uroman: optional, no GPU needed. If not installed, normalize_word still runs
# (only loses the ability to romanize Vietnamese via uroman; NFD-strip still works).
try:
    from uroman import Uroman
    _URO = Uroman()
except Exception:
    _URO = None


def normalize_word(w: str) -> str:
    """Romanize a (Vietnamese) word into the MMS latin token set [a-z']. Try uroman first,
    then strip NFD marks, lowercase, replace 'd-stroke' -> 'd', drop chars outside [a-z']."""
    if _URO is not None:
        try:
            w = _URO.romanize_string(w)
        except Exception:
            pass
    w = unicodedata.normalize("NFD", w)
    w = "".join(c for c in w if unicodedata.category(c) != "Mn")
    w = w.lower().replace("đ", "d").replace("’", "'")
    return re.sub(r"[^a-z']", "", w)


def read_words(txt_path: str) -> list[str]:
    """Read all tokens from the txt file (one line per cue), preserving order."""
    words: list[str] = []
    with open(txt_path, encoding="utf-8") as f:
        for ln in f:
            words += ln.split()
    return words


def load_audio(path: str, sample_rate: int, max_sec: float = 0.0):
    """Read audio via soundfile (avoids torchaudio.load needing torchcodec), resample if needed,
    take the leading max_sec seconds if max_sec > 0. Returns (Tensor[1,N], sr)."""
    import torch
    import torchaudio
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32")
    wav = torch.from_numpy(data)
    if wav.ndim == 2:
        wav = wav.mean(dim=1)
    wav = wav.unsqueeze(0)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
        sr = sample_rate
    if max_sec > 0:
        wav = wav[:, : int(max_sec * sr)]
    return wav, sr


def get_models(device):
    """Load MMS_FA model + tokenizer + aligner once, reused across many files."""
    from torchaudio.pipelines import MMS_FA as bundle
    return bundle.get_model().to(device), bundle.get_tokenizer(), bundle.get_aligner()


def assemble_words(words_raw, align_idx, token_spans, ratio, sr) -> list[dict]:
    """Map token_spans (list of torchaudio spans) back to the original positions in words_raw.

    - align_idx[k] is the index in words_raw corresponding to token_spans[k].
    - Any word not in align_idx (could not be normalized) -> start/end/score = None.
    - Time = frame_index * ratio / sr (in seconds).
    - score = mean score of the word's tokens.
    """
    times: dict[int, tuple] = {}
    for k, spans in enumerate(token_spans):
        i = align_idx[k]
        if not spans:
            continue                       # no token -> let this word fall into the None branch
        start = spans[0].start * ratio / sr
        end = spans[-1].end * ratio / sr
        score = sum(s.score for s in spans) / len(spans)
        times[i] = (start, end, score)
    out = []
    for i, w in enumerate(words_raw):
        if i in times:
            s, e, sc = times[i]
            out.append({"w": w, "start": float(s), "end": float(e), "score": float(sc)})
        else:
            out.append({"w": w, "start": None, "end": None, "score": None})
    return out


def _chunk_bounds(n_samples: int, chunk: int, min_tail: int) -> list[tuple[int, int]]:
    """Boundaries of emission chunks; merge a short tail (< min_tail) into the last chunk so it does NOT
    create a tiny chunk (a too-short chunk makes wav2vec2's conv raise a 'kernel size' error)."""
    bounds: list[tuple[int, int]] = []
    i = 0
    while i < n_samples:
        end = i + chunk
        if n_samples - end < min_tail:
            end = n_samples
        bounds.append((i, end))
        i = end
    return bounds


def _emission(model, wav, dev, emit_chunk_sec: float, sr: int):
    """Chunked emission (bounded GPU mem) -> tensor [1, frames, vocab] on CPU.
    Merge the short tail via _chunk_bounds to avoid a tiny chunk that breaks conv."""
    import torch
    chunk = int(emit_chunk_sec * sr)
    min_tail = int(0.5 * sr)   # last chunk not shorter than 0.5s (avoid conv 'kernel size' error)
    ems = []
    with torch.inference_mode():
        for start, end in _chunk_bounds(wav.size(1), chunk, min_tail):
            emi, _ = model(wav[:, start:end].to(dev))
            ems.append(emi.cpu())
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    return torch.cat(ems, dim=1)


def align(
    audio_path: str,
    words_raw: list[str],
    device: str = "cpu",
    emit_chunk_sec: float = config.EMIT_CHUNK_SEC,
    max_sec: float = 0.0,
    models: tuple | None = None,
) -> list[dict]:
    """Forced-align words_raw onto audio_path using MMS_FA.

    Run chunked emission (emit_chunk_sec) to avoid 4GB GPU OOM.
    models=(model,tokenizer,aligner) can be passed in to reuse across many files.
    Returns list[Word] = [{w, start, end, score}] (start=None if it could not be aligned).
    """
    import torch
    dev = torch.device(device)
    wav, sr = load_audio(audio_path, config.SAMPLE_RATE, max_sec)
    norm = [normalize_word(w) for w in words_raw]
    align_idx = [i for i, nw in enumerate(norm) if nw]
    align_words_list = [norm[i] for i in align_idx]
    model, tokenizer, aligner = models if models is not None else get_models(dev)
    emission = _emission(model, wav, dev, emit_chunk_sec, sr)
    with torch.inference_mode():
        token_spans = aligner(emission[0].to(dev), tokenizer(align_words_list))
    ratio = wav.size(1) / emission.size(1)
    return assemble_words(words_raw, align_idx, token_spans, ratio, sr)


def main():
    import os
    import torch
    ap = argparse.ArgumentParser(description="MMS forced-align -> word-level json")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--txt", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--key", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--emit-chunk-sec", type=float, default=config.EMIT_CHUNK_SEC)
    ap.add_argument("--max-sec", type=float, default=0.0)
    a = ap.parse_args()
    words_raw = read_words(a.txt)
    out_dir = os.path.dirname(a.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    words = align(a.audio, words_raw, a.device, a.emit_chunk_sec, a.max_sec)
    with open(a.out_json, "w", encoding="utf-8") as f:
        json.dump({"key": a.key or a.audio, "audio": a.audio, "words": words}, f, ensure_ascii=False)
    n_al = sum(1 for w in words if w["start"] is not None)
    print(f"aligned {n_al}/{len(words)} words in {time.time() - t0:.1f}s -> {a.out_json}")


if __name__ == "__main__":
    main()
