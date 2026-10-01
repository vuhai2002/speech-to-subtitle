"""Bridge chunked_transcribe -> realign: build .srt from the chunked transcript.

Align EACH CHUNK with MMS (realign.align_words.align), then add the chunk's time offset,
gather all words onto a single timeline, then pass them through realign's cue-building stage
(cue_builder -> cue_clamp -> srt_writer). Do NOT use window_align on the whole file (it mistimes when
the file ends with a long stretch of no speech).

A chunk VAD hears as silent (manifest "silent": true) contributes only the sentences MMS aligns with a mean
word score >= config.SILENT_MIN_SCORE (see silent_scoring). The alignment stats and every such sentence's
score go into run_trace.json (run_trace.add_build), also when no cue can be built (exit 4). A merged teaching
chunk with no word MMS can align (empty text, or only symbols/digits) also stops the lecture with exit 4.

Run: python -m transcribe.chunked_transcribe.build_srt --out-dir <out_ct> [--device cuda]
Output: <out_ct>/<original file name>.srt
"""
import argparse
import json
import sys
import time
from pathlib import Path

from . import config, run_trace, silent_scoring
from transcribe import exit_codes

sys.path.insert(0, str(config.PROJECT_ROOT))  # so realign in the repo can be imported


def _shift(words: list[dict], off: float) -> None:
    for w in words:
        if w["start"] is not None:
            w["start"] += off
            w["end"] += off


def _stats(words: list[dict]) -> dict:
    timed = [w for w in words if w["start"] is not None]
    scores = [w["score"] for w in timed if w.get("score") is not None]
    return {"total_words": len(words), "aligned_words": len(timed),
            "mean_score": round(sum(scores) / len(scores), 3) if scores else None}


def _widen(region: tuple | None, span: tuple | None) -> tuple | None:
    """The VAD speech region stretched to also cover kept silent-chunk words, so clamp_cues never squashes a kept
    sentence that lies before the first or after the last VAD speech."""
    if span is None:
        return region
    if region is None:
        return span
    return (min(region[0], span[0]), max(region[1], span[1]))


NO_ALIGNABLE_WORD = "no alignable word"


def _unaligned(words: list[str]) -> list[dict]:
    return [{"w": w, "start": None, "end": None, "score": None} for w in words]


def _alignable(words: list[str]) -> bool:
    """True when at least one word keeps a letter MMS can align. normalize_word keeps only [a-z'], so empty text
    and text of only "...", "♪" or digits has none - and MMS crashes on an empty token list."""
    from realign.align_words import normalize_word
    return any(normalize_word(w) for w in words)


class _UnalignableTeachingChunk(exit_codes.NoAlignedWords):
    """A merged teaching chunk has no word MMS can align. Exit 4 like any lecture with nothing to write; carries the
    per-chunk results so far, so run_trace.json still records which chunk stopped the build."""

    def __init__(self, message: str, results: dict[int, dict]):
        super().__init__(message)
        self.results = results


def _align_silent(align, mp3: str, flat: list[str], device: str, models) -> tuple[list[dict], str | None]:
    """MMS-align the words of a chunk VAD hears as silent without ever failing the lecture.

    Text MMS cannot take - no alignable token at all (only "...", "♪", digits) or more letters than the audio has
    frames - comes back unaligned (every sentence then scores None and is dropped) with the reason, recorded in
    run_trace.json. A teaching chunk is never dropped like this: with no alignable word the lecture stops with
    exit 4 (_aligned_words), and any other alignment failure is raised.
    """
    unaligned = _unaligned(flat)
    if not _alignable(flat):
        return unaligned, NO_ALIGNABLE_WORD
    try:
        return align(mp3, flat, device, models=models), None
    except RuntimeError as e:
        return unaligned, "align failed: " + " ".join(str(e).split())[:200]


def _pin_unaligned_edges(words: list[dict]) -> None:
    """Give a kept sentence's leading and trailing unaligned words (a number, a dash, a symbol: no MMS token) the
    time of the sentence's own first / last aligned word. Otherwise cue building gives a leading one the time of
    the previous timed word on the whole timeline - for a silent chunk, possibly a teaching word minutes earlier."""
    timed = [i for i, w in enumerate(words) if w["start"] is not None]
    if not timed:
        return
    first, last = timed[0], timed[-1]
    for w in words[:first]:
        w["start"] = w["end"] = words[first]["start"]
    for w in words[last + 1:]:
        w["start"] = w["end"] = words[last]["end"]


def _aligned_words(out_dir: str, device: str, models: tuple | None = None):
    """Align each merged chunk and add its offset -> (global_words, speech_region, original_file_name, results).

    results[idx] holds the chunk's alignment stats and, for a silent chunk, its scored sentences (times from the
    start of the file) for run_trace.json. MMS is loaded once (or `models` is reused) for every chunk.
    """
    from realign.align_words import align, get_models
    out = Path(out_dir)
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    included = [m for m in manifest if m["included"]]
    models = models if models is not None else get_models(device)
    global_words: list[dict] = []
    results: dict[int, dict] = {}
    kept_span = None
    for m in included:
        k, off = m["idx"], m["start"]
        mp3 = str(out / "chunks" / f"{k:02d}.mp3")
        text = (out / "chunks" / f"{k:02d}.txt").read_text(encoding="utf-8")
        t0 = time.time()
        if m.get("silent"):
            sentences = silent_scoring.split_sentences(text)
            words, align_error = _align_silent(align, mp3, [w for s in sentences for w in s], device, models)
            _shift(words, off)
            scored = silent_scoring.score_sentences(sentences, words, config.SILENT_MIN_SCORE)
            kept = []
            for s in scored:
                if s["kept"]:
                    ws = [{**w, "silent": True} for w in s["words"]]   # copies: the stats below count real alignment
                    _pin_unaligned_edges(ws)
                    kept += ws
            global_words += kept
            timed = [w for w in kept if w["start"] is not None]
            if timed:
                lo, hi = timed[0]["start"], timed[-1]["end"]
                kept_span = (lo, hi) if kept_span is None else (min(kept_span[0], lo), max(kept_span[1], hi))
            results[k] = {**_stats(words), "sentences": [
                {key: s[key] for key in ("text", "start", "end", "score", "kept")} for s in scored]}
            if align_error:
                results[k]["align_error"] = align_error
            n_kept = sum(1 for s in scored if s["kept"])
            print(f"    chunk {k:02d} align {results[k]['aligned_words']}/{len(words)} words ({time.time() - t0:.0f}s)"
                  f" | silent: kept {n_kept}/{len(scored)} sentences (score >= {config.SILENT_MIN_SCORE:g})",
                  flush=True)
        else:
            raw = text.split()
            if not _alignable(raw):
                # Empty text (a whole file under a minute: under 30 s of speech skips the text check) or only
                # symbols/digits. MMS would crash on it, and a teaching chunk never drops out quietly: stop the
                # lecture with exit 4 and let the trace name the chunk.
                results[k] = {**_stats(_unaligned(raw)), "align_error": NO_ALIGNABLE_WORD}
                raise _UnalignableTeachingChunk(
                    f"chunk {k:02d} ({off / 60:.1f}-{m['end'] / 60:.1f} min) has speech but no word MMS can align"
                    f" ({len(raw)} word{'' if len(raw) == 1 else 's'} in its text)", results)
            words = align(mp3, raw, device, models=models)
            _shift(words, off)
            global_words += words
            results[k] = _stats(words)
            print(f"    chunk {k:02d} align {results[k]['aligned_words']}/{len(words)} words ({time.time() - t0:.0f}s)",
                  flush=True)
    segs = plan.get("segments") or []
    region = _widen((segs[0][0], segs[-1][1]) if segs else None, kept_span)
    name = Path(plan["input"]).stem
    return global_words, region, name, results


def _save_trace(out: Path, trace: dict | None, results: dict, summary: dict) -> None:
    if trace is None:   # a run dir from an older run_pipeline has no trace: nothing to complete
        return
    # The threshold THIS build judged the silent sentences with: a rebuild may use another SILENT_MIN_SCORE than
    # run_pipeline recorded, and the trace must not claim the old one.
    trace.setdefault("meta", {}).setdefault("thresholds", {})["silent_min_score"] = config.SILENT_MIN_SCORE
    run_trace.save(out, run_trace.add_build(trace, results, summary))


def build(out_dir: str, device: str = "cuda", models: tuple | None = None) -> str:
    """Build the .srt. `models` = preloaded realign MMS models, reused when many builds run in one process."""
    import realign.config as rcfg
    from realign.cue_builder import build_cues
    from realign.cue_clamp import clamp_cues
    from realign.srt_writer import cues_to_srt
    out = Path(out_dir)
    print(f"{time.strftime('%H:%M:%S')} align each chunk (MMS, {device})", flush=True)
    trace = run_trace.read(out)
    results: dict[int, dict] = {}
    try:
        words, region, name, results = _aligned_words(out_dir, device, models)
        if not any(w.get("start") is not None for w in words):
            raise exit_codes.NoAlignedWords(f"no aligned word out of {len(words)} (nothing to write)")
        print(f"{time.strftime('%H:%M:%S')} build cues (realign) from {len(words)} words | vad {region}", flush=True)
        # A 1-2 word line at either end of the lecture is merged into its neighbour as a likely misalignment -
        # unless it is a kept silent sentence: that is real text at its own time, possibly minutes away. And a short
        # line is never merged back into a cue that ended more than ISOLATION_GAP earlier (kept silent sentences
        # are sparse, so the previous cue can be a teaching line minutes before).
        cues = build_cues(words, rcfg, merge_head=not words[0].get("silent"), merge_tail=not words[-1].get("silent"),
                          max_back_gap=rcfg.ISOLATION_GAP)
        cues = clamp_cues(cues, region, rcfg)
        if not cues:
            raise exit_codes.NoAlignedWords(f"{len(words)} words built no cue (nothing to write)")
    except exit_codes.NoAlignedWords as e:
        # An unalignable teaching chunk stops the alignment loop itself: its results so far ride on the exception.
        _save_trace(out, trace, getattr(e, "results", results), {"cues": 0, "last_cue_end": None, "mean_score": None,
                                                                 "exit": exit_codes.NO_ALIGNED_WORDS})
        raise
    srt_path = out / f"{name}.srt"
    srt_path.write_text(cues_to_srt(cues), encoding="utf-8")
    scores = [w["score"] for w in words if w.get("score") is not None]
    mean_val = round(sum(scores) / len(scores), 3) if scores else None
    last = cues[-1]["end"]
    _save_trace(out, trace, results, {"cues": len(cues), "last_cue_end": round(last, 3), "mean_score": mean_val,
                                      "exit": exit_codes.OK})
    mean = f"{mean_val:.3f}" if mean_val is not None else "n/a"
    print(f"{time.strftime('%H:%M:%S')} done: {len(cues)} cues | last cue ends {int(last)//60:02d}:{int(last)%60:02d}"
          f" | mean MMS score {mean} | -> {srt_path}", flush=True)
    return str(srt_path)


def main():
    ap = argparse.ArgumentParser(description="Build .srt from the chunked transcript (align each chunk + realign cue building)")
    ap.add_argument("--out-dir", required=True, help="run_pipeline output directory")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    try:
        build(a.out_dir, a.device)
    except exit_codes.NoAlignedWords as e:
        print(f"NO_ALIGNED_WORDS: {e}", file=sys.stderr, flush=True)
        sys.exit(exit_codes.NO_ALIGNED_WORDS)


if __name__ == "__main__":
    main()
