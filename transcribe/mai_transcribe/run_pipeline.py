"""Orchestrator: audio -> ~10-minute chunks -> transcribe with MAI (PARALLEL) -> merge + per-word timestamps.

The chunk-cutting stage REUSES chunked_transcribe.audio_utils (single source of truth for ffmpeg/VAD/cut plan).
It differs from chunked_transcribe only in the transcription backend (MAI instead of 9router) and in MAI returning per-word timestamps.

Output in <out>:
  mono16k.mp3, plan.json
  chunks/NN.mp3, chunks/NN.json (raw MAI JSON), chunks/NN.txt
  raw_transcript.txt   merged transcript (chunks with no speech dropped)
  mai_words.json       whole-file per-word timestamps [{w,start,end,score}] with chunk offsets added -> build_srt
  manifest.json        per-chunk status (word count, cost, merged or dropped)

Why ~10-minute chunks: same as chunked_transcribe, to avoid silent omissions and stay under OpenRouter's 25 MB
request limit. See docs/mai-transcribe-notes.md.
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from transcribe.chunked_transcribe import audio_utils

from . import config, transcribe_client


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _process_chunk(c: dict, out: Path, mono: str, log_lock: threading.Lock) -> dict:
    """Cut one chunk from mono16k then transcribe with MAI (with retry). Runs in one pool thread."""
    k = c["idx"]
    mp3 = str(out / "chunks" / f"{k:02d}.mp3")
    audio_utils.cut_chunk(mono, c["start"], c["end"], mp3)
    expect_speech = audio_utils.chunk_has_speech(c)
    r = transcribe_client.transcribe_with_retry(mp3, str(out / "chunks" / f"{k:02d}.json"), expect_speech)
    res = r["res"]
    text = res.get("text", "")
    (out / "chunks" / f"{k:02d}.txt").write_text(text, encoding="utf-8")
    words = len(text.split())
    rec = {**c, "attempts": r["attempts"], "http": res.get("http"), "n_words": words,
           "duration": res.get("duration"), "cost": res.get("cost"), "included": False,
           "note": "; ".join(r["reasons"])}
    if r["reasons"]:
        rec["note"] = "error after retry: " + rec["note"]
    elif not expect_speech:
        rec["note"] = ("hallucination? (chunk has no speech but returned words)" if words > 20 else "skipped (no speech)")
    else:
        rec["included"] = True
    with log_lock:
        state = "MERGED" if rec["included"] else "SKIP: " + rec["note"]
        print(f"    chunk {k:02d} {c['start']/60:5.1f}-{c['end']/60:5.1f}m | speech {c['speech_sec']:4.0f}s | {words:5} words "
              f"| {r['attempts']} tries | ${res.get('cost')} | {state}", flush=True)
    return rec


def _global_words(out: Path, manifest: list[dict]) -> list[dict]:
    """Gather per-word timestamps of the MERGED chunks, add the chunk.start offset -> whole-file timeline.

    Format matches realign.cue_builder: [{w, start, end, score=None}]. Words missing a timestamp are dropped.
    """
    gwords: list[dict] = []
    for m in manifest:
        if not m["included"]:
            continue
        off = m["start"]
        raw = json.loads((out / "chunks" / f"{m['idx']:02d}.json").read_text(encoding="utf-8"))
        for w in raw.get("words") or []:
            s, e = w.get("start"), w.get("end")
            gwords.append({"w": w.get("word", ""),
                           "start": (s + off) if s is not None else None,
                           "end": (e + off) if e is not None else None, "score": None})
    return gwords


def run(input_path: str, out_dir: str) -> dict:
    out = Path(out_dir)
    (out / "chunks").mkdir(parents=True, exist_ok=True)

    mono = str(out / "mono16k.mp3")
    print(f"{_now()} [1/4] ffmpeg -> mono 16kHz", flush=True)
    dur = audio_utils.to_mono16k(input_path, mono)
    print(f"{_now()} [2/4] VAD + plan cuts", flush=True)
    segs = audio_utils.speech_segments(mono)
    plan = audio_utils.build_plan(mono, dur, segs)
    (out / "plan.json").write_text(json.dumps({"dur": dur, "input": input_path, "model": config.MODEL,
                                               "segments": segs, "chunks": plan}, ensure_ascii=False), encoding="utf-8")
    print(f"    audio {dur:.0f}s | {len(segs)} speech segments | {len(plan)} chunks", flush=True)

    print(f"{_now()} [3/4] transcribe {len(plan)} chunks ({config.MODEL}, {config.CONCURRENCY} parallel threads)", flush=True)
    log_lock = threading.Lock()
    results: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=config.CONCURRENCY) as ex:
        futs = {ex.submit(_process_chunk, c, out, mono, log_lock): c["idx"] for c in plan}
        for fut in as_completed(futs):
            rec = fut.result()
            results[rec["idx"]] = rec

    manifest = [results[c["idx"]] for c in plan]
    included = [(m, (out / "chunks" / f"{m['idx']:02d}.txt").read_text(encoding="utf-8").strip())
                for m in manifest if m["included"]]
    (out / "raw_transcript.txt").write_text("\n".join(t for _, t in included), encoding="utf-8")
    (out / "mai_words.json").write_text(json.dumps(_global_words(out, manifest), ensure_ascii=False), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    total_words = sum(len(t.split()) for _, t in included)
    total_cost = sum(m["cost"] for m in manifest if m.get("cost"))
    failed = [m["idx"] for m in manifest if m["note"].startswith("error after retry")]
    if failed:
        print(f"{_now()} [4/4] INCOMPLETE: {len(failed)} chunks failed after retry: {failed}. Re-run.", flush=True)
    print(f"{_now()} [4/4] done: {len(included)}/{len(plan)} chunks merged | {total_words} words | "
          f"total ${total_cost:.4f} -> raw_transcript.txt + mai_words.json", flush=True)
    return {"out_dir": str(out), "chunks": len(plan), "included": len(included), "words": total_words,
            "dur": dur, "cost": total_cost, "failed": failed}


def main():
    ap = argparse.ArgumentParser(description="Transcribe audio in ~10-minute chunks with MAI-Transcribe-2 (parallel)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default=None, help="default: config.MODEL (microsoft/mai-transcribe-2)")
    ap.add_argument("--workers", type=int, default=None, help="number of chunks sent in parallel, default config.CONCURRENCY")
    a = ap.parse_args()
    if a.model:
        config.MODEL = a.model
    if a.workers:
        config.CONCURRENCY = a.workers
    t0 = time.time()
    run(a.input, a.out_dir)
    print(f"total time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
