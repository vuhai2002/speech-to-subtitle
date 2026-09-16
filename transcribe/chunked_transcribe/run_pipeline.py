"""Pure-code orchestrator: audio -> ~10-minute chunks -> transcribe (PARALLEL) -> merge -> raw output.

Output (no AI):
  <out>/mono16k.mp3, <out>/plan.json
  <out>/chunks/NN.mp3, <out>/chunks/NN.txt
  <out>/raw_transcript.txt   merged transcript (chunks with no speech dropped)
  <out>/manifest.json        per-chunk status + guard

Processes config.CONCURRENCY chunks in parallel (default 3). Each chunk retries up to MAX_ATTEMPTS on failure.
On 403 (account locked): soft stop - mark the chunk as not transcribed, report the file as incomplete for a re-run.
Chunks are independent, so they are merged back in order once done.
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from . import audio_utils, config, transcribe_client


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _process_chunk(c: dict, out: Path, mono: str, prompt: str, log_lock: threading.Lock) -> dict:
    """Cut one chunk from mono16k then transcribe (with retry). Runs in one pool thread."""
    k = c["idx"]
    mp3 = str(out / "chunks" / f"{k:02d}.mp3")
    audio_utils.cut_chunk(mono, c["start"], c["end"], mp3)
    r = transcribe_client.transcribe_with_retry(
        mp3, prompt, c["speech_sec"], str(out / "chunks" / f"{k:02d}_req.json"), str(out / "chunks" / f"{k:02d}_raw.sse"))
    res = r["res"]
    text = res.get("text", "")
    (out / "chunks" / f"{k:02d}.txt").write_text(text, encoding="utf-8")
    words = len(text.split())
    rec = {**c, "attempts": r["attempts"], "http": res.get("http"), "served_model": res.get("served_model"),
           "finish": res.get("finish"), "words": words, "usage": res.get("usage"),
           "included": False, "locked": r["locked"], "note": "; ".join(r["reasons"])}
    if r["locked"]:
        rec["note"] = "403 account locked - not transcribed"
    elif c["speech_sec"] < config.MIN_SPEECH_CHUNK_SEC:
        rec["note"] = ("fabricated (chunk has no speech but returned words)" if words > 20 else "skipped (no speech)")
    else:
        rec["included"] = True
    density = words / (c["speech_sec"] / 60) if c["speech_sec"] > 0 else 0
    with log_lock:
        state = "MERGED" if rec["included"] else ("LOCKED-403" if r["locked"] else "SKIP: " + rec["note"])
        print(f"    chunk {k:02d} {c['start']/60:5.1f}-{c['end']/60:5.1f}m | speech {c['speech_sec']:4.0f}s | {words:5} words "
              f"({density:3.0f}/min) | {r['attempts']} tries | {res.get('finish')} | {state}", flush=True)
    return rec


def run(input_path: str, out_dir: str, prompt_file: str | None = None) -> dict:
    out = Path(out_dir)
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    prompt = config.load_prompt(prompt_file)

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
        futs = {ex.submit(_process_chunk, c, out, mono, prompt, log_lock): c["idx"] for c in plan}
        for fut in as_completed(futs):
            rec = fut.result()
            results[rec["idx"]] = rec

    manifest = [results[c["idx"]] for c in plan]
    locked = [m["idx"] for m in manifest if m.get("locked")]
    included = [(m, (out / "chunks" / f"{m['idx']:02d}.txt").read_text(encoding="utf-8").strip())
                for m in manifest if m["included"]]
    (out / "raw_transcript.txt").write_text("\n".join(t for _, t in included), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    total_words = sum(len(t.split()) for _, t in included)
    if locked:
        print(f"{_now()} [4/4] INCOMPLETE: {len(locked)} chunks hit 403 (account locked): {locked}. "
              f"Wait for quota then re-run.", flush=True)
    print(f"{_now()} [4/4] done: {len(included)}/{len(plan)} chunks merged | {total_words} words -> raw_transcript.txt", flush=True)
    return {"out_dir": str(out), "chunks": len(plan), "included": len(included), "words": total_words,
            "dur": dur, "locked": locked}


def main():
    ap = argparse.ArgumentParser(description="Transcribe audio in ~10-minute chunks (pure code, parallel)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default=None, help="default: config.MODEL")
    ap.add_argument("--workers", type=int, default=None, help="number of chunks sent in parallel, default config.CONCURRENCY")
    ap.add_argument("--prompt-file", default=None, help="path to a text file with a prompt override (default: built-in prompt)")
    a = ap.parse_args()
    if a.model:
        config.MODEL = a.model
    if a.workers:
        config.CONCURRENCY = a.workers
    t0 = time.time()
    run(a.input, a.out_dir, a.prompt_file)
    print(f"total time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
