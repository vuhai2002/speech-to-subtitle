"""Pure-code orchestrator: audio -> ~10-minute chunks -> transcribe (PARALLEL) -> merge -> raw output.

Output (no AI):
  <out>/mono16k.mp3, <out>/plan.json
  <out>/polarity.json        polarity check of the mono mixdown: sections of the source flipped, output check (audio_utils.py)
  <out>/chunks/NN.mp3, <out>/chunks/NN.txt
  <out>/raw_transcript.txt   merged transcript of the chunks with speech (a silent chunk's text is scored later by build_srt)
  <out>/manifest.json        per-chunk status + guard
  <out>/run_trace.json       everything the run did: every try, outcome and final text per chunk (run_trace.py)

Processes config.CONCURRENCY chunks in parallel (default 3). Each chunk retries up to MAX_ATTEMPTS on failure.
On 403 (account locked): soft stop - mark the chunk as not transcribed, report the file as incomplete
(exit code 3, see transcribe/exit_codes.py).
If the mono mix still cancels the voice after the polarity flip, nothing is transcribed (exit code 5).
Chunks are independent, so they are merged back in order once done.
"""
import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from . import audio_utils, config, mai_fallback, polarity, run_trace, transcribe_client
from transcribe import exit_codes


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


SILENT_BLOCKED_NOTE = "silent blocked: gemini filters refused a chunk without speech"


def silent_note() -> str:
    """Manifest note of a chunk VAD hears as silent whose text goes on to build_srt's sentence scoring."""
    return f"silent: sentences kept when MMS score >= {config.SILENT_MIN_SCORE:g}"


def _process_chunk(c: dict, out: Path, mono: str, prompt: str, log_lock: threading.Lock) -> dict:
    """Cut one chunk from mono16k then transcribe (with retry). Runs in one pool thread.

    A chunk VAD hears speech in: merged; if Gemini's filter still blocks it after the retries, MAI transcribes
    it when an OpenRouter key is set (engine="mai"), else it fails like any other chunk.
    A chunk VAD hears as silent: its text is merged with "silent": true and build_srt keeps only the sentences
    MMS aligns with a score >= config.SILENT_MIN_SCORE; no text = "skipped (no speech)"; still blocked after
    the retries = left empty with SILENT_BLOCKED_NOTE and never sent to MAI (a paid call on a chunk with no
    teaching).
    """
    k = c["idx"]
    mp3 = str(out / "chunks" / f"{k:02d}.mp3")
    audio_utils.cut_chunk(mono, c["start"], c["end"], mp3)
    silent = not audio_utils.chunk_has_speech(c)
    r = transcribe_client.transcribe_with_retry(
        mp3, prompt, c["speech_sec"], str(out / "chunks" / f"{k:02d}_req.json"), str(out / "chunks" / f"{k:02d}_raw.sse"))
    res = r["res"]
    blocked = bool(r.get("blocked"))
    text = "" if blocked else res.get("text", "")
    reasons = r["reasons"]
    engine, mai = None, None
    if blocked and not silent:
        fb = mai_fallback.transcribe_blocked_chunk(mp3, str(out / "chunks" / f"{k:02d}_mai.json"), c)
        mai = {"outcome": "ok" if fb["ok"] else "failed", "reason": fb["why"], "words": len(fb["text"].split()),
               "cost": fb.get("cost")}
        if fb["ok"]:
            text, engine, reasons = fb["text"], "mai", []
        else:
            reasons = reasons + [fb["why"]]
    (out / "chunks" / f"{k:02d}.txt").write_text(text, encoding="utf-8")
    words = len(text.split())
    rec = {**c, "attempts": r["attempts"], "http": res.get("http"), "served_model": res.get("served_model"),
           "finish": res.get("finish"), "words": words, "usage": res.get("usage"),
           "included": False, "locked": r["locked"], "silent": silent, "note": "; ".join(reasons),
           "history": r.get("history", []), "mai": mai}
    if engine:
        rec["engine"] = engine
    if r["locked"]:
        rec["note"] = "403 account locked - not transcribed"
    elif blocked and silent:
        rec["note"] = SILENT_BLOCKED_NOTE
    elif reasons:                                   # failed after all retries (error / empty / unreachable / blocked) -> do not merge
        rec["note"] = "error after retry: " + rec["note"]
    elif silent:
        if words:
            rec["included"], rec["note"] = True, silent_note()
        else:
            rec["note"] = "skipped (no speech)"
    else:
        rec["included"] = True
        if engine == "mai":
            rec["note"] = "mai fallback: gemini blocked by filters"
    density = words / (c["speech_sec"] / 60) if c["speech_sec"] > 0 else 0
    with log_lock:
        state = "MERGED" if rec["included"] else ("LOCKED-403" if r["locked"] else "SKIP: " + rec["note"])
        if engine == "mai":
            tail = " | MAI (gemini blocked)"
        elif rec["included"] and silent:
            tail = " | silent: scored at build"
        else:
            tail = ""
        print(f"    chunk {k:02d} {c['start']/60:5.1f}-{c['end']/60:5.1f}m | speech {c['speech_sec']:4.0f}s | {words:5} words "
              f"({density:3.0f}/min) | {r['attempts']} tries | {res.get('finish')} | {state}{tail}", flush=True)
    return rec


def run(input_path: str, out_dir: str, prompt_file: str | None = None) -> dict:
    started_at = run_trace.now_iso()
    out = Path(out_dir)
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    prompt = config.load_prompt(prompt_file)

    mono = str(out / "mono16k.mp3")
    print(f"{_now()} [1/4] ffmpeg -> mono 16kHz", flush=True)
    dur = audio_utils.to_mono16k(input_path, mono)
    pol_file = out / audio_utils.POLARITY_FILE
    pol = json.loads(pol_file.read_text(encoding="utf-8")) if pol_file.exists() else None
    if pol and pol["sections"]:
        print(f"    polarity: {len(pol['sections'])} inverted section(s) flipped before the mixdown", flush=True)
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
    failed = [m["idx"] for m in manifest if m["note"].startswith("error after retry")]
    texts = {m["idx"]: (out / "chunks" / f"{m['idx']:02d}.txt").read_text(encoding="utf-8").strip() for m in manifest}
    included = [(m, texts[m["idx"]]) for m in manifest if m["included"]]
    # raw_transcript.txt holds teaching text only: a silent chunk's text is unscored until build_srt.
    (out / "raw_transcript.txt").write_text("\n".join(t for m, t in included if not m.get("silent")), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    run_trace.save(out, run_trace.new_trace(
        input_path=input_path, duration_sec=dur, prompt=prompt, segments=segs, manifest=manifest, texts=texts,
        started_at=started_at, mai_enabled=bool(mai_fallback.mai_config.API_KEY), polarity=pol))
    total_words = sum(len(t.split()) for m, t in included if not m.get("silent"))   # the words of raw_transcript.txt
    n_silent = sum(1 for m, _ in included if m.get("silent"))
    if locked:
        print(f"{_now()} [4/4] INCOMPLETE: {len(locked)} chunks hit 403 (account locked): {locked}. "
              f"Wait for quota then re-run.", flush=True)
    if failed:
        print(f"{_now()} [4/4] INCOMPLETE: {len(failed)} chunks failed after retry: {failed}. Re-run.", flush=True)
    print(f"{_now()} [4/4] done: {len(included)}/{len(plan)} chunks merged | {total_words} words -> raw_transcript.txt"
          + (f" | {n_silent} silent chunk(s) scored at build" if n_silent else ""), flush=True)
    return {"out_dir": str(out), "chunks": len(plan), "included": len(included), "words": total_words,
            "dur": dur, "locked": locked, "failed": failed}


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
    try:
        res = run(a.input, a.out_dir, a.prompt_file)
    except polarity.AudioCancelled as e:
        print(f"AUDIO_CANCELLED: {e}", file=sys.stderr, flush=True)
        sys.exit(exit_codes.AUDIO_CANCELLED)
    print(f"total time {time.time() - t0:.0f}s")
    bad = sorted(set(res["failed"]) | set(res["locked"]))
    if bad:
        print(f"INCOMPLETE: chunks {bad} have no transcript (failed after retry or 403); "
              "see manifest.json", file=sys.stderr, flush=True)
        sys.exit(exit_codes.INCOMPLETE)


if __name__ == "__main__":
    main()
