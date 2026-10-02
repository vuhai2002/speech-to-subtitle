"""run_trace.json: one file that tells everything a run did, so a run never has to be repeated to be audited.

run_pipeline writes it at the end of every run, complete or not: the run metadata, the VAD segments and, per
chunk, every Gemini try with its outcome, the MAI fallback result and the final text, plus the optional
"polarity" record of the mono mixdown (the sections of the source flipped, see polarity.py). build_srt adds the
alignment results, every sentence's score for chunks VAD hears as silent, and the cue summary (add_build).
The desktop app adds its own "app" section and archives the file. Times are seconds from the start of the
input file. Schema version 1.
"""
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from . import config

FILE = "run_trace.json"
VERSION = 1


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def git_commit() -> str | None:
    """Short commit of the repo this code runs from, or None when git is unavailable."""
    try:
        out = subprocess.run(["git", "-C", str(config.PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _outcome(rec: dict) -> str:
    if rec.get("locked"):
        return "locked"
    if rec.get("engine") == "mai":
        return "mai"
    if rec["note"].startswith("silent blocked"):
        return "silent-blocked"
    if rec["note"].startswith("error after retry"):
        return "failed"
    if rec.get("silent"):
        return "silent-scored" if rec["included"] else "silent-empty"
    return "merged"


def chunk_entry(rec: dict, text: str) -> dict:
    """The trace view of one manifest record (see run_pipeline._process_chunk). `text` is kept only for a merged
    chunk: the final Gemini or MAI text; a chunk that was not merged has no usable text."""
    return {"idx": rec["idx"], "start": rec["start"], "end": rec["end"], "speech_sec": rec["speech_sec"],
            "vad": "silent" if rec.get("silent") else "speech", "attempts": rec.get("history", []),
            "mai": rec.get("mai"), "outcome": _outcome(rec), "note": rec["note"],
            "text": text if rec["included"] else ""}


def new_trace(*, input_path: str, duration_sec: float, prompt: str, segments: list, manifest: list[dict],
              texts: dict[int, str], started_at: str, mai_enabled: bool, polarity: dict | None = None) -> dict:
    """The trace of a finished run_pipeline run. `polarity` is the record to_mono16k left in polarity.json (which
    sections of the source were flipped before the mixdown); a trace without it has no "polarity" key."""
    trace = {
        "version": VERSION,
        "meta": {
            "mode": "full", "input": Path(input_path).name, "duration_sec": round(duration_sec, 3),
            "pipeline_commit": git_commit(), "model": config.MODEL,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "thresholds": {"min_speech_chunk_sec": config.MIN_SPEECH_CHUNK_SEC,
                           "min_speech_ratio": config.MIN_SPEECH_RATIO,
                           "silent_min_score": config.SILENT_MIN_SCORE, "max_attempts": config.MAX_ATTEMPTS},
            "mai_enabled": mai_enabled, "started_at": started_at, "finished_at": now_iso(),
        },
        "vad": {"speech_sec": round(sum(e - s for s, e in segments), 1), "segments": segments},
        "chunks": [chunk_entry(m, texts.get(m["idx"], "")) for m in manifest],
    }
    if polarity is not None:
        trace["polarity"] = polarity
    return trace


def add_build(trace: dict, results: dict[int, dict], summary: dict) -> dict:
    """Fold build_srt's results into the trace: per chunk the "build" alignment stats and, for a chunk VAD hears
    as silent, its scored "sentences"; plus the run-level "build" summary {cues, last_cue_end, mean_score, exit}."""
    for entry in trace.get("chunks", []):
        r = results.get(entry.get("idx"))
        if r is None:
            continue
        entry["build"] = {k: r[k] for k in ("total_words", "aligned_words", "mean_score")}
        if r.get("align_error"):
            entry["build"]["align_error"] = r["align_error"]
        if "sentences" in r:
            entry["sentences"] = r["sentences"]
    trace["build"] = summary
    return trace


def _finite(value):
    """`value` with every NaN or infinite float replaced by None. JSON has no such numbers: json.dumps would write
    a bare NaN token, which strict readers (the desktop's JSON.parse, Postgres jsonb) refuse, losing the trace."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def write(out_dir: str | Path, trace: dict) -> None:
    text = json.dumps(_finite(trace), ensure_ascii=False, indent=1, allow_nan=False)
    (Path(out_dir) / FILE).write_text(text, encoding="utf-8")


def save(out_dir: str | Path, trace: dict) -> None:
    """write(), but a trace that cannot be written only warns on stderr: it is an audit aid and must never change
    a run's outcome (exit code, manifest.json, the .srt)."""
    try:
        write(out_dir, trace)
    except (OSError, TypeError, ValueError) as e:
        print(f"WARNING: could not write {FILE}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)


def read(out_dir: str | Path) -> dict | None:
    """The trace, or None when it is missing or unreadable (an older run, or a crash before it was written)."""
    try:
        return json.loads((Path(out_dir) / FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
