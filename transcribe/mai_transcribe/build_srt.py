"""Build .srt from MAI's NATIVE timestamps (NO MMS forced-align, NO GPU needed).

MAI returns each word as {word,start,end} with punctuation -> passed straight through realign's cue-building
stage (cue_builder -> cue_clamp -> srt_writer). Since the model already provides real timestamps, the MMS
align step of chunked_transcribe's build_srt is skipped entirely. Verified: the resulting SRT is equivalent to
Gemini+MMS (see docs/mai-transcribe-notes.md).

Run: python -m transcribe.mai_transcribe.build_srt --out-dir <out>
Output: <out>/<original file name>.srt
"""
import argparse
import json
import sys
import time
from pathlib import Path

from . import config
from transcribe import exit_codes

sys.path.insert(0, str(config.PROJECT_ROOT))  # so realign in the repo can be imported


def build(out_dir: str) -> str:
    import realign.config as rcfg
    from realign.cue_builder import build_cues
    from realign.cue_clamp import clamp_cues
    from realign.srt_writer import cues_to_srt
    out = Path(out_dir)
    words = json.loads((out / "mai_words.json").read_text(encoding="utf-8"))
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    segs = plan.get("segments") or []
    region = (segs[0][0], segs[-1][1]) if segs else None
    name = Path(plan["input"]).stem
    if not any(w.get("start") is not None for w in words):
        raise exit_codes.NoAlignedWords(f"no timestamped word out of {len(words)} (nothing to write)")
    print(f"{time.strftime('%H:%M:%S')} build cues from {len(words)} MAI words | vad {region}", flush=True)
    cues = build_cues(words, rcfg)
    cues = clamp_cues(cues, region, rcfg)
    if not cues:
        raise exit_codes.NoAlignedWords(f"{len(words)} words built no cue (nothing to write)")
    srt_path = out / f"{name}.srt"
    srt_path.write_text(cues_to_srt(cues), encoding="utf-8")
    last = cues[-1]["end"]
    print(f"{time.strftime('%H:%M:%S')} done: {len(cues)} cues | last cue ends "
          f"{int(last)//60:02d}:{int(last)%60:02d} -> {srt_path}", flush=True)
    return str(srt_path)


def main():
    ap = argparse.ArgumentParser(description="Build .srt from MAI's native timestamps (no MMS, no GPU)")
    ap.add_argument("--out-dir", required=True, help="run_pipeline output directory")
    a = ap.parse_args()
    try:
        build(a.out_dir)
    except exit_codes.NoAlignedWords as e:
        print(f"NO_ALIGNED_WORDS: {e}", file=sys.stderr, flush=True)
        sys.exit(exit_codes.NO_ALIGNED_WORDS)


if __name__ == "__main__":
    main()
