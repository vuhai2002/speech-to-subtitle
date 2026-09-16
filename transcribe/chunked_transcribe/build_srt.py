"""Bridge chunked_transcribe -> realign: build .srt from the chunked transcript.

Align EACH CHUNK with MMS (realign.align_words.align), then add the chunk's time offset,
gather all words onto a single timeline, then pass them through realign's cue-building stage
(cue_builder -> cue_clamp -> srt_writer). Do NOT use window_align on the whole file (it mistimes when
the file ends with a long stretch of no speech).

Run: python -m transcribe.chunked_transcribe.build_srt --out-dir <out_ct> [--device cuda]
Output: <out_ct>/<original file name>.srt
"""
import argparse
import json
import sys
import time
from pathlib import Path

from . import config

sys.path.insert(0, str(config.PROJECT_ROOT))  # so realign in the repo can be imported


def _aligned_words(out_dir: str, device: str) -> tuple[list[dict], tuple | None, str]:
    """Align each merged chunk, add offsets -> (global_words, speech_region, original_file_name)."""
    from realign.align_words import align, get_models
    out = Path(out_dir)
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    included = [m for m in manifest if m["included"]]
    models = get_models(device)
    global_words: list[dict] = []
    for m in included:
        k, off = m["idx"], m["start"]
        mp3 = str(out / "chunks" / f"{k:02d}.mp3")
        words_raw = (out / "chunks" / f"{k:02d}.txt").read_text(encoding="utf-8").split()
        t0 = time.time()
        words = align(mp3, words_raw, device, models=models)
        for w in words:
            if w["start"] is not None:
                w["start"] += off
                w["end"] += off
        global_words += words
        al = sum(1 for w in words if w["start"] is not None)
        print(f"    chunk {k:02d} align {al}/{len(words)} words ({time.time() - t0:.0f}s)", flush=True)
    segs = plan.get("segments") or []
    region = (segs[0][0], segs[-1][1]) if segs else None
    name = Path(plan["input"]).stem
    return global_words, region, name


def build(out_dir: str, device: str = "cuda") -> str:
    import realign.config as rcfg
    from realign.cue_builder import build_cues
    from realign.cue_clamp import clamp_cues
    from realign.srt_writer import cues_to_srt
    out = Path(out_dir)
    print(f"{time.strftime('%H:%M:%S')} align each chunk (MMS, {device})", flush=True)
    words, region, name = _aligned_words(out_dir, device)
    print(f"{time.strftime('%H:%M:%S')} build cues (realign) from {len(words)} words | vad {region}", flush=True)
    cues = build_cues(words, rcfg)
    cues = clamp_cues(cues, region, rcfg)
    srt_path = out / f"{name}.srt"
    srt_path.write_text(cues_to_srt(cues), encoding="utf-8")
    scores = [w["score"] for w in words if w.get("score") is not None]
    last = cues[-1]["end"] if cues else 0.0
    print(f"{time.strftime('%H:%M:%S')} done: {len(cues)} cues | last cue ends {int(last)//60:02d}:{int(last)%60:02d}"
          f" | mean MMS score {sum(scores)/len(scores):.3f} | -> {srt_path}", flush=True)
    return str(srt_path)


def main():
    ap = argparse.ArgumentParser(description="Build .srt from the chunked transcript (align each chunk + realign cue building)")
    ap.add_argument("--out-dir", required=True, help="run_pipeline output directory")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    build(a.out_dir, a.device)


if __name__ == "__main__":
    main()
