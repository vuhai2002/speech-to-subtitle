"""Stage 3 (segment): words.json (+VAD) -> cue (production rules) -> clamp -> srt.
Re-runs instantly for all 713 files without re-aligning."""
import argparse
import json

import realign.config as config
from realign.cue_builder import build_cues
from realign.cue_clamp import clamp_cues
from realign.srt_writer import cues_to_srt


def segment(words_json: str, out_srt: str, vad_region=None, cfg=config) -> list[dict]:
    with open(words_json, encoding="utf-8") as f:
        data = json.load(f)
    cues = build_cues(data["words"], cfg)
    cues = clamp_cues(cues, vad_region, cfg)
    with open(out_srt, "w", encoding="utf-8") as f:
        f.write(cues_to_srt(cues))
    return cues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words-json", required=True)
    ap.add_argument("--out-srt", required=True)
    a = ap.parse_args()
    with open(a.words_json, encoding="utf-8") as f:
        data = json.load(f)
    region = tuple(data["vad"]) if data.get("vad") else None
    cues = segment(a.words_json, a.out_srt, region)
    print(f"wrote {len(cues)} cues -> {a.out_srt}")


if __name__ == "__main__":
    main()
