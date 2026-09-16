"""Pure-code self-check, GROUNDED IN THE AUDIO (no AI, no reference transcript).

For each merged chunk: MMS forced alignment with '*' tokens inserted (slots where the audio has no matching text).
  - OMISSION: a star slot holding >= OMISSION_FLAG_SEC seconds of speech (VAD) -> likely a missing span.
  - NEEDS RE-LISTENING: a star slot holding OMISSION_REVIEW..FLAG seconds.
  - HALLUCINATION: >= HALLUCINATION_MIN_RUN consecutive words falling in a VAD-silent region.
Plus: repeated words at the seam between chunks. Exports qc_report.json + prints a raw summary.
"""
import json
import re
import sys
import time
from pathlib import Path

from . import audio_utils, config

sys.path.insert(0, str(config.PROJECT_ROOT))  # use realign's MMS in the repo


def _fmt(s: float) -> str:
    return "%02d:%02d" % (int(s) // 60, int(s) % 60)


def _align_star(models, emission_tools, mp3_path: str, words_raw: list[str]):
    """Time each word + the star slots. Returns (words, stars) with times relative to the chunk."""
    import torch
    from realign.align_words import load_audio, normalize_word, _emission
    import realign.config as rc
    model, tokenizer, aligner = models
    dev = next(model.parameters()).device
    wav, sr = load_audio(mp3_path, config.SAMPLE_RATE)
    emission = _emission(model, wav, dev, rc.EMIT_CHUNK_SEC, sr)
    ratio = wav.size(1) / emission.size(1)
    seq = [("star", "*")]
    for i, w in enumerate(words_raw):
        nw = normalize_word(w)
        if not nw:
            continue
        seq.append(("word", nw, i))
        if config.STAR_EVERY_WORD or re.search(r"[.!?;:,]$", w):
            seq.append(("star", "*"))
    if seq[-1][0] != "star":
        seq.append(("star", "*"))
    with torch.inference_mode():
        spans = aligner(emission[0].to(dev), tokenizer([s[1] for s in seq]))
    words, stars = [], []
    for item, sp in zip(seq, spans):
        if not sp:
            continue
        s = sp[0].start * ratio / sr
        e = sp[-1].end * ratio / sr
        if item[0] == "word":
            words.append({"w": words_raw[item[2]], "start": s, "end": e,
                          "score": sum(x.score for x in sp) / len(sp)})
        else:
            stars.append({"start": s, "end": e})
    torch.cuda.empty_cache()
    return words, stars


def _norm(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def check(out_dir: str) -> dict:
    from realign.align_words import get_models
    out = Path(out_dir)
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    segs_all = plan["segments"]
    included = [m for m in manifest if m["included"]]
    print(f"{time.strftime('%H:%M:%S')} self-check {len(included)} chunks (MMS + star token)", flush=True)
    models = get_models("cuda")

    report = {"model": plan["model"], "params": {"omission_flag_sec": config.OMISSION_FLAG_SEC,
              "review_sec": config.OMISSION_REVIEW_SEC, "hallucination_run": config.HALLUCINATION_MIN_RUN},
              "chunks": [], "seams": [], "excluded": [m for m in manifest if not m["included"]]}
    texts = {}
    for m in included:
        k = m["idx"]
        off = m["start"]
        mp3 = str(out / "chunks" / f"{k:02d}.mp3")
        text = (out / "chunks" / f"{k:02d}.txt").read_text(encoding="utf-8").strip()
        texts[k] = text
        words_raw = text.split()
        segs = [[max(0.0, s - off), min(m["end"] - off, e - off)] for s, e in segs_all if e > off and s < m["end"]]
        words, stars = _align_star(models, None, mp3, words_raw)
        for st in stars:
            st["speech"] = audio_utils.speech_in(segs, st["start"], st["end"])
        omission = [{"from": _fmt(s["start"] + off), "to": _fmt(s["end"] + off), "speech": round(s["speech"], 1)}
                    for s in sorted(stars, key=lambda x: -x["speech"]) if s["speech"] >= config.OMISSION_REVIEW_SEC]
        silent = [audio_utils.speech_in(segs, w["start"] - config.SILENCE_PAD_SEC, w["end"] + config.SILENCE_PAD_SEC) == 0.0
                  for w in words]
        runs, j = [], 0
        while j < len(words):
            if silent[j]:
                t = j
                while t < len(words) and silent[t]:
                    t += 1
                if t - j >= config.HALLUCINATION_MIN_RUN:
                    runs.append({"from": _fmt(words[j]["start"] + off), "to": _fmt(words[t - 1]["end"] + off),
                                 "words": t - j, "text": " ".join(w["w"] for w in words[j:j + 12])})
                j = t
            else:
                j += 1
        info = {"idx": k, "start_min": round(m["start"] / 60, 1), "end_min": round(m["end"] / 60, 1),
                "speech_sec": m["speech_sec"], "words": len(words_raw), "attempts": m["attempts"],
                "mean_score": round(sum(w["score"] for w in words) / max(1, len(words)), 3),
                "omission_flags": [o for o in omission if o["speech"] >= config.OMISSION_FLAG_SEC],
                "review": [o for o in omission if o["speech"] < config.OMISSION_FLAG_SEC],
                "hallucination_runs": runs}
        report["chunks"].append(info)
        print(f"   chunk {k:02d} {info['start_min']:5.1f}-{info['end_min']:5.1f}m | {info['words']:5} words | MMS score {info['mean_score']}"
              f" | OMISSION {len(info['omission_flags'])} | review {len(info['review'])} | HALLUC {len(info['hallucination_runs'])}", flush=True)
        for o in info["omission_flags"]:
            print(f"      !! OMISSION {o['from']}-{o['to']}: {o['speech']}s of speech with no text", flush=True)
        for r in info["hallucination_runs"]:
            print(f"      ?? HALLUC {r['from']}-{r['to']} ({r['words']} words): {r['text']}", flush=True)

    nz = lambda xs: [re.sub(r"\W", "", x.lower()) for x in xs]
    for a, b in zip(included, included[1:]):
        ta, tb = texts[a["idx"]].split(), texts[b["idx"]].split()
        dup = max([n for n in range(1, 31) if len(ta) >= n and len(tb) >= n and nz(ta[-n:]) == nz(tb[:n])] or [0])
        report["seams"].append({"at": _fmt(a["end"]), "between": [a["idx"], b["idx"]], "dup_words": dup})
        if dup:
            print(f"   seam {a['idx']:02d}|{b['idx']:02d} at {_fmt(a['end'])}: REPEAT {dup} words at the edge", flush=True)

    (out / "qc_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    n_flag = sum(len(c["omission_flags"]) for c in report["chunks"])
    n_rev = sum(len(c["review"]) for c in report["chunks"])
    n_hall = sum(len(c["hallucination_runs"]) for c in report["chunks"])
    n_dup = sum(1 for s in report["seams"] if s["dup_words"])
    print(f"{time.strftime('%H:%M:%S')} TOTAL: omissions {n_flag} | review {n_rev} | halluc {n_hall} | seam repeats {n_dup} "
          f"| chunks dropped {len(report['excluded'])} -> qc_report.json", flush=True)
    return report


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Self-check the chunked transcript (pure code, audio-grounded)")
    ap.add_argument("--out-dir", required=True)
    check(ap.parse_args().out_dir)


if __name__ == "__main__":
    main()
