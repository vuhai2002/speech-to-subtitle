"""Re-align batch orchestrator: pair -> copy Y:->temp -> VAD -> align (cache+resume)
-> segment -> manifest.csv. Model loaded once. Safe to re-run (resume)."""
import argparse
import csv
import json
import os
import random
import shutil
import sys
import time

import realign.config as config
from realign.pair_audio_srt import build_pairs, format_unpaired_report
from realign.srt_to_txt import words_from_file
from realign.align_words import align, get_models
from realign.window_align import window_align
from realign.segment_cues import segment

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MANIFEST_FIELDS = ["key", "srt_name", "audio_name", "status", "words",
                   "aligned", "cues", "mean_score", "vad", "error"]


def out_paths(out_dir: str, srt_name: str):
    stem = os.path.splitext(srt_name)[0]
    return os.path.join(out_dir, "words", stem + ".json"), os.path.join(out_dir, "srt", stem + ".srt")


def should_align(words_json: str, force: bool) -> bool:
    return force or not os.path.exists(words_json)


def select_pairs(pairs, limit=0, sample=0, seed=0, only=""):
    ps = [p for p in pairs if only.lower() in p["key"]] if only else list(pairs)
    if sample and sample < len(ps):
        ps = sorted(random.Random(seed).sample(ps, sample), key=lambda p: p["srt_name"])
    elif limit and limit < len(ps):
        ps = ps[:limit]
    return ps


def copy_with_retry(src, dst, attempts=3, sleep=2.0):
    last = None
    for _ in range(attempts):
        try:
            shutil.copyfile(src, dst)
            return
        except Exception as e:  # noqa: BLE001 - Drive may drop the network
            last = e
            time.sleep(sleep)
    raise last


def _safe_vad(path, vad_model):
    try:
        from realign.vad_speech_region import speech_region
        return speech_region(path, model=vad_model)
    except Exception:
        return None


def process_one(pair, out_dir, device, models, vad_model, emit_chunk_sec, max_sec, force, window_sec=0.0):
    key, srt_name = pair["key"], pair["srt_name"]
    words_json, out_srt = out_paths(out_dir, srt_name)
    row = {f: "" for f in MANIFEST_FIELDS}
    row.update(key=key, srt_name=srt_name, audio_name=os.path.basename(pair["audio_path"]))
    try:
        if should_align(words_json, force):
            tmp = os.path.join(out_dir, "_tmp", os.path.splitext(srt_name)[0] + ".mp3")
            os.makedirs(os.path.dirname(tmp), exist_ok=True)
            copy_with_retry(pair["audio_path"], tmp)
            try:
                words_raw = words_from_file(pair["srt_path"], pair.get("kind", "srt"))
                region = _safe_vad(tmp, vad_model)
                if window_sec > 0:
                    words = window_align(tmp, words_raw, device, emit_chunk_sec,
                                         target_window_sec=window_sec, models=models)
                else:
                    words = align(tmp, words_raw, device, emit_chunk_sec, max_sec, models=models)
                os.makedirs(os.path.dirname(words_json), exist_ok=True)
                with open(words_json, "w", encoding="utf-8") as fh:
                    json.dump({"key": key, "audio": pair["audio_path"], "vad": region, "words": words},
                              fh, ensure_ascii=False)
            finally:
                try:
                    os.remove(tmp)               # delete temp even if align fails (avoid filling the disk)
                except OSError:
                    pass
        with open(words_json, encoding="utf-8") as fh:
            data = json.load(fh)
        region = tuple(data["vad"]) if data.get("vad") else None
        os.makedirs(os.path.dirname(out_srt), exist_ok=True)
        cues = segment(words_json, out_srt, region)
        sc = [w["score"] for w in data["words"] if w["score"] is not None]
        row.update(status="ok", words=len(data["words"]),
                   aligned=sum(1 for w in data["words"] if w["start"] is not None),
                   cues=len(cues), mean_score=round(sum(sc) / len(sc), 3) if sc else "",
                   vad=(f"{region[0]:.1f}-{region[1]:.1f}" if region else "none"))
    except Exception as e:  # noqa: BLE001 - one failed file does not stop the whole batch
        row.update(status="error", error=str(e)[:200])
    return row


def run(args):
    src_dir = args.txt_dir or args.srt_dir
    ext = ".txt" if args.txt_dir else ".srt"
    pairs, us, ua = build_pairs(src_dir, args.audio_dir or config.AUDIO_DIRS, ext)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "unpaired-realign.md"), "w", encoding="utf-8") as f:
        f.write(format_unpaired_report(us, ua))
    todo = select_pairs(pairs, args.limit, args.sample, args.seed, args.only)
    print(f"pairs={len(pairs)} todo={len(todo)} (unpaired_srt={len(us)})")

    models = get_models(args.device)
    from realign.vad_speech_region import load_vad_model
    vad_model = load_vad_model()

    man_path = os.path.join(args.out_dir, "manifest.csv")
    with open(man_path, "w", encoding="utf-8", newline="") as mf:
        wr = csv.DictWriter(mf, fieldnames=MANIFEST_FIELDS)
        wr.writeheader()
        for i, p in enumerate(todo, 1):
            t0 = time.time()
            row = process_one(p, args.out_dir, args.device, models, vad_model,
                              args.emit_chunk_sec, args.max_sec, args.force, args.window_sec)
            wr.writerow(row)
            mf.flush()
            print(f"[{i}/{len(todo)}] {row['status']:5} {row['srt_name'][:45]:45} "
                  f"cues={row['cues']} ({time.time() - t0:.0f}s)")
    print(f"done -> {man_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srt-dir", help=".srt directory (exactly one of --srt-dir / --txt-dir)")
    ap.add_argument("--txt-dir", help=".txt transcript directory (exactly one of --srt-dir / --txt-dir)")
    ap.add_argument("--audio-dir", action="append")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--emit-chunk-sec", type=float, default=config.EMIT_CHUNK_SEC)
    ap.add_argument("--max-sec", type=float, default=0.0)
    ap.add_argument("--window-sec", type=float, default=0.0,
                    help="If >0: forced-align in windows of ~N seconds (for long files that OOM)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if bool(a.srt_dir) == bool(a.txt_dir):
        ap.error("need exactly ONE of --srt-dir or --txt-dir")
    run(a)


if __name__ == "__main__":
    main()
