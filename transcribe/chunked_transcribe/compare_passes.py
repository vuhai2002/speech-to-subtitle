"""Compare 2 independent transcripts of the SAME file (2 passes / 2 models) to find suspect spots - pure code.

For each chunk (same VAD boundaries): align the 2 word sequences with difflib, list the differing spans.
Spots where the 2 transcripts agree = trusted; differences >= min-run words = need listening (catches
mis-transcribed words that MMS is blind to). Output: print a summary + export json of the differing spots
with estimated timestamps.

Run: python -m transcribe.chunked_transcribe.compare_passes --a <out_dir_A> --b <out_dir_B> [--min-run 4]
"""
import argparse
import difflib
import json
import re
from pathlib import Path

_norm = lambda s: re.findall(r"\w+", s.lower())
_fmt = lambda s: "%02d:%02d" % (int(s) // 60, int(s) % 60)


def compare(a_dir: str, b_dir: str, min_run: int = 4) -> dict:
    A, B = Path(a_dir), Path(b_dir)
    plan = json.loads((A / "plan.json").read_text(encoding="utf-8"))
    chunks = {c["idx"]: c for c in plan["chunks"]}
    total = {"same": 0, "a_only": 0, "b_only": 0}
    diffs, per_chunk = [], []
    for k in sorted(chunks):
        fa, fb = A / "chunks" / f"{k:02d}.txt", B / "chunks" / f"{k:02d}.txt"
        if not (fa.exists() and fb.exists()):
            continue
        wa, wb = _norm(fa.read_text(encoding="utf-8")), _norm(fb.read_text(encoding="utf-8"))
        off, dur = chunks[k]["start"], chunks[k]["end"] - chunks[k]["start"]
        same = adiff = bdiff = blocks = 0
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=wa, b=wb, autojunk=False).get_opcodes():
            if tag == "equal":
                same += i2 - i1
                continue
            na, nb = i2 - i1, j2 - j1
            adiff += na if tag != "insert" else 0
            bdiff += nb if tag != "delete" else 0
            blocks += 1
            if max(na, nb) >= min_run:
                t = off + dur * (i1 / max(1, len(wa)))
                kind = {"delete": "only in A", "insert": "only in B", "replace": "different words"}[tag]
                diffs.append({"chunk": k, "at": _fmt(t), "sec": round(t), "kind": kind,
                              "len": max(na, nb), "a": " ".join(wa[i1:i2]), "b": " ".join(wb[j1:j2])})
        total["same"] += same
        total["a_only"] += adiff
        total["b_only"] += bdiff
        agree = 100 * same / max(1, same + max(adiff, bdiff))
        per_chunk.append({"idx": k, "a_words": len(wa), "b_words": len(wb), "agree": round(agree, 1), "diff_blocks": blocks})
        print(f"chunk {k:02d} {_fmt(off)}-{_fmt(chunks[k]['end'])} | A {len(wa):5} words, B {len(wb):5} words | match {agree:4.1f}% | {blocks} diff spans")
    diffs.sort(key=lambda d: d["sec"])
    ts, ta, tb = total["same"], total["a_only"], total["b_only"]
    agree_all = 100 * ts / max(1, ts + max(ta, tb))
    print(f"\nTOTAL: same {ts} | only A {ta} | only B {tb} | agreement {agree_all:.1f}% | diffs >= {min_run} words: {len(diffs)}")
    out = {"a_dir": str(A), "b_dir": str(B), "min_run": min_run, "agree_pct": round(agree_all, 1),
           "total": total, "per_chunk": per_chunk, "diffs": diffs}
    (A / "compare_passes.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {A / 'compare_passes.json'}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Compare 2 independent transcripts of the same file")
    ap.add_argument("--a", required=True, help="out-dir of pass 1")
    ap.add_argument("--b", required=True, help="out-dir of pass 2")
    ap.add_argument("--min-run", type=int, default=4, help="only list differences >= this many words")
    a = ap.parse_args()
    compare(a.a, a.b, a.min_run)


if __name__ == "__main__":
    main()
