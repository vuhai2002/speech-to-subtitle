"""Export a side-by-side comparison HTML of 2 transcripts (left-right columns), highlighting differences.

One table row per aligned span (difflib) so the 2 columns always line up. Differences >= --major words are
numbered + get a navigation bar to jump to them. Punctuation/casing is preserved when displaying.

Run: python -m transcribe.chunked_transcribe.render_compare_html --a <dirA> --b <dirB> --out cmp.html
"""
import argparse
import difflib
import html
import json
import re
from pathlib import Path

_key = lambda t: re.sub(r"[^0-9a-zà-ỹ']", "", t.lower())
_fmt = lambda s: "%02d:%02d" % (int(s) // 60, int(s) % 60)
_esc = lambda toks: html.escape(" ".join(toks))

CSS = """
:root { --bg:#fff; --fg:#1a1a1a; --line:#e2e2e2; --hdr:#f3f4f6; --eq:#555;
  --a-bg:#fff4d6; --a-fg:#7a4f00; --b-bg:#d9f0ff; --b-fg:#004a6e; --badge:#d33; }
@media (prefers-color-scheme: dark) { :root { --bg:#161616; --fg:#e6e6e6; --line:#333; --hdr:#222;
  --eq:#9a9a9a; --a-bg:#4a3a10; --a-fg:#ffd98a; --b-bg:#123043; --b-fg:#9fd8ff; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font-family:'Segoe UI',system-ui,'Noto Sans',Arial,sans-serif; font-size:15px; line-height:1.6; }
header { position:sticky; top:0; background:var(--bg); border-bottom:2px solid var(--line); padding:10px 14px; z-index:5; }
h1 { font-size:17px; margin:0 0 6px; }
.cols { display:flex; gap:16px; font-weight:600; }
.cols .a { color:var(--a-fg); } .cols .b { color:var(--b-fg); }
.nav { margin-top:8px; font-size:13px; line-height:2; }
.nav a { display:inline-block; margin-right:6px; padding:1px 7px; border:1px solid var(--line); border-radius:5px; text-decoration:none; color:var(--fg); }
.legend { font-size:12.5px; color:var(--eq); margin-top:6px; }
table { width:100%; border-collapse:collapse; }
td { width:50%; vertical-align:top; padding:2px 14px; }
tr.hdr td { background:var(--hdr); font-weight:700; text-align:center; padding:8px; border-top:1px solid var(--line); }
td.eq { color:var(--eq); }
tr.minor td.a, tr.major td.a { background:var(--a-bg); color:var(--a-fg); }
tr.minor td.b, tr.major td.b { background:var(--b-bg); color:var(--b-fg); }
tr.major td { outline:2px solid var(--badge); }
tr.major { scroll-margin-top:120px; }
.badge { display:inline-block; min-width:18px; text-align:center; background:var(--badge); color:#fff; font-size:12px; font-weight:700; border-radius:9px; padding:0 5px; margin-right:6px; }
i { color:var(--eq); }
"""


def render(a_dir: str, b_dir: str, out_path: str, a_name="pass 1", b_name="pass 2", major=4) -> str:
    A, B = Path(a_dir), Path(b_dir)
    chunks = {c["idx"]: c for c in json.loads((A / "plan.json").read_text(encoding="utf-8"))["chunks"]}
    rows, nav, major_no = [], [], 0
    for k in sorted(chunks):
        ta = (A / "chunks" / f"{k:02d}.txt").read_text(encoding="utf-8").split()
        tb = (B / "chunks" / f"{k:02d}.txt").read_text(encoding="utf-8").split()
        ka, kb = [_key(t) for t in ta], [_key(t) for t in tb]
        off, dur = chunks[k]["start"], chunks[k]["end"] - chunks[k]["start"]
        rows.append(f'<tr class="hdr"><td colspan="2">Chunk {k:02d} &middot; {_fmt(off)} - {_fmt(chunks[k]["end"])}</td></tr>')
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ka, b=kb, autojunk=False).get_opcodes():
            la, lb = _esc(ta[i1:i2]), _esc(tb[j1:j2])
            if tag == "equal":
                rows.append(f'<tr><td class="eq">{la}</td><td class="eq">{lb}</td></tr>')
                continue
            big = max(i2 - i1, j2 - j1) >= major
            anchor = ""
            if big:
                major_no += 1
                t = off + dur * (i1 / max(1, len(ka)))
                anchor = f' id="d{major_no}"'
                nav.append((major_no, _fmt(t)))
                badge = f'<span class="badge">{major_no}</span>'
                la, lb = badge + (la or "<i>(none)</i>"), badge + (lb or "<i>(none)</i>")
            else:
                la, lb = la or "<i>(none)</i>", lb or "<i>(none)</i>"
            rows.append(f'<tr class="{"major" if big else "minor"}"{anchor}><td class="a">{la}</td><td class="b">{lb}</td></tr>')
    nav_html = " ".join(f'<a href="#d{n}">{n} ({ts})</a>' for n, ts in nav)
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width, initial-scale=1">'
           f'<title>Transcript comparison</title><style>{CSS}</style></head><body>'
           f'<header><h1>Transcript comparison</h1>'
           f'<div class="cols"><div class="a">&#9632; {html.escape(a_name)}</div>'
           f'<div class="b">&#9632; {html.escape(b_name)}</div></div>'
           f'<div class="nav"><b>{major_no} major differences (&ge;{major} words):</b> {nav_html}</div>'
           f'<div class="legend">Yellow background = pass 1. Blue background = pass 2. Red outline = numbered major differences. '
           f'Grey text = identical. Estimated timestamps.</div></header>'
           f'<table><tbody>{chr(10).join(rows)}</tbody></table></body></html>')
    Path(out_path).write_text(doc, encoding="utf-8")
    print(f"major (>={major} words): {major_no} | -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Export a comparison HTML of 2 transcripts")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--a-name", default="pass 1")
    ap.add_argument("--b-name", default="pass 2")
    ap.add_argument("--major", type=int, default=4)
    a = ap.parse_args()
    render(a.a, a.b, a.out, a.a_name, a.b_name, a.major)


if __name__ == "__main__":
    main()
