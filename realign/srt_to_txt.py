"""SRT -> transcript txt: take the cue text (keep words + punctuation), drop index/timestamp.
Line breaks in the txt do not matter (the segment stage rebuilds cues by word time),
but KEEP punctuation because segment uses it to break sentences.
"""
import argparse
import re


def parse_srt_cues(srt_text: str) -> list[str]:
    srt_text = srt_text.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")
    cues: list[str] = []
    for block in re.split(r"\n\s*\n", srt_text):
        lines = [ln for ln in block.split("\n") if ln.strip() != ""]
        if not lines:
            continue
        ts_idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if ts_idx is None:
            continue
        text = " ".join(ln.strip() for ln in lines[ts_idx + 1:]).strip()
        if text:
            cues.append(text)
    return cues


def srt_to_transcript(srt_text: str) -> str:
    return "\n".join(parse_srt_cues(srt_text))


def words_from_file(path: str, kind: str) -> list[str]:
    """Read a transcript -> list of words to align. kind='txt': read directly with .split() (raw transcript);
    'srt': join cue text then .split() (drop index/timestamp)."""
    with open(path, encoding="utf-8") as fh:
        raw = fh.read()
    if kind == "txt":
        return raw.split()
    return " ".join(parse_srt_cues(raw)).split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.srt, encoding="utf-8") as f:
        text = f.read()
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(srt_to_transcript(text) + "\n")
    print(f"wrote transcript -> {a.out}")


if __name__ == "__main__":
    main()
