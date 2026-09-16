"""Pair .srt <-> .mp3 by normalized title. The normalized key is used ONLY internally to pair
the two filename sets (does not need to exactly match the BE logic)."""
import argparse
import os
import re
import unicodedata


def normalize_key(title: str) -> str:
    """Normalize title: NFD, strip marks, lowercase, đ->d, collapse spaces."""
    s = unicodedata.normalize("NFD", title)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower().replace("đ", "d")
    return re.sub(r"\s+", " ", s).strip()


def index_audio(audio_dirs: list[str]) -> dict[str, str]:
    """Index .mp3 files from a list of directories: {normalized_key -> mp3_path}."""
    idx: dict[str, str] = {}
    for d in audio_dirs:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.lower().endswith(".mp3"):
                continue
            key = normalize_key(os.path.splitext(name)[0])
            idx.setdefault(key, os.path.join(d, name))  # keep the first file on key collision
    return idx


def build_pairs(src_dir: str, audio_dirs: list[str], ext: str = ".srt") -> tuple[list[dict], list[dict], list[dict]]:
    """Pair transcript (.srt or .txt) <-> .mp3.

    Args:
        src_dir: directory containing transcripts (.srt or .txt)
        audio_dirs: list of directories containing .mp3
        ext: source extension to pair (".srt" or ".txt")

    Returns:
        (pairs, unpaired_srt, unpaired_audio) where:
        - pairs: [{"srt_path", "srt_name", "kind", "audio_path", "key"}, ...] (kind = "srt"|"txt")
        - unpaired_srt: [{"srt_name", "srt_path", "key"}, ...]
        - unpaired_audio: [{"audio_name", "audio_path", "key"}, ...]
    """
    if not os.path.isdir(src_dir):
        raise ValueError(f"src_dir not found or not a directory: {src_dir!r}")
    audio_idx = index_audio(audio_dirs)
    kind = ext.lstrip(".").lower()
    used: set[str] = set()
    pairs, unpaired_srt = [], []
    for name in sorted(os.listdir(src_dir)):
        if not name.lower().endswith(ext.lower()):
            continue
        key = normalize_key(os.path.splitext(name)[0])
        src_path = os.path.join(src_dir, name)
        if key in audio_idx:
            used.add(key)
            pairs.append({"srt_path": src_path, "srt_name": name, "kind": kind,
                          "audio_path": audio_idx[key], "key": key})
        else:
            unpaired_srt.append({"srt_name": name, "srt_path": src_path, "key": key})
    unpaired_audio = [{"audio_name": os.path.basename(p), "audio_path": p, "key": k}
                      for k, p in audio_idx.items() if k not in used]
    return pairs, unpaired_srt, unpaired_audio


def format_unpaired_report(unpaired_srt: list[dict], unpaired_audio: list[dict]) -> str:
    """Format the report of leftover files (not yet paired)."""
    out = ["# Leftover files (not yet paired)", "",
           f"## SRT without audio ({len(unpaired_srt)})"]
    out += [f"- {u['srt_name']}  (key: {u['key']})" for u in unpaired_srt] or ["- (none)"]
    out += ["", f"## Audio without srt ({len(unpaired_audio)})"]
    out += [f"- {u['audio_name']}  (key: {u['key']})" for u in unpaired_audio] or ["- (none)"]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srt-dir", required=True)
    ap.add_argument("--audio-dir", action="append", required=True)
    ap.add_argument("--report")
    a = ap.parse_args()
    pairs, us, ua = build_pairs(a.srt_dir, a.audio_dir)
    print(f"pairs={len(pairs)} unpaired_srt={len(us)} unpaired_audio={len(ua)}")
    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            f.write(format_unpaired_report(us, ua))
        print(f"report -> {a.report}")


if __name__ == "__main__":
    main()
