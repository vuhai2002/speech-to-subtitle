"""Build cues from word-level data per production rules (Netflix VN / BBC / ESIST).
Breaks: hard-break at sentence end / long pause (>= LONG_PAUSE); a medium pause (>= PAUSE_SPLIT) is a
preferred break point in _fill_k (no hard split). Enforce CPS reading speed. Merge tiny cues.
Merge boundary cues of 1-2 misaligned words (gap >= ISOLATION_GAP) into the neighbor."""
import realign.config as config
from realign.text_metrics import char_count, wrap_two_lines


def fill_word_times(words: list[dict]) -> list[dict]:
    words = [dict(w) for w in words]
    n = len(words)
    for i, w in enumerate(words):
        if w.get("start") is None or w.get("end") is None:
            prev_e = next((words[j]["end"] for j in range(i - 1, -1, -1)
                           if words[j].get("end") is not None), 0.0)
            nxt_s = next((words[j]["start"] for j in range(i + 1, n)
                          if words[j].get("start") is not None), prev_e + 0.3)
            w["start"] = prev_e
            w["end"] = max(prev_e + 0.05, nxt_s)
    return words


def _make_cue(ws: list[dict]) -> dict:
    return {"text": " ".join(w["w"] for w in ws),
            "start": ws[0]["start"], "end": ws[-1]["end"]}


def _fits_lines(text: str, cfg) -> bool:
    """True if text word-wraps into <= LINES_MAX lines, each line <= CPL_MAX
    (greedy word-wrap, no splitting a word). Replaces the old CHAR_MAX threshold."""
    lines, cur = 1, 0
    for word in text.split(" "):
        wlen = char_count(word)
        if wlen > cfg.CPL_MAX:
            return False                       # a single word already longer than one line
        if cur == 0:
            cur = wlen
        elif cur + 1 + wlen <= cfg.CPL_MAX:
            cur += 1 + wlen
        else:
            lines += 1
            cur = wlen
            if lines > cfg.LINES_MAX:
                return False
    return lines <= cfg.LINES_MAX


def _join(words: list[dict]) -> str:
    return " ".join(w["w"] for w in words)


def _split_segments(words: list[dict], cfg) -> list[list[dict]]:
    """Split words into segments at hard-breaks: sentence end, or a LONG pause >= LONG_PAUSE.
    (A medium pause does not split here - it is a preferred break point in _fill_k.)"""
    ends = tuple(cfg.SENTENCE_END)
    segs, cur = [], []
    for w in words:
        if cur:
            prev = cur[-1]
            if prev["w"].endswith(ends) or (w["start"] - prev["end"]) >= cfg.LONG_PAUSE:
                segs.append(cur)
                cur = []
        cur.append(w)
    if cur:
        segs.append(cur)
    return segs


def _fill_k(words: list[dict], k: int, cfg) -> list[list[dict]]:
    """Split words into at most k balanced groups (~total/k chars), preferring breaks after
    comma/`;`/`:` OR at a medium pause (>= PAUSE_SPLIT) when already near target."""
    target = char_count(_join(words)) / k
    groups, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        if (k - len(groups)) <= 1:
            continue
        if (len(words) - i - 1) <= (k - len(groups) - 1):
            groups.append(cur)
            cur = []
            continue
        acc = char_count(_join(cur))
        ends_soft = w["w"].endswith((",", ";", ":"))
        pause_after = (words[i + 1]["start"] - w["end"]) >= cfg.PAUSE_SPLIT
        if (acc >= target * 0.6 and (ends_soft or pause_after)) or acc >= target:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def _segment_to_cues(words: list[dict], cfg) -> list[list[dict]]:
    """1 segment -> word groups, each group <= 2 lines <= CPL. A segment longer than 1 cue
    -> split evenly (minimum cue count) + prefer breaking at commas (avoid stub tails)."""
    if _fits_lines(_join(words), cfg):
        return [words]
    for k in range(2, len(words) + 1):
        groups = _fill_k(words, k, cfg)
        if len(groups) == k and all(_fits_lines(_join(g), cfg) for g in groups):
            return groups
    return [[w] for w in words]                             # rare fallback (a single word too long)


def _enforce_min_display(cues: list[dict], cfg) -> list[dict]:
    out, i = [], 0
    while i < len(cues):
        c = dict(cues[i])
        needed = max(cfg.DUR_MIN, char_count(c["text"]) / cfg.CPS_MAX)
        nxt_start = cues[i + 1]["start"] if i + 1 < len(cues) else None
        cap = c["start"] + cfg.DUR_MAX
        if nxt_start is not None:
            cap = min(cap, nxt_start - cfg.GAP_MIN)
        want_end = min(max(c["end"], c["start"] + needed), cap)
        if (want_end - c["start"]) >= cfg.DUR_MIN - 1e-6:
            c["end"] = want_end
            out.append(c)
            i += 1
            continue
        # not enough room to reach the floor -> merge into the previous cue (preferred) or the next, if chars fit
        if out and _fits_lines(out[-1]["text"] + " " + c["text"], cfg):
            out[-1]["text"] += " " + c["text"]
            out[-1]["end"] = max(out[-1]["end"], c["end"])
            i += 1
            continue
        if i + 1 < len(cues) and _fits_lines(c["text"] + " " + cues[i + 1]["text"], cfg):
            nxt = dict(cues[i + 1])
            nxt["text"] = c["text"] + " " + nxt["text"]
            nxt["start"] = c["start"]
            cues[i + 1] = nxt
            i += 1
            continue
        # cannot merge -> accept it, at least reaching FLOOR
        c["end"] = max(c["end"], want_end, c["start"] + cfg.FLOOR)
        out.append(c)
        i += 1
    return out


def _deisolate_boundaries(cues: list[dict], cfg) -> list[dict]:
    """Merge a first/last cue of only 1-2 words split off by gap >= ISOLATION_GAP (boundary
    misalignment at file start/end) into the neighbor, dropping the off timestamp. Only if result stays <= 2 lines <= CPL."""
    if len(cues) < 2:
        return cues
    cues = [dict(c) for c in cues]
    if (len(cues[0]["text"].split()) <= 2
            and (cues[1]["start"] - cues[0]["end"]) >= cfg.ISOLATION_GAP
            and _fits_lines(cues[0]["text"] + " " + cues[1]["text"], cfg)):
        cues[1]["text"] = cues[0]["text"] + " " + cues[1]["text"]
        cues = cues[1:]
    if (len(cues) >= 2 and len(cues[-1]["text"].split()) <= 2
            and (cues[-1]["start"] - cues[-2]["end"]) >= cfg.ISOLATION_GAP
            and _fits_lines(cues[-2]["text"] + " " + cues[-1]["text"], cfg)):
        cues[-2]["text"] = cues[-2]["text"] + " " + cues[-1]["text"]
        cues = cues[:-1]
    return cues


def build_cues(words: list[dict], cfg=config) -> list[dict]:
    if not words:
        return []
    words = fill_word_times(words)
    raw = []
    for seg in _split_segments(words, cfg):
        for group in _segment_to_cues(seg, cfg):
            raw.append(_make_cue(group))
    cues = _enforce_min_display(raw, cfg)
    cues = _deisolate_boundaries(cues, cfg)
    for c in cues:
        c["text"] = wrap_two_lines(c["text"], cfg.CPL_MAX)
    return cues
