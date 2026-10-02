"""Polarity-aware mono mixdown; the desktop and web apps must keep the same rule and thresholds.

A source whose two channels are polarity-inverted (one is the negative of the other) cancels to near silence when it
is mixed to mono, and a lecture can be inverted in only part of it. This module finds the inverted SECTIONS second by
second, builds the ffmpeg filter that flips the right channel inside them, and checks that the final mix keeps the
voice. Pure code: statistics come in as numpy arrays; audio_utils.to_mono16k runs ffmpeg.
"""
from dataclasses import dataclass

import numpy as np

ANALYSIS_RATE = 8000        # Hz: enough for speech level and phase
BLOCK_SEC = 0.05            # seam search block
SIGNAL_DB = -50.0           # a second has signal when its louder channel is at least this loud
FLIP_MARGIN_DB = 6.0        # inverted when (L-R)/2 is at least this much louder than (L+R)/2
MERGE_NORMAL_SEC = 3        # a normal run shorter than this between two inverted runs joins them
MIN_SECTION_SEC = 5         # an inverted run shorter than this is left alone
SEAM_SEARCH_SEC = 1.0       # a section edge moves to the quietest block within +- this
RAMP_SEC = 0.02             # the flip ramps linearly over this, centred on the seam
CHECK_DROP_DB = 10.0        # output check: the mix may not be this far below the louder channel ...
CHECK_MIN_SEC = 5           # ... for this many consecutive seconds with signal


def db(energy) -> np.ndarray:
    return 10 * np.log10(np.asarray(energy, dtype=np.float64) + 1e-12)


@dataclass
class Stats:
    """Per-second mean energies of L, R, (L+R)/2, (L-R)/2, and per-BLOCK_SEC mean of (L^2 + R^2) / 2."""
    left: np.ndarray
    right: np.ndarray
    mix: np.ndarray
    diff: np.ndarray
    blocks: np.ndarray


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index ranges where flags is true."""
    out, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i))
            start = None
    return out


def _labels(s: Stats) -> np.ndarray:
    """Per second: 1 inverted, 0 normal, -1 no signal."""
    inverted = db(s.diff) >= db(s.mix) + FLIP_MARGIN_DB
    loud = np.maximum(db(s.left), db(s.right)) >= SIGNAL_DB
    return np.where(loud, inverted.astype(int), -1)


def _fill_silence(lab: np.ndarray) -> np.ndarray:
    """Each -1 takes the label of the nearest labelled second; two different nearest labels, or none at all, give 0."""
    known = np.flatnonzero(lab >= 0)
    if known.size == 0:
        return np.zeros_like(lab)
    out = lab.copy()
    for i in np.flatnonzero(lab < 0):
        j = int(np.searchsorted(known, i))
        left = int(known[j - 1]) if j > 0 else None
        right = int(known[j]) if j < known.size else None
        dl = i - left if left is not None else None
        dr = right - i if right is not None else None
        if dr is None or (dl is not None and dl < dr):
            out[i] = lab[left]
        elif dl is None or dr < dl:
            out[i] = lab[right]
        else:
            out[i] = lab[left] if lab[left] == lab[right] else 0
    return out


def _smooth(lab: np.ndarray) -> np.ndarray:
    out = lab.copy()
    inv = _runs(out == 1)
    for (_, a_end), (b_start, _) in zip(inv, inv[1:]):
        if b_start - a_end < MERGE_NORMAL_SEC:
            out[a_end:b_start] = 1
    for start, end in _runs(out == 1):
        if end - start < MIN_SECTION_SEC:
            out[start:end] = 0
    return out


def _seam(second: int, blocks: np.ndarray) -> float:
    """Centre, in seconds, of the quietest block within +- SEAM_SEARCH_SEC of a section edge."""
    per_sec = round(1 / BLOCK_SEC)
    lo = max(0, int(round((second - SEAM_SEARCH_SEC) * per_sec)))
    hi = min(len(blocks), int(round((second + SEAM_SEARCH_SEC) * per_sec)))
    if hi <= lo:
        return float(second)
    k = lo + int(np.argmin(blocks[lo:hi]))
    return round((k + 0.5) * BLOCK_SEC, 3)


def find_sections(s: Stats) -> list[tuple[float, float | None]]:
    """Sections to flip: (start, end) in seconds; start 0.0 = from the beginning, end None = to the end."""
    lab = _smooth(_fill_silence(_labels(s)))
    n = len(lab)
    return [(0.0 if a == 0 else _seam(a, s.blocks), None if b >= n else _seam(b, s.blocks))
            for a, b in _runs(lab == 1)]


def flip_filter(sections: list[tuple[float, float | None]]) -> str:
    """Filter for a named stereo stream: the right channel times -1 inside the sections, linear RAMP_SEC ramps at
    inner edges; pan re-names the layout (aeval leaves an unnamed "2 channels" layout the encoders reject). The ramps
    use n/s (samples counted from the first sample, the timeline the statistics use), never t, which starts at the
    stream's start time. n restarts at 0 when ffmpeg rebuilds the filter graph on a mid-stream sample-rate,
    sample-format or layout change, which delays every later flip by the time of the rebuild; the output check
    refuses the file once that cancels CHECK_MIN_SEC or more signal seconds."""
    h = RAMP_SEC / 2
    terms = []
    for start, end in sections:
        rise = "1" if start <= 0 else f"clip((n/s-({start - h:.3f}))/{RAMP_SEC},0,1)"
        fall = "1" if end is None else f"clip((({end + h:.3f})-n/s)/{RAMP_SEC},0,1)"
        terms.append(f"{rise}*{fall}")
    gain = "1-2*(" + "+".join(terms) + ")"
    return f"aeval=exprs='val(0)|val(1)*({gain})',pan=stereo|c0=c0|c1=c1"


def cancelled_stretches(mix_db: np.ndarray, louder_db: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) seconds where CHECK_MIN_SEC or more consecutive signal seconds have a mix CHECK_DROP_DB or more
    below the louder channel. Seconds without signal are skipped: they neither count nor break a run."""
    signal = np.flatnonzero(np.asarray(louder_db) >= SIGNAL_DB)
    bad = np.asarray(mix_db)[signal] <= np.asarray(louder_db)[signal] - CHECK_DROP_DB
    return [(int(signal[a]), int(signal[b - 1]) + 1) for a, b in _runs(bad) if b - a >= CHECK_MIN_SEC]


def _hms(sec: float) -> str:
    s = int(sec)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


class AudioCancelled(RuntimeError):
    """The output check failed: the mono mix still cancels the voice in these [start, end) seconds."""

    def __init__(self, stretches: list[tuple[int, int]]):
        self.stretches = list(stretches)
        super().__init__("the mono mix is cancelled in " + ", ".join(f"{_hms(a)}-{_hms(b)}" for a, b in stretches))


def summary(channels: int, sections: list[tuple[float, float | None]], check: str,
            stretches: list[tuple[int, int]] = ()) -> dict:
    """The polarity.json / run_trace.json "polarity" record."""
    return {"channels": channels, "sections": [[a, b] for a, b in sections], "check": check,
            "cancelled": [[a, b] for a, b in stretches]}


class StatsAccumulator:
    """Builds Stats from interleaved s16le stereo PCM at `rate`, fed in chunks of any size (a frame or even a
    sample may be split across chunks). One second at a time is reduced, so a 2-hour file needs no big buffer."""

    def __init__(self, rate: int = ANALYSIS_RATE):
        self.rate = rate
        self.block = int(round(rate * BLOCK_SEC))
        self._rest = b""
        self._frames = np.zeros((0, 2))
        self._sec: list[tuple[float, float, float, float]] = []
        self._blocks: list[float] = []

    def feed(self, pcm: bytes) -> None:
        data = self._rest + pcm
        usable = len(data) // 4 * 4
        self._rest = data[usable:]
        if usable:
            x = np.frombuffer(data[:usable], dtype="<i2").astype(np.float64).reshape(-1, 2) / 32768.0
            self._frames = np.concatenate([self._frames, x])
        while len(self._frames) >= self.rate:
            self._reduce(self._frames[:self.rate])
            self._frames = self._frames[self.rate:]

    def _reduce(self, x: np.ndarray) -> None:
        left, right = x[:, 0], x[:, 1]
        self._sec.append((float(np.mean(left ** 2)), float(np.mean(right ** 2)),
                          float(np.mean(((left + right) / 2) ** 2)), float(np.mean(((left - right) / 2) ** 2))))
        for i in range(0, len(x), self.block):
            b = x[i:i + self.block]
            self._blocks.append(float(np.mean(b ** 2)))

    def finish(self) -> Stats:
        if len(self._frames):
            self._reduce(self._frames)
            self._frames = np.zeros((0, 2))
        a = np.array(self._sec, dtype=np.float64).reshape(-1, 4)
        return Stats(left=a[:, 0], right=a[:, 1], mix=a[:, 2], diff=a[:, 3], blocks=np.array(self._blocks))
