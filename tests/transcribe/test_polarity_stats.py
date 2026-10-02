"""StatsAccumulator turns interleaved s16le stereo PCM, fed in arbitrary byte chunks, into per-second statistics."""
import numpy as np

from transcribe.chunked_transcribe import polarity

RATE = polarity.ANALYSIS_RATE


def pcm(left: np.ndarray, right: np.ndarray) -> bytes:
    return (np.stack([left, right], axis=1) * 32767).astype("<i2").tobytes()


def test_seconds_blocks_and_odd_chunk_sizes():
    t = np.arange(int(2.5 * RATE)) / RATE
    x = 0.5 * np.sin(2 * np.pi * 300 * t)
    right = np.where(t < 1.0, x, -x)
    right[t >= 2.0] = 0.0
    left = np.where(t >= 2.0, 0.0, x)
    data = pcm(left, right)
    acc = polarity.StatsAccumulator()
    for i in range(0, len(data), 1001):          # odd sizes split frames and samples
        acc.feed(data[i:i + 1001])
    s = acc.finish()
    assert len(s.left) == 3                       # 1 s, 1 s, then the last half second
    assert len(s.blocks) == 50                    # 20 blocks per second
    assert polarity.db(s.mix[0]) > polarity.db(s.diff[0]) + 40     # first second in phase
    assert polarity.db(s.diff[1]) > polarity.db(s.mix[1]) + 40     # second second inverted
    assert polarity.db(s.left[2]) < -100                            # silence
