"""Unit tests for realign.window_align pure helpers (fake emission, no GPU/model)."""
import torch

from realign.window_align import _blank_runs, _window_bounds


def _emission(n_frames, blank_frames):
    """Emission [1, n, 3]; frames in blank_frames have argmax=0 (blank), the rest argmax=1."""
    e = torch.zeros(1, n_frames, 3)
    e[0, :, 1] = 1.0
    for f in blank_frames:
        e[0, f, 0] = 2.0
    return e


def test_blank_runs_respects_min_run():
    e = _emission(30, list(range(5, 9)) + list(range(15, 25)))   # run len 4 and 10
    assert _blank_runs(e, min_run=5) == [(15, 25)]               # run 4 (<5) dropped


def test_window_bounds_splits_at_silence_center():
    e = _emission(100, range(40, 60))                            # blank-run [40,60) -> center 50
    fb, wb = _window_bounds(e, n_align=1000, dur_sec=100.0,
                            target_window_sec=50.0, ratio=320.0, sr=16000)
    assert fb == [0, 50, 100]                                    # split at the silence in the middle
    assert wb == [0, 500, 1000]                                  # words split proportionally to frames


def test_window_bounds_k1_when_short():
    e = _emission(100, [])
    fb, wb = _window_bounds(e, n_align=500, dur_sec=100.0,
                            target_window_sec=1500.0, ratio=320.0, sr=16000)
    assert fb == [0, 100] and wb == [0, 500]                     # dur < target -> 1 window
