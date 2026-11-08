"""Unit tests for the harmonic notch core, independent of ComfyUI.

    python tests/test_dehum.py
    pytest tests
"""

import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dehum import suppress_harmonics

SR = 32000
F0 = 54.7


def _tone(freq: float, seconds: float = 1.0, amplitude: float = 0.3) -> torch.Tensor:
    t = torch.arange(int(seconds * SR)) / SR
    return amplitude * torch.sin(2 * math.pi * freq * t)


def _audio(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0).unsqueeze(0)


def _level_db(x: torch.Tensor, freq: float, skip: float = 0.2) -> float:
    """Steady-state amplitude in dB of the component at `freq`, via projection."""
    seg = x[int(skip * SR):]
    t = torch.arange(seg.numel()) / SR
    cos = (seg * torch.cos(2 * math.pi * freq * t)).mean()
    sin = (seg * torch.sin(2 * math.pi * freq * t)).mean()
    return 20 * math.log10(2 * math.hypot(cos.item(), sin.item()) + 1e-12)


def test_shape_dtype_device_preserved():
    x = _audio(_tone(F0)).to(torch.float32)
    y = suppress_harmonics(x, SR, fundamental_hz=F0)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.device == x.device


def test_fundamental_is_notched():
    x = _audio(_tone(F0))
    y = suppress_harmonics(x, SR, fundamental_hz=F0, harmonics=1, q=10.0, reduction_db=40.0)
    drop = _level_db(x[0, 0], F0) - _level_db(y[0, 0], F0)
    assert drop > 25.0, drop


def test_harmonics_are_notched():
    for k in (2, 3, 4):
        x = _audio(_tone(F0 * k))
        y = suppress_harmonics(x, SR, fundamental_hz=F0, harmonics=6, q=10.0, reduction_db=40.0)
        drop = _level_db(x[0, 0], F0 * k) - _level_db(y[0, 0], F0 * k)
        assert drop > 25.0, (k, drop)


def test_between_harmonics_preserved():
    # A tone halfway between two harmonics must survive nearly intact.
    freq = F0 * 2.5
    x = _audio(_tone(freq))
    y = suppress_harmonics(x, SR, fundamental_hz=F0, harmonics=6, q=20.0, reduction_db=40.0)
    drop = _level_db(x[0, 0], freq) - _level_db(y[0, 0], freq)
    assert drop < 1.0, drop


def test_reduction_db_sets_depth():
    x = _audio(_tone(F0))
    shallow = suppress_harmonics(x, SR, fundamental_hz=F0, harmonics=1, q=10.0, reduction_db=12.0)
    deep = suppress_harmonics(x, SR, fundamental_hz=F0, harmonics=1, q=10.0, reduction_db=40.0)
    shallow_drop = _level_db(x[0, 0], F0) - _level_db(shallow[0, 0], F0)
    deep_drop = _level_db(x[0, 0], F0) - _level_db(deep[0, 0], F0)
    assert 8.0 < shallow_drop < 16.0, shallow_drop
    assert deep_drop > shallow_drop + 15.0, (shallow_drop, deep_drop)


def test_zero_reduction_transparent():
    x = _audio(_tone(F0))
    assert torch.equal(suppress_harmonics(x, SR, fundamental_hz=F0, reduction_db=0.0), x)


def test_zero_mix_transparent():
    x = _audio(_tone(F0))
    assert torch.equal(suppress_harmonics(x, SR, fundamental_hz=F0, mix=0.0), x)


def test_silence_stays_silent():
    x = torch.zeros(1, 1, SR)
    y = suppress_harmonics(x, SR, fundamental_hz=F0)
    assert torch.count_nonzero(y) == 0


def test_very_short_clip_passthrough():
    x = torch.randn(1, 1, 2)
    assert torch.equal(suppress_harmonics(x, SR, fundamental_hz=F0), x)


def test_stereo_channels_linked():
    left = _tone(F0)
    right = _tone(F0) * 0.5
    x = torch.stack([left, right]).unsqueeze(0)
    y = suppress_harmonics(x, SR, fundamental_hz=F0, harmonics=1, reduction_db=40.0)
    # Same filter on both channels: the 2:1 amplitude ratio is preserved.
    ratio = y[0, 0].abs().max() / (y[0, 1].abs().max() + 1e-12)
    assert abs(ratio.item() - 2.0) < 0.05, ratio


def test_batching_is_independent():
    buzz = _tone(F0)
    clean = _tone(F0 * 2.5)
    x = torch.stack([buzz, clean]).unsqueeze(1)
    y = suppress_harmonics(x, SR, fundamental_hz=F0, harmonics=6, q=20.0, reduction_db=40.0)
    # The clean item sits between notches: its level survives (IIR phase shift means
    # raw samples differ, so compare amplitude at its frequency, not sample equality).
    drop = _level_db(clean, F0 * 2.5) - _level_db(y[1, 0], F0 * 2.5)
    assert drop < 1.0, drop
    buzz_drop = _level_db(buzz, F0) - _level_db(y[0, 0], F0)
    assert buzz_drop > 25.0, buzz_drop


def test_invalid_fundamental_passthrough():
    x = _audio(_tone(F0))
    assert torch.equal(suppress_harmonics(x, SR, fundamental_hz=0.0), x)
    assert torch.equal(suppress_harmonics(x, SR, fundamental_hz=SR), x)


def test_bad_params_raise():
    x = _audio(_tone(F0))
    for kwargs in ({"q": 0.0}, {"harmonics": 0}):
        try:
            suppress_harmonics(x, SR, fundamental_hz=F0, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


def test_wrong_shape_raises():
    try:
        suppress_harmonics(torch.randn(SR), SR, fundamental_hz=F0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for 1-D input")


def test_long_audio_cost():
    x = torch.randn(1, 2, 60 * SR) * 0.1
    start = time.perf_counter()
    suppress_harmonics(x, SR, fundamental_hz=50.0, harmonics=12)
    elapsed = time.perf_counter() - start
    print(f"  60 s stereo, 12 notches: {elapsed:.2f} s")
    assert elapsed < 15.0


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        start = time.perf_counter()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - report every failure, keep running the rest
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name} ({time.perf_counter() - start:.2f} s)")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)