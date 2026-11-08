"""Unit tests for the reference-gated de-percuss core, independent of ComfyUI.

    python tests/test_gated_deperc.py
    pytest tests
"""

import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gated_deperc import reference_gated_deperc

SR = 32000


def _tone(freq: float, seconds: float, amplitude: float = 0.3) -> torch.Tensor:
    t = torch.arange(int(seconds * SR)) / SR
    return amplitude * torch.sin(2 * math.pi * freq * t)


def _audio(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0).unsqueeze(0)


def _scene(seconds: float = 5.0, burst=(2.0, 3.5), freq: float = 2500.0):
    """Low-level noise background with a distinctive tone burst in `burst`."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(int(seconds * SR), generator=g) * 0.01
    if burst is not None:
        a, b = burst
        x[int(a * SR):int(b * SR)] += _tone(freq, b - a)
    return x


def test_shape_dtype_device_preserved():
    x = _audio(_scene())
    ref = _audio(_tone(2500.0, 1.0))
    y = reference_gated_deperc(x, SR, ref, SR)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.device == x.device


def test_detects_burst_and_preserves_outside():
    x = _audio(_scene(burst=(2.0, 3.5)))
    ref = _audio(_tone(2500.0, 1.0))
    y = reference_gated_deperc(x, SR, ref, SR, match_threshold=0.9, min_region_sec=0.3, crossfade_ms=20.0)
    xi, yi = x[0, 0], y[0, 0]
    # Far outside the burst: bit-for-bit unchanged.
    assert torch.equal(yi[:int(1.5 * SR)], xi[:int(1.5 * SR)])
    assert torch.equal(yi[int(4.2 * SR):], xi[int(4.2 * SR):])
    # Inside the burst: processing changed something.
    assert not torch.equal(yi[int(2.2 * SR):int(3.3 * SR)], xi[int(2.2 * SR):int(3.3 * SR)])


def test_no_match_passes_through():
    x = _audio(_scene(burst=None))  # noise only, no tone
    ref = _audio(_tone(2500.0, 1.0))
    y = reference_gated_deperc(x, SR, ref, SR, match_threshold=0.95)
    assert torch.equal(y, x)


def test_zero_reduction_transparent():
    x = _audio(_scene())
    ref = _audio(_tone(2500.0, 1.0))
    assert torch.equal(reference_gated_deperc(x, SR, ref, SR, reduction_db=0.0), x)


def test_zero_mix_transparent():
    x = _audio(_scene())
    ref = _audio(_tone(2500.0, 1.0))
    assert torch.equal(reference_gated_deperc(x, SR, ref, SR, mix=0.0), x)


def test_silent_reference_raises():
    x = _audio(_scene())
    ref = torch.zeros(1, 1, SR)
    try:
        reference_gated_deperc(x, SR, ref, SR)
    except ValueError:
        return
    raise AssertionError("expected ValueError for silent reference")


def test_wrong_shape_raises():
    ref = _audio(_tone(2500.0, 1.0))
    try:
        reference_gated_deperc(torch.randn(SR), SR, ref, SR)
    except ValueError:
        return
    raise AssertionError("expected ValueError for 1-D input")


def test_bad_threshold_raises():
    x = _audio(_scene())
    ref = _audio(_tone(2500.0, 1.0))
    try:
        reference_gated_deperc(x, SR, ref, SR, match_threshold=1.5)
    except ValueError:
        return
    raise AssertionError("expected ValueError for threshold > 1")


def test_stereo_channels_linked():
    mono = _scene(burst=(2.0, 3.5))
    x = torch.stack([mono, mono * 0.5]).unsqueeze(0)
    ref = _audio(_tone(2500.0, 1.0))
    y = reference_gated_deperc(x, SR, ref, SR, match_threshold=0.9, min_region_sec=0.3)
    # Outside the burst both channels untouched.
    assert torch.equal(y[0, 0, :int(1.5 * SR)], x[0, 0, :int(1.5 * SR)])
    assert torch.equal(y[0, 1, :int(1.5 * SR)], x[0, 1, :int(1.5 * SR)])


def test_reference_resampled():
    x = _audio(_scene(burst=(2.0, 3.5)))
    # Reference at a different sample rate must still match after resampling.
    ref_sr = 16000
    t = torch.arange(ref_sr) / ref_sr
    ref = _audio(0.3 * torch.sin(2 * math.pi * 2500.0 * t))
    y = reference_gated_deperc(x, SR, ref, ref_sr, match_threshold=0.9, min_region_sec=0.3, crossfade_ms=20.0)
    assert torch.equal(y[0, 0, :int(1.5 * SR)], x[0, 0, :int(1.5 * SR)])
    assert not torch.equal(y[0, 0, int(2.2 * SR):int(3.3 * SR)], x[0, 0, int(2.2 * SR):int(3.3 * SR)])


def test_cost():
    x = torch.randn(1, 2, 30 * SR) * 0.05
    ref = _audio(_tone(2500.0, 1.0))
    start = time.perf_counter()
    reference_gated_deperc(x, SR, ref, SR)
    elapsed = time.perf_counter() - start
    print(f"  30 s stereo: {elapsed:.2f} s")
    assert elapsed < 30.0


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