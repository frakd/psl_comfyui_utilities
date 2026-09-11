"""Unit tests for the harshness suppressor core, independent of ComfyUI.

    python tests/test_dsp.py
    pytest tests
"""

import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dsp
from dsp import HOP_LENGTH, N_FFT, suppress_harshness

SR = 44100


def _bandpass(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    spec = torch.fft.rfft(x)
    freqs = torch.fft.rfftfreq(x.numel(), 1.0 / SR)
    spec[(freqs < low) | (freqs > high)] = 0
    return torch.fft.irfft(spec, n=x.numel())


def _tone(freq: float, length: int, amplitude: float = 0.2) -> torch.Tensor:
    t = torch.arange(length) / SR
    return amplitude * torch.sin(2 * math.pi * freq * t)


def _energy_db(x: torch.Tensor, windows, low: float, high: float) -> float:
    """Mean energy in dB inside `low`..`high` Hz, averaged over time windows."""
    total = 0.0
    for start, end in windows:
        seg = x[int(start * SR):int(end * SR)]
        spec = torch.fft.rfft(seg * torch.hann_window(seg.numel()))
        freqs = torch.fft.rfftfreq(seg.numel(), 1.0 / SR)
        total += spec[(freqs >= low) & (freqs <= high)].abs().square().sum().item()
    return 10 * math.log10(total / len(windows) + 1e-30)


def _speech(seconds: float = 6.0, seed: int = 0):
    """Voiced body with a steep rolloff plus periodic /s/ bursts.

    Returns the signal and the (start, end) windows of the bursts.
    """
    g = torch.Generator().manual_seed(seed)
    n = int(seconds * SR)
    freqs = torch.fft.rfftfreq(n, 1.0 / SR)

    body = torch.fft.irfft(torch.fft.rfft(torch.randn(n, generator=g)) / (1.0 + (freqs / 700.0) ** 2), n=n)
    body = body / body.abs().max() * 0.5
    t = torch.arange(n) / SR
    body = body * (0.5 + 0.5 * torch.sin(2 * math.pi * 4.0 * t)) ** 2

    bursts = [(round(s, 2), round(s + 0.12, 2)) for s in torch.arange(0.4, seconds - 0.3, 0.8).tolist()]
    sibilant = torch.zeros(n)
    for start, end in bursts:
        i0, i1 = int(start * SR), int(end * SR)
        sibilant[i0:i1] = _bandpass(torch.randn(i1 - i0, generator=g), 5000.0, 9000.0)
        body[i0:i1] *= 0.35
    sibilant = sibilant / sibilant.abs().max() * 0.12
    return body + sibilant, bursts


def _gaps(seconds: float, bursts):
    edges = [0.15] + [t for b in bursts for t in b] + [seconds - 0.15]
    return [(edges[i], edges[i + 1]) for i in range(0, len(edges) - 1, 2)]


def test_shape_dtype_device_are_preserved():
    for shape in [(1, 1, SR), (2, 2, SR * 2), (3, 1, 7777)]:
        x = torch.randn(*shape) * 0.1
        y = suppress_harshness(x, SR)
        assert y.shape == x.shape, (y.shape, x.shape)
        assert y.dtype == x.dtype and y.device == x.device
        assert torch.isfinite(y).all()


def test_harsh_burst_is_attenuated_without_touching_the_rest():
    length = SR * 2
    body = _tone(300.0, length, 0.2)
    burst = torch.zeros(length)
    burst[SR * 8 // 10:SR * 12 // 10] = _tone(6500.0, SR * 4 // 10, 0.12)
    dry = body + burst
    y = suppress_harshness(dry.reshape(1, 1, length), SR)[0, 0]

    hit = [(0.85, 1.15)]
    quiet = [(0.1, 0.7)]
    assert _energy_db(y, hit, 5000, 8000) - _energy_db(dry, hit, 5000, 8000) <= -5.0
    assert abs(_energy_db(y, quiet, 5000, 8000) - _energy_db(dry, quiet, 5000, 8000)) < 0.5
    assert abs(_energy_db(y, hit, 200, 400) - _energy_db(dry, hit, 200, 400)) < 0.3


def test_band_selection_is_respected():
    length = SR * 2
    low_burst = torch.zeros(length)
    low_burst[SR:SR * 3 // 2] = _tone(2700.0, SR // 2, 0.12)
    high_burst = torch.zeros(length)
    high_burst[SR:SR * 3 // 2] = _tone(6500.0, SR // 2, 0.12)
    dry = _tone(120.0, length, 0.2) + low_burst + high_burst

    narrow = suppress_harshness(dry.reshape(1, 1, length), SR, harsh_low_hz=2500.0, harsh_high_hz=3000.0)[0, 0]
    window = [(1.05, 1.45)]
    assert _energy_db(narrow, window, 2600, 2800) - _energy_db(dry, window, 2600, 2800) <= -5.0
    assert abs(_energy_db(narrow, window, 6000, 7000) - _energy_db(dry, window, 6000, 7000)) < 0.2


def test_sibilants_are_reduced_but_vowels_are_left_alone():
    signal, bursts = _speech()
    y = suppress_harshness(signal.reshape(1, 1, -1), SR)[0, 0]
    vowels = _gaps(6.0, bursts)

    burst_cut = _energy_db(y, bursts, 5000, 9000) - _energy_db(signal, bursts, 5000, 9000)
    vowel_cut = _energy_db(y, vowels, 5000, 9000) - _energy_db(signal, vowels, 5000, 9000)
    print(f"  sibilant cut {burst_cut:.2f} dB, vowel cut {vowel_cut:.2f} dB")
    assert burst_cut <= -4.0, burst_cut
    assert vowel_cut >= -1.5, vowel_cut


def test_zero_reduction_is_transparent():
    x = torch.randn(2, 2, SR) * 0.1
    y = suppress_harshness(x, SR, max_reduction_db=0.0)
    assert torch.allclose(y, x, atol=1e-5)


def test_zero_mix_passes_the_input_through():
    x = torch.randn(1, 2, SR) * 0.1
    y = suppress_harshness(x, SR, mix=0.0)
    assert torch.equal(y, x)


def test_silence_stays_silent():
    x = torch.zeros(1, 2, SR)
    y = suppress_harshness(x, SR)
    assert torch.isfinite(y).all()
    assert y.abs().max() == 0.0


def test_very_short_clips():
    for length in (1, 100, 1024, 1025, N_FFT, N_FFT + 1, 5000):
        x = torch.randn(1, 1, length) * 0.1
        y = suppress_harshness(x, SR)
        assert y.shape == x.shape, length
        assert torch.isfinite(y).all(), length


def test_degenerate_band_passes_audio_through():
    x = torch.randn(1, 1, SR) * 0.1
    # An inverted band selects no bin; the node must not fail the workflow.
    assert torch.equal(suppress_harshness(x, SR, harsh_low_hz=3000.0, harsh_high_hz=1000.0), x)


def test_channels_share_one_gain_curve():
    length = SR * 2
    body = _tone(300.0, length, 0.2)
    burst = torch.zeros(length)
    burst[SR:SR * 3 // 2] = _tone(6500.0, SR // 2, 0.12)
    left = body + burst
    right = body + burst * 0.4
    x = torch.stack([left, right]).unsqueeze(0)

    y = suppress_harshness(x, SR)
    window = [(1.05, 1.45)]
    left_cut = _energy_db(y[0, 0], window, 5000, 8000) - _energy_db(left, window, 5000, 8000)
    right_cut = _energy_db(y[0, 1], window, 5000, 8000) - _energy_db(right, window, 5000, 8000)
    print(f"  left cut {left_cut:.2f} dB, right cut {right_cut:.2f} dB")
    assert abs(left_cut - right_cut) < 0.5, (left_cut, right_cut)
    assert left_cut <= -5.0


def test_smoothing_follows_the_attack_and_release_times():
    frames_per_ms = SR / HOP_LENGTH / 1000.0
    step = torch.zeros(1, 400)
    step[0, 100:200] = 1.0

    attack_ms, release_ms = 5.0, 120.0
    y = dsp._smooth(step, attack_ms, release_ms, SR)
    rise = int((y[0, 100:200] < 0.9).sum().item())
    decay = int((y[0, 200:] > 0.1).sum().item())
    print(f"  rise to 90% after {rise} frames ({rise / frames_per_ms:.1f} ms), "
          f"decay to 10% after {decay} frames ({decay / frames_per_ms:.1f} ms)")

    assert rise <= 4 * attack_ms * frames_per_ms
    assert 0.5 * release_ms * frames_per_ms <= decay <= 4 * release_ms * frames_per_ms
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_smoothing_matches_the_sequential_definition():
    drive = torch.rand(3, 500)
    attack_ms, release_ms = 5.0, 120.0
    attack = dsp._coeff(attack_ms, SR)
    release = dsp._coeff(release_ms, SR)

    expected = torch.empty_like(drive)
    previous = torch.zeros(3)
    for frame in range(drive.shape[1]):
        target = drive[:, frame]
        coeff = torch.where(target > previous, attack, release)
        previous = target + (previous - target) * coeff
        expected[:, frame] = previous

    assert torch.allclose(dsp._smooth(drive, attack_ms, release_ms, SR), expected, atol=1e-6)


def test_batching_is_independent():
    signal, _ = _speech(seconds=2.0)
    quiet = _tone(300.0, signal.numel(), 0.2)
    x = torch.stack([signal, quiet]).unsqueeze(1)
    y = suppress_harshness(x, SR)
    assert torch.allclose(y[1, 0], quiet, atol=1e-4)


def test_long_audio_cost():
    x = torch.randn(1, 2, 60 * SR) * 0.1
    start = time.perf_counter()
    suppress_harshness(x, SR)
    audio_seconds = time.perf_counter() - start

    frames = int(300 * SR / HOP_LENGTH)
    drive = torch.rand(1, frames)
    start = time.perf_counter()
    dsp._smooth(drive, 5.0, 120.0, SR)
    smoother_seconds = time.perf_counter() - start
    print(f"  60 s stereo: {audio_seconds:.2f} s; smoother alone on 5 min: {smoother_seconds:.2f} s")

    assert audio_seconds < 30.0
    assert smoother_seconds < 2.0


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
