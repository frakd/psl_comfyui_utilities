"""Harmonic-series notch filtering (de-hum / de-buzz).

A periodic low "哒哒" / buzz / mains hum is not a percussive transient: it is a
tonal fundamental plus its integer harmonics (e.g. 54.7 Hz -> 109, 211, 312 Hz).
Harmonic/percussive separation cannot grab it because a *regularly repeating*
tone looks harmonic (stationary across time) to HPSS — measured on a real track,
HPSS percussive attenuation moved the pulse signature by only ~15% while dulling
the whole mix by 3 dB. A bank of narrow biquad notches placed on the harmonic
series removes the tone where it lives and leaves everything between the notches
untouched.

Each notch is a standard 2nd-order IIR (scipy.signal.iirnotch) with a full zero at
its centre. `reduction_db` sets the effective depth by blending the fully notched
signal back with the dry one, so 30 dB leaves ~3% of the tone, 12 dB leaves ~25%.
`q` sets the width (bandwidth = f / q); higher q is narrower and spares more of the
wanted material around each harmonic.

scipy ships with ComfyUI, so this adds no new dependency. Filtering runs in
float64 on CPU for biquad precision, then the result is returned on the input's
original dtype and device.
"""

import logging

import numpy as np
import torch
from scipy.signal import iirnotch, sosfilt

log = logging.getLogger(__name__)


def suppress_harmonics(
    waveform: torch.Tensor,
    sample_rate: int,
    fundamental_hz: float = 50.0,
    harmonics: int = 8,
    q: float = 15.0,
    reduction_db: float = 30.0,
    mix: float = 1.0,
) -> torch.Tensor:
    """Notch `fundamental_hz` and its integer harmonics across the whole track.

    `waveform` is [batch, channels, samples]; the result keeps the same shape,
    dtype and device. All channels share one filter bank so the stereo image is
    preserved.
    """
    if waveform.dim() != 3:
        raise ValueError(
            f"suppress_harmonics: expected audio of shape [batch, channels, samples], got {tuple(waveform.shape)}"
        )
    if sample_rate <= 0:
        raise ValueError(f"suppress_harmonics: sample rate must be positive, got {sample_rate}")

    nyquist = sample_rate / 2.0
    if fundamental_hz <= 0.0 or fundamental_hz >= nyquist:
        log.warning(
            "harmonic notch fundamental %.3f Hz is outside (0, %.0f) Hz, passing audio through",
            fundamental_hz,
            nyquist,
        )
        return waveform
    if reduction_db <= 0.0 or mix <= 0.0:
        return waveform
    if q <= 0.0:
        raise ValueError(f"suppress_harmonics: q must be positive, got {q}")
    if harmonics < 1:
        raise ValueError(f"suppress_harmonics: harmonics must be >= 1, got {harmonics}")

    freqs = [fundamental_hz * k for k in range(1, int(harmonics) + 1) if fundamental_hz * k < nyquist]
    if not freqs:
        return waveform

    # Each iirnotch is one second-order section; cascade them for the whole series.
    sections = []
    for freq in freqs:
        b, a = iirnotch(freq, q, fs=sample_rate)
        sections.append(np.concatenate([b, a]))
    sos = np.asarray(sections, dtype=np.float64)

    batch, channels, length = waveform.shape
    if length < 4:
        log.warning("harmonic notch: clip of %d samples is too short to filter, passing through", length)
        return waveform

    dry = waveform.detach().to(device="cpu", dtype=torch.float64).numpy().reshape(batch * channels, length)
    notched = sosfilt(sos, dry, axis=-1)

    depth = 1.0 - 10.0 ** (-reduction_db / 20.0)
    wet = dry + (notched - dry) * (depth * mix)

    out = torch.from_numpy(wet.reshape(batch, channels, length)).to(device=waveform.device, dtype=waveform.dtype)
    log.debug(
        "harmonic notch: %d notches at %s Hz, q=%.1f depth=%.1f dB mix=%.2f",
        len(freqs),
        ", ".join(f"{f:.1f}" for f in freqs),
        q,
        reduction_db,
        mix,
    )
    return out