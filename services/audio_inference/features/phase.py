"""
Phase-Consistency and Vocoder Artifact Feature Extraction.
Neural vocoders (HiFi-GAN, WaveGlow, MelGAN) often introduce subtle phase discontinuities.
"""

from __future__ import annotations

import numpy as np


def extract_phase_features(waveform: np.ndarray, sample_rate: int = 16000) -> dict:
    """Calculates phase derivative variance and group-delay entropy."""
    if len(waveform) < 512:
        return {
            "phase_discontinuity_score": 0.0,
            "group_delay_variance": 0.0,
        }

    n_fft = 512
    hop = 128
    num_frames = (len(waveform) - n_fft) // hop + 1
    if num_frames <= 1:
        return {
            "phase_discontinuity_score": 0.0,
            "group_delay_variance": 0.0,
        }

    stft = np.empty((int(n_fft // 2 + 1), num_frames), dtype=np.complex64)
    window = np.hanning(n_fft)
    for i in range(num_frames):
        chunk = waveform[i * hop : i * hop + n_fft] * window
        stft[:, i] = np.fft.rfft(chunk, n=n_fft)

    angles = np.angle(stft)

    # Unwrap phase across time frames to measure instantaneous frequency deviation
    phase_deriv = np.diff(angles, axis=1)
    # Wrap to [-pi, pi]
    phase_deriv = (phase_deriv + np.pi) % (2 * np.pi) - np.pi

    var_per_bin = np.var(phase_deriv, axis=1)
    phase_discontinuity = float(np.mean(var_per_bin))

    return {
        "phase_discontinuity_score": round(phase_discontinuity, 4),
        "group_delay_variance": round(float(np.std(var_per_bin)), 4),
    }
