"""
Linear Frequency Cepstral Coefficients (LFCC) Feature Extraction.
Standard acoustic feature used in ASVspoof baselines for logical access / synthetic speech detection.
"""

from __future__ import annotations

import numpy as np
from scipy.fftpack import dct


def extract_lfcc(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_filter: int = 70,
    n_lfcc: int = 60,
    win_len: float = 0.025,
    hop_len: float = 0.010,
) -> np.ndarray:
    """Extracts static, delta, and delta-delta LFCC features."""
    if len(waveform) == 0:
        return np.zeros((1, n_lfcc * 3))

    frame_len = int(win_len * sample_rate)
    frame_step = int(hop_len * sample_rate)
    signal_len = len(waveform)

    # Frame signal
    num_frames = 1 + int(math_ceil_ratio(signal_len - frame_len, frame_step))
    pad_signal_len = (num_frames - 1) * frame_step + frame_len
    z = np.zeros(pad_signal_len - signal_len)
    pad_signal = np.append(waveform, z)

    indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)
    ).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Hamming window & FFT
    frames *= np.hamming(frame_len)
    nfft = 512
    mag_frames = np.absolute(np.fft.rfft(frames, nfft))
    pow_frames = (1.0 / nfft) * (mag_frames**2)

    # Linear filterbank
    low_freq = 0
    high_freq = sample_rate / 2
    linear_points = np.linspace(low_freq, high_freq, n_filter + 2)
    bin_points = np.floor((nfft + 1) * linear_points / sample_rate)

    fbank = np.zeros((n_filter, int(np.floor(nfft / 2 + 1))))
    for m in range(1, n_filter + 1):
        f_m_minus = int(bin_points[m - 1])
        f_m = int(bin_points[m])
        f_m_plus = int(bin_points[m + 1])
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin_points[m - 1]) / max(bin_points[m] - bin_points[m - 1], 1)
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin_points[m + 1] - k) / max(bin_points[m + 1] - bin_points[m], 1)

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)

    # DCT to obtain LFCCs
    raw_lfcc = dct(filter_banks, type=2, axis=1, norm="ortho")[:, :n_lfcc]

    # Delta and delta-delta
    delta = _compute_delta(raw_lfcc)
    delta2 = _compute_delta(delta)

    return np.hstack((raw_lfcc, delta, delta2))


def math_ceil_ratio(n: int, d: int) -> int:
    return max(0, int(np.ceil(n / d)))


def _compute_delta(feat: np.ndarray, N: int = 2) -> np.ndarray:
    if feat.shape[0] <= 1:
        return np.zeros_like(feat)
    pad = np.pad(feat, ((N, N), (0, 0)), mode="edge")
    delta = np.zeros_like(feat)
    denom = 2 * sum([i**2 for i in range(1, N + 1)])
    for n in range(1, N + 1):
        delta += n * (pad[N + n : N + n + feat.shape[0], :] - pad[N - n : N - n + feat.shape[0], :])
    return delta / denom
