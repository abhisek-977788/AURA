"""
Mel-Spectrogram Feature Extraction for Speech Forensics.
"""

from __future__ import annotations

import numpy as np


def extract_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> np.ndarray:
    """Extracts normalized log-mel spectrogram array."""
    if len(waveform) < hop_length:
        return np.zeros((n_mels, 1))

    # STFT
    window = np.hanning(n_fft)
    num_frames = (len(waveform) - n_fft) // hop_length + 1
    if num_frames <= 0:
        return np.zeros((n_mels, 1))

    stft_matrix = np.empty((int(n_fft // 2 + 1), num_frames), dtype=np.complex64)
    for i in range(num_frames):
        chunk = waveform[i * hop_length : i * hop_length + n_fft] * window
        stft_matrix[:, i] = np.fft.rfft(chunk, n=n_fft)

    magnitudes = np.abs(stft_matrix) ** 2

    # Mel filterbank
    mel_fb = _create_mel_filterbank(sample_rate, n_fft, n_mels)
    mel_spec = np.dot(mel_fb, magnitudes)
    mel_spec = np.maximum(mel_spec, 1e-5)
    log_mel_spec = 10.0 * np.log10(mel_spec)

    # Normalize to [-1, 1]
    mean = np.mean(log_mel_spec)
    std = np.std(log_mel_spec) + 1e-8
    return np.clip((log_mel_spec - mean) / (2 * std), -1.0, 1.0)


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _create_mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    low_mel = _hz_to_mel(0)
    high_mel = _hz_to_mel(sr / 2)
    mels = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = _mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fb = np.zeros((n_mels, int(n_fft // 2 + 1)))
    for i in range(n_mels):
        for j in range(bins[i], bins[i + 1]):
            fb[i, j] = (j - bins[i]) / max(bins[i + 1] - bins[i], 1)
        for j in range(bins[i + 1], bins[i + 2]):
            fb[i, j] = (bins[i + 2] - j) / max(bins[i + 2] - bins[i + 1], 1)
    return fb
