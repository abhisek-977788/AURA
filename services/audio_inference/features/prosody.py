"""
Prosody and Pitch Dynamics Extraction for Synthetic Voice Forensics.
Measures pitch (F0) contour smoothness, unnatural robotization, and prosodic jitter.
"""

from __future__ import annotations

import numpy as np


def extract_prosody(waveform: np.ndarray, sample_rate: int = 16000) -> dict:
    """Calculates F0 pitch contour, pitch statistics, and speech rate indicators."""
    if len(waveform) < 1024:
        return {
            "f0_mean_hz": 0.0,
            "f0_std_hz": 0.0,
            "f0_range_hz": 0.0,
            "prosodic_jitter": 0.0,
            "energy_entropy": 0.0,
        }

    # Frame-level autocorrelation pitch estimator
    frame_size = int(0.03 * sample_rate)  # 30ms
    hop_size = int(0.015 * sample_rate)   # 15ms
    pitches = []

    min_lag = int(sample_rate / 400)  # Max pitch 400Hz
    max_lag = int(sample_rate / 60)   # Min pitch 60Hz

    for start in range(0, len(waveform) - frame_size, hop_size):
        frame = waveform[start : start + frame_size]
        # Energy check (voiced frame)
        energy = np.sum(frame**2)
        if energy < 1e-4:
            continue

        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(frame) - 1 :]
        if len(corr) > max_lag:
            lag_region = corr[min_lag:max_lag]
            peak_idx = np.argmax(lag_region) + min_lag
            if corr[peak_idx] > 0.3 * corr[0]:
                pitch = sample_rate / peak_idx
                pitches.append(pitch)

    pitches_arr = np.array(pitches) if pitches else np.array([0.0])
    f0_mean = float(np.mean(pitches_arr))
    f0_std = float(np.std(pitches_arr))
    f0_range = float(np.max(pitches_arr) - np.min(pitches_arr)) if len(pitches) > 0 else 0.0

    # Synthetic TTS voices often present unnaturally low or abnormally step-wise pitch jitter
    diffs = np.diff(pitches_arr) if len(pitches_arr) > 1 else np.array([0.0])
    prosodic_jitter = float(np.mean(np.abs(diffs)))

    return {
        "f0_mean_hz": round(f0_mean, 2),
        "f0_std_hz": round(f0_std, 2),
        "f0_range_hz": round(f0_range, 2),
        "prosodic_jitter": round(prosodic_jitter, 2),
        "voiced_frames_count": len(pitches),
    }
