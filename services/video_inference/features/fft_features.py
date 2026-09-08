"""
2D Frequency Domain / Fourier Transform Analysis for Generative Model Artifacts.
Detects periodic checkerboard patterns and abnormal high-frequency distributions produced by GAN upsampling.
"""

from __future__ import annotations

import numpy as np


def extract_fft_features(face_image_gray: np.ndarray) -> dict:
    """Computes 2D FFT magnitude spectrum and high-frequency power ratio."""
    h, w = face_image_gray.shape[:2]
    if h < 32 or w < 32:
        return {"high_freq_ratio": 0.0, "checkerboard_score": 0.0, "spectral_peak_count": 0}

    # 2D FFT with center shift
    f = np.fft.fft2(face_image_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)

    # Calculate radially averaged power spectrum
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    max_r = int(min(cx, cy))
    low_freq_mask = r < (max_r * 0.3)
    high_freq_mask = r > (max_r * 0.7)

    total_energy = np.sum(np.abs(fshift)) + 1e-6
    high_energy = np.sum(np.abs(fshift)[high_freq_mask])
    high_freq_ratio = float(high_energy / total_energy)

    # Detect checkerboard periodicity (common in transposed convolution)
    corner_regions = (
        magnitude_spectrum[: h // 8, : w // 8],
        magnitude_spectrum[: h // 8, -w // 8 :],
        magnitude_spectrum[-h // 8 :, : w // 8],
        magnitude_spectrum[-h // 8 :, -w // 8 :],
    )
    corner_energy = np.mean([np.mean(c) for c in corner_regions])
    center_energy = np.mean(magnitude_spectrum[cy - 5 : cy + 5, cx - 5 : cx + 5])
    checkerboard_score = float(corner_energy / max(center_energy, 1.0))

    return {
        "high_freq_ratio": round(high_freq_ratio, 4),
        "checkerboard_score": round(checkerboard_score, 4),
        "spectral_peak_count": int(np.sum(magnitude_spectrum > (np.mean(magnitude_spectrum) + 3 * np.std(magnitude_spectrum)))),
    }
