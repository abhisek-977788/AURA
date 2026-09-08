"""
Evaluation Metrics for Deepfake & Synthetic Voice Detection.
Implements EER (Equal Error Rate), ROC-AUC, Brier score, and Expected Calibration Error (ECE).
"""

from __future__ import annotations

import numpy as np


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[float, float]:
    """
    Computes Equal Error Rate (EER) and the optimal threshold.
    y_true: 0 for genuine/bonafide, 1 for spoof/synthetic.
    """
    thresholds = np.sort(np.unique(y_scores))
    fprs = []
    fnrs = []

    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)

    if positives == 0 or negatives == 0:
        return 0.0, 0.5

    for th in thresholds:
        fnr = np.sum((y_scores < th) & (y_true == 1)) / positives
        fpr = np.sum((y_scores >= th) & (y_true == 0)) / negatives
        fnrs.append(fnr)
        fprs.append(fpr)

    fnrs = np.array(fnrs)
    fprs = np.array(fprs)

    # Point where |fpr - fnr| is minimized
    idx = np.nanargmin(np.absolute(fnrs - fprs))
    eer = float((fnrs[idx] + fprs[idx]) / 2.0)
    opt_threshold = float(thresholds[idx])

    return eer, opt_threshold


def compute_brier_score(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """Mean squared error of calibrated probabilities against binary ground truth."""
    return float(np.mean((y_probs - y_true) ** 2))


def compute_ece(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (y_probs >= bin_lower) & (y_probs < bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)
