"""
Unit tests for Blink Analysis (EAR calculations and state machine).
"""

import numpy as np
from services.video_inference.features.blink import BlinkAnalyzer


def test_ear_formula():
    # Construct artificial eye points:
    # p1 (0, 0), p4 (10, 0) -> horizontal distance = 10
    # p2 (3, 2), p6 (3, -2) -> vertical 1 = 4
    # p3 (7, 2), p5 (7, -2) -> vertical 2 = 4
    # EAR = (4 + 4) / (2 * 10) = 8 / 20 = 0.40
    pts = np.array([
        [0.0, 0.0],
        [3.0, 2.0],
        [7.0, 2.0],
        [10.0, 0.0],
        [7.0, -2.0],
        [3.0, -2.0],
    ])
    ear = BlinkAnalyzer.calculate_ear(pts)
    assert round(ear, 2) == 0.40


def test_blink_detection_sequence():
    analyzer = BlinkAnalyzer(ear_threshold=0.20, consecutive_frames_min=2)
    # Series with a blink event (ear drops below 0.20 for 2 frames)
    left_ear = [0.30, 0.30, 0.15, 0.14, 0.31, 0.30, 0.29]
    right_ear = [0.30, 0.31, 0.16, 0.13, 0.30, 0.29, 0.30]

    analysis = analyzer.analyze_sequence(left_ear, right_ear, fps=2.0)
    assert analysis.available is True
    assert analysis.blink_count == 1
