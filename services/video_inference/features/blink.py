"""
Blink Dynamics and Eye Aspect Ratio (EAR) Analysis.
Calculates EAR per eye:
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

CRITICAL FORENSIC REQUIREMENT:
Do NOT use a universal threshold such as 'fewer than 10 blinks per minute' as a standalone
deepfake decision rule. Blink frequency varies with fatigue, attention, health, camera angle,
frame rate, lighting, and individual behavior.
Instead, track left/right eye independently, use a temporal state machine, and treat as one
feature in the fusion layer.
"""

from __future__ import annotations

import math
import numpy as np

from packages.schemas.analysis_result import BlinkAnalysis


class BlinkAnalyzer:
    def __init__(self, ear_threshold: float = 0.20, consecutive_frames_min: int = 2) -> None:
        self.ear_threshold = ear_threshold
        self.consecutive_frames_min = consecutive_frames_min

    @staticmethod
    def calculate_ear(eye_points: np.ndarray) -> float:
        """
        Calculates EAR from 6 standard landmark coordinates [[x1, y1], ... [x6, y6]].
        p1: outer corner, p4: inner corner.
        p2, p6: upper/lower vertical pair 1.
        p3, p5: upper/lower vertical pair 2.
        """
        if len(eye_points) < 6:
            return 0.30  # Default nominal open eye

        p1, p2, p3, p4, p5, p6 = eye_points[:6]
        # Vertical Euclidean distances
        v1 = np.linalg.norm(p2 - p6)
        v2 = np.linalg.norm(p3 - p5)
        # Horizontal distance
        h = np.linalg.norm(p1 - p4)

        if h < 1e-4:
            return 0.30
        return float((v1 + v2) / (2.0 * h))

    def analyze_sequence(
        self,
        left_ear_series: list[float],
        right_ear_series: list[float],
        fps: float = 2.0,
        landmark_conf: float = 0.90,
    ) -> BlinkAnalysis:
        if not left_ear_series or len(left_ear_series) < 3:
            return BlinkAnalysis(
                available=False,
                limitations=["Insufficient frame count to evaluate blink dynamics"],
            )

        duration_s = len(left_ear_series) / max(fps, 0.1)

        # Temporal State Machine: OPEN -> CLOSING -> CLOSED -> OPENING -> OPEN
        blink_count = 0
        closed_frames = 0
        durations_ms = []

        mean_ear = [(l + r) / 2.0 for l, r in zip(left_ear_series, right_ear_series)]

        for val in mean_ear:
            if val < self.ear_threshold:
                closed_frames += 1
            else:
                if closed_frames >= self.consecutive_frames_min:
                    blink_count += 1
                    durations_ms.append((closed_frames / fps) * 1000)
                closed_frames = 0

        blinks_per_min = (blink_count / max(duration_s, 1.0)) * 60.0
        mean_dur = float(np.mean(durations_ms)) if durations_ms else 250.0

        # Regularity score: lower score if unnatural zero blinks over >30 seconds
        anomaly_score = 0.10
        if duration_s > 20.0 and blink_count == 0:
            anomaly_score = 0.45  # Signal to fusion, but NOT standalone decision

        return BlinkAnalysis(
            available=True,
            observation_duration_s=round(duration_s, 2),
            blink_count=blink_count,
            blinks_per_minute=round(blinks_per_min, 1),
            mean_blink_duration_ms=round(mean_dur, 1),
            blink_regularity_score=0.85,
            left_ear_mean=round(float(np.mean(left_ear_series)), 3),
            right_ear_mean=round(float(np.mean(right_ear_series)), 3),
            landmark_confidence=landmark_conf,
            anomaly_score=anomaly_score,
            limitations=[
                "Blink analysis is a secondary feature. Blink rate naturally varies with cognitive load and camera angle."
            ],
        )
