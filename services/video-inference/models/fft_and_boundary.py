"""
FFT Frequency-Domain Classifier and Face Boundary Artifact Detector.
"""

from __future__ import annotations

import time
import numpy as np

from packages.schemas.analysis_result import ModelInferenceResult, ModalityType
from services.video-inference.features.fft_features import extract_fft_features


class FFTClassifier:
    def __init__(self) -> None:
        self.model_name = "FFT-Spectral"
        self.model_version = "1.1.0"

    async def predict(self, face_crops: list[np.ndarray]) -> ModelInferenceResult:
        start_time = time.perf_counter()

        if not face_crops:
            return ModelInferenceResult(
                model_name=self.model_name,
                model_version=self.model_version,
                modality=ModalityType.FREQUENCY,
                synthetic_score=0.5,
                confidence=0.0,
                limitations=["No faces provided for frequency analysis"],
            )

        scores = []
        for face in face_crops[:10]:
            gray = np.mean(face, axis=2) if face.ndim == 3 else face
            fft_data = extract_fft_features(gray)
            # High frequency checkerboard + energy discrepancy indicator
            c_score = fft_data["checkerboard_score"]
            s = 1.0 / (1.0 + np.exp(-(c_score - 1.2) * 3.0))
            scores.append(s)

        mean_score = float(np.mean(scores))
        latency = (time.perf_counter() - start_time) * 1000

        return ModelInferenceResult(
            model_name=self.model_name,
            model_version=self.model_version,
            modality=ModalityType.FREQUENCY,
            synthetic_score=round(mean_score, 4),
            confidence=0.84,
            inference_latency_ms=round(latency, 2),
            raw_scores={"mean_checkerboard_ratio": round(float(np.mean([s for s in scores])), 3)},
        )


class FaceBoundaryDetector:
    def __init__(self) -> None:
        self.model_name = "FaceBoundary-Artifacts"
        self.model_version = "1.0.0"

    async def predict(self, face_crops: list[np.ndarray]) -> ModelInferenceResult:
        start_time = time.perf_counter()

        if not face_crops:
            return ModelInferenceResult(
                model_name=self.model_name,
                model_version=self.model_version,
                modality=ModalityType.FACE_BOUNDARY,
                synthetic_score=0.5,
                confidence=0.0,
                limitations=["No faces available to test boundary warping"],
            )

        # Inspect perimeter gradient contrast (detects paste/warp seams)
        edge_discontinuities = []
        for face in face_crops[:8]:
            border_top = face[:4, :, :]
            border_bottom = face[-4:, :, :]
            inner = face[8:12, :, :]
            diff = float(np.abs(np.mean(border_top) - np.mean(inner)))
            edge_discontinuities.append(diff)

        avg_edge = float(np.mean(edge_discontinuities)) if edge_discontinuities else 0.0
        # Normal faces have gradual lighting; harsh seams indicate face swaps
        synth_score = float(np.clip(avg_edge / 40.0, 0.05, 0.90))
        latency = (time.perf_counter() - start_time) * 1000

        return ModelInferenceResult(
            model_name=self.model_name,
            model_version=self.model_version,
            modality=ModalityType.FACE_BOUNDARY,
            synthetic_score=round(synth_score, 4),
            confidence=0.72,
            inference_latency_ms=round(latency, 2),
        )
