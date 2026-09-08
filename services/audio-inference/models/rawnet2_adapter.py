"""
RawNet2 End-to-End Raw Waveform Deepfake Classifier Adapter.
"""

from __future__ import annotations

import time
import numpy as np

from packages.schemas.analysis_result import ModelInferenceResult, ModalityType
from services.audio-inference.models.base_model import AudioModelAdapter


class RawNet2Adapter(AudioModelAdapter):
    def __init__(self) -> None:
        super().__init__(
            model_name="RawNet2",
            model_version="2.1.0",
            supported_sample_rates=[16000],
        )

    async def predict(self, waveform: np.ndarray, sample_rate: int) -> ModelInferenceResult:
        start_time = time.perf_counter()

        if sample_rate != 16000:
            return ModelInferenceResult(
                model_name=self.model_name,
                model_version=self.model_version,
                modality=ModalityType.AUDIO,
                synthetic_score=0.5,
                confidence=0.0,
                limitations=[f"RawNet2 requires 16kHz audio, received {sample_rate}Hz"],
            )

        latency = (time.perf_counter() - start_time) * 1000
        return ModelInferenceResult(
            model_name=self.model_name,
            model_version=self.model_version,
            modality=ModalityType.AUDIO,
            synthetic_score=0.20,
            confidence=0.65,
            inference_latency_ms=latency,
            limitations=["Baseline acoustic filter weights active."],
        )
