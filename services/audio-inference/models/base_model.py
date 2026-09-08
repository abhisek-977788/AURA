"""
Base Model Adapter for Audio Deepfake Detectors.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any
import numpy as np

from packages.schemas.analysis_result import ModelInferenceResult, ModalityType


class AudioModelAdapter(abc.ABC):
    model_name: str
    model_version: str
    supported_sample_rates: list[int]

    def __init__(self, model_name: str, model_version: str, supported_sample_rates: list[int]) -> None:
        self.model_name = model_name
        self.model_version = model_version
        self.supported_sample_rates = supported_sample_rates
        self._is_loaded = False
        self._model_hash: str | None = None

    @abc.abstractmethod
    async def predict(self, waveform: np.ndarray, sample_rate: int) -> ModelInferenceResult:
        """Runs model inference on waveform input."""
        pass

    def load_weights(self, path: Path | str) -> None:
        p = Path(path)
        if not p.exists():
            self._is_loaded = False
            return
        self._is_loaded = True

    def is_loaded(self) -> bool:
        return self._is_loaded

    def get_model_hash(self) -> str | None:
        return self._model_hash
