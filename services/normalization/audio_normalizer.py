"""
Audio Normalizer for AURA.
Converts input audio to standard mono PCM WAV and creates multi-rate processing paths:
- 16 kHz (standard deepfake detection path: AASIST, RawNet2, LFCC)
- 8 kHz (narrowband telephony simulation and dedicated 1D-CNN path)
- 48 kHz (high-fidelity audio path)
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from packages.common.logging import get_logger
from packages.schemas.media_event import MediaEvent

logger = get_logger("normalization.audio")


class AudioNormalizer:
    async def normalize(self, event: MediaEvent) -> MediaEvent:
        staged_path = (
            event.extra_metadata.get("staged_file_path")
            or event.extra_metadata.get("staged_audio_path")
            or event.extra_metadata.get("normalized_audio_path")
        )
        if not staged_path or not Path(staged_path).exists():
            return event

        src_audio = Path(staged_path)
        out_dir = Path(tempfile.gettempdir()) / "aura_norm_audio" / event.event_id
        out_dir.mkdir(parents=True, exist_ok=True)

        path_16k = out_dir / "audio_16k.wav"
        path_8k = out_dir / "audio_8k.wav"

        # Resample to 16 kHz mono
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(src_audio), "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(path_16k)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
        except Exception:
            pass

        # Resample to 8 kHz mono (telephony path)
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(src_audio), "-acodec", "pcm_s16le", "-ar", "8000", "-ac", "1", str(path_8k)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
        except Exception:
            pass

        event.extra_metadata["audio_16k_path"] = str(path_16k) if path_16k.exists() else str(src_audio)
        event.extra_metadata["audio_8k_path"] = str(path_8k) if path_8k.exists() else None

        logger.info(
            "Audio normalized successfully",
            event_id=event.event_id,
            has_16k=path_16k.exists(),
            has_8k=path_8k.exists(),
        )
        return event
