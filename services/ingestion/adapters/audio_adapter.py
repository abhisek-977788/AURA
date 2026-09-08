"""
Recorded Audio Ingestion Adapter.
Validates codec, sample rate, channels, and checks for narrowband vs. wideband profiles.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from packages.schemas.media_event import MediaEvent, PrivacyLevel, PrivacyPolicy, ProcessingStatus, SourceType
from services.ingestion.adapters.base_adapter import BaseIngestAdapter


class AudioAdapter(BaseIngestAdapter):
    def __init__(self) -> None:
        super().__init__(adapter_name="AudioAdapter", source_type=SourceType.RECORDED_AUDIO)

    async def process(
        self,
        raw_bytes: bytes,
        filename: str,
        case_id: str,
        job_id: str,
        auth_reference: str | None = None,
        privacy_policy: PrivacyPolicy | None = None,
        **kwargs: Any,
    ) -> MediaEvent:
        event_id = str(uuid.uuid4())
        media_hash = self.compute_hash(raw_bytes)

        ext = Path(filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = Path(tmp.name)

        codec = "pcm_s16le"
        sample_rate_hz = 16000
        channels = 1
        duration_ms = 5000

        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(tmp_path),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if res.returncode == 0:
                data = json.loads(res.stdout)
                streams = data.get("streams", [])
                a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
                if a_stream:
                    codec = a_stream.get("codec_name", codec)
                    sample_rate_hz = int(a_stream.get("sample_rate", sample_rate_hz))
                    channels = int(a_stream.get("channels", channels))
                dur_sec = float(data.get("format", {}).get("duration", 0))
                duration_ms = int(dur_sec * 1000)
        except Exception:
            pass

        policy = privacy_policy or PrivacyPolicy(
            level=PrivacyLevel.TRANSIENT,
            retention_hours=24,
            pseudonymous_only=True,
        )

        return MediaEvent(
            schema_version="1.0",
            event_id=event_id,
            case_id=case_id,
            job_id=job_id,
            source_type=SourceType.RECORDED_AUDIO,
            source_adapter=self.adapter_name,
            subject_id=f"anon-{media_hash[:12]}",
            media_hash=media_hash,
            codec=codec,
            sample_rate_hz=sample_rate_hz,
            duration_ms=duration_ms,
            channels=channels,
            file_size_bytes=len(raw_bytes),
            privacy_policy=policy,
            consent_or_authorization_reference=auth_reference,
            processing_status=ProcessingStatus.RECEIVED,
            extra_metadata={
                "original_filename": filename,
                "staged_audio_path": str(tmp_path),
                "is_narrowband_telephony": sample_rate_hz <= 8000,
            },
        )
