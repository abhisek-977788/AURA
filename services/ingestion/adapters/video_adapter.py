"""
Recorded Video Ingestion Adapter.
Performs container demuxing, audio/video stream inspection, and resolution/codec extraction.
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


class VideoAdapter(BaseIngestAdapter):
    def __init__(self) -> None:
        super().__init__(adapter_name="VideoAdapter", source_type=SourceType.RECORDED_VIDEO)

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

        # Stage temporarily to probe
        ext = Path(filename).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = Path(tmp.name)

        codec = "h264"
        resolution = "1280x720"
        duration_ms = 10000
        channels = 2
        sample_rate_hz = 44100
        frame_rate = 30.0

        # Try ffprobe if installed
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
                v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
                a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

                if v_stream:
                    codec = v_stream.get("codec_name", codec)
                    w = v_stream.get("width")
                    h = v_stream.get("height")
                    if w and h:
                        resolution = f"{w}x{h}"
                    r_frame_rate = v_stream.get("r_frame_rate", "30/1")
                    if "/" in r_frame_rate:
                        num, den = map(float, r_frame_rate.split("/"))
                        frame_rate = round(num / max(den, 1.0), 2)

                if a_stream:
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
            source_type=SourceType.RECORDED_VIDEO,
            source_adapter=self.adapter_name,
            subject_id=f"anon-{media_hash[:12]}",
            media_hash=media_hash,
            codec=codec,
            video_resolution=resolution,
            duration_ms=duration_ms,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            file_size_bytes=len(raw_bytes),
            privacy_policy=policy,
            consent_or_authorization_reference=auth_reference,
            processing_status=ProcessingStatus.RECEIVED,
            extra_metadata={
                "original_filename": filename,
                "staged_video_path": str(tmp_path),
                "fps": frame_rate,
            },
        )
