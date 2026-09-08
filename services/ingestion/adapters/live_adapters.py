"""
Live Video and Live Audio Ingestion Adapters for WebRTC client sessions and SBC SIP/RTP interception relays.
Enforces ephemeral retention: NO raw media is stored by default unless high-threat evidence capture is triggered.
"""

from __future__ import annotations

import uuid
from typing import Any

from packages.schemas.media_event import MediaEvent, PrivacyLevel, PrivacyPolicy, ProcessingStatus, SourceType
from services.ingestion.adapters.base_adapter import BaseIngestAdapter


class LiveVideoAdapter(BaseIngestAdapter):
    def __init__(self) -> None:
        super().__init__(adapter_name="LiveVideoAdapter", source_type=SourceType.LIVE_VIDEO)

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
        policy = privacy_policy or PrivacyPolicy(
            level=PrivacyLevel.EPHEMERAL,
            retention_hours=0,
            pseudonymous_only=True,
        )

        return MediaEvent(
            schema_version="1.0",
            event_id=event_id,
            case_id=case_id,
            job_id=job_id,
            source_type=SourceType.LIVE_VIDEO,
            source_adapter=self.adapter_name,
            subject_id=f"anon-live-{event_id[:8]}",
            media_hash=None,  # No static hash for unbounded live streams
            storage_path=None, # NEVER store raw live stream by default
            codec="raw_frame_rgb",
            video_resolution=kwargs.get("resolution", "640x480"),
            privacy_policy=policy,
            consent_or_authorization_reference=auth_reference,
            processing_status=ProcessingStatus.RECEIVED,
            extra_metadata={
                "session_id": kwargs.get("session_id"),
                "timestamp_ms": kwargs.get("timestamp_ms"),
                "frame_size_bytes": len(raw_bytes),
            },
        )


class LiveAudioAdapter(BaseIngestAdapter):
    def __init__(self) -> None:
        super().__init__(adapter_name="LiveAudioAdapter", source_type=SourceType.LIVE_AUDIO)

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
        policy = privacy_policy or PrivacyPolicy(
            level=PrivacyLevel.EPHEMERAL,
            retention_hours=0,
            pseudonymous_only=True,
        )

        sample_rate = kwargs.get("sample_rate_hz", 8000)

        return MediaEvent(
            schema_version="1.0",
            event_id=event_id,
            case_id=case_id,
            job_id=job_id,
            source_type=SourceType.LIVE_AUDIO,
            source_adapter=self.adapter_name,
            subject_id=f"anon-live-{event_id[:8]}",
            media_hash=None,
            storage_path=None,
            codec="pcm_s16le",
            sample_rate_hz=sample_rate,
            channels=1,
            privacy_policy=policy,
            consent_or_authorization_reference=auth_reference,
            processing_status=ProcessingStatus.RECEIVED,
            extra_metadata={
                "session_id": kwargs.get("session_id"),
                "is_telephony_8k": sample_rate == 8000,
            },
        )
