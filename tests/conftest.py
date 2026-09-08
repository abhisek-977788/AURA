"""
Pytest configuration and shared fixtures for AURA tests.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from packages.schemas.media_event import MediaEvent, PrivacyPolicy, PrivacyLevel, ProcessingStatus, SourceType
from packages.schemas.analysis_result import (
    AnalysisResult,
    Decision,
    ModalityResult,
    ModalityType,
    ModelInferenceResult,
    QualityReport,
)


@pytest.fixture
def sample_video_event() -> MediaEvent:
    return MediaEvent(
        event_id="test-video-event-001",
        case_id="case-123",
        job_id="job-456",
        source_type=SourceType.RECORDED_VIDEO,
        source_adapter="VideoAdapter",
        subject_id="anon-subject-1234",
        media_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        video_resolution="1280x720",
        codec="h264",
        duration_ms=5000,
        sample_rate_hz=16000,
        channels=1,
        privacy_policy=PrivacyPolicy(level=PrivacyLevel.TRANSIENT, retention_hours=24),
    )


@pytest.fixture
def sample_audio_event() -> MediaEvent:
    return MediaEvent(
        event_id="test-audio-event-001",
        case_id="case-123",
        job_id="job-789",
        source_type=SourceType.RECORDED_AUDIO,
        source_adapter="AudioAdapter",
        subject_id="anon-audio-5678",
        media_hash="a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e",
        sample_rate_hz=16000,
        channels=1,
        duration_ms=4000,
        codec="pcm_s16le",
    )
