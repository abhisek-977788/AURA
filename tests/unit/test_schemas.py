"""
Unit tests for MediaEvent, AnalysisResult, and EvidenceManifest schemas.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from packages.schemas.media_event import MediaEvent, SourceType
from packages.schemas.analysis_result import AnalysisResult, Decision
from packages.schemas.evidence_manifest import EvidenceManifest, ChainOfCustodyEventType


def test_media_event_creation(sample_video_event: MediaEvent):
    assert sample_video_event.event_id == "test-video-event-001"
    assert sample_video_event.source_type == SourceType.RECORDED_VIDEO
    assert sample_video_event.is_live() is False


def test_pseudonymous_subject_validation():
    # Attempting to assign real email or clear ID pattern should fail
    with pytest.raises(ValidationError):
        MediaEvent(
            source_type=SourceType.PHOTO,
            source_adapter="PhotoAdapter",
            subject_id="name:JohnDoe@example.com",
        )


def test_analysis_result_inconclusive_factory():
    res = AnalysisResult.inconclusive(
        event_id="evt-999",
        reason="Severe signal degradation",
    )
    assert res.decision == Decision.INCONCLUSIVE
    assert res.synthetic_media_probability is None
    assert res.confidence is None
    assert "Severe signal degradation" in res.limitations
    assert res.legal_disclaimer != ""


def test_evidence_manifest_custody():
    manifest = EvidenceManifest(
        case_id="case-001",
        event_id="evt-001",
        analysis_id="an-001",
        acquisition_source="VideoAdapter",
        ingestion_timestamp="2026-09-08T10:00:00Z",
        processing_timestamp="2026-09-08T10:01:00Z",
        original_media_hash="abc123hash",
        software_version="0.1.0",
    )
    manifest.add_custody_event(
        event_type=ChainOfCustodyEventType.ACQUIRED,
        actor_id="examiner-01",
        actor_role="forensic_examiner",
        description="Acquired from authorized storage",
        component="Gateway",
    )
    assert len(manifest.chain_of_custody) == 1
    assert manifest.chain_of_custody[0].actor_id == "examiner-01"
