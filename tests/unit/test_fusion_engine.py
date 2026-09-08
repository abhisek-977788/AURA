"""
Unit tests for Score Fusion Engine:
Verifies missing modality resilience, thresholding, and uncertainty estimation.
"""

import pytest
from packages.schemas.analysis_result import Decision, ModalityResult, ModalityType, ModelInferenceResult
from services.fusion.fusion_engine import FusionEngine


@pytest.mark.asyncio
async def test_fusion_both_modalities():
    engine = FusionEngine(high_risk_threshold=0.80, medium_risk_threshold=0.50)

    audio_res = ModalityResult(
        modality=ModalityType.AUDIO,
        available=True,
        ensemble_score=0.85,
        ensemble_confidence=0.90,
    )
    video_res = ModalityResult(
        modality=ModalityType.VIDEO,
        available=True,
        ensemble_score=0.88,
        ensemble_confidence=0.85,
    )

    res = await engine.fuse(
        event_id="evt-fusion-1",
        audio_result=audio_res,
        video_result=video_res,
    )

    assert res.decision == Decision.HIGH_RISK
    assert res.synthetic_media_probability >= 0.80
    assert res.requires_human_review is True
    assert ModalityType.AUDIO in res.available_modalities
    assert ModalityType.VIDEO in res.available_modalities


@pytest.mark.asyncio
async def test_missing_modality_is_not_evidence_of_fakery():
    engine = FusionEngine()

    # Video only, clean natural video
    video_res = ModalityResult(
        modality=ModalityType.VIDEO,
        available=True,
        ensemble_score=0.10,
        ensemble_confidence=0.80,
    )

    # Audio is MISSING
    res = await engine.fuse(
        event_id="evt-fusion-video-only",
        audio_result=None,
        video_result=video_res,
    )

    # Missing audio should NOT artificially increase fake probability
    assert res.decision == Decision.LOW_RISK
    assert res.synthetic_media_probability <= 0.20
    assert ModalityType.AUDIO in res.missing_modalities
    assert any("Audio modality not present" in lim for lim in res.limitations)


@pytest.mark.asyncio
async def test_low_confidence_produces_inconclusive():
    engine = FusionEngine(min_confidence_for_decision=0.50)

    audio_res = ModalityResult(
        modality=ModalityType.AUDIO,
        available=True,
        ensemble_score=0.95,
        ensemble_confidence=0.20,  # Below threshold
    )

    res = await engine.fuse(
        event_id="evt-fusion-uncertain",
        audio_result=audio_res,
    )

    assert res.decision == Decision.INCONCLUSIVE
