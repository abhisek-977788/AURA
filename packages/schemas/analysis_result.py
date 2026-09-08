"""
Analysis result schemas — produced by the fusion layer after model inference.

IMPORTANT DESIGN CONSTRAINTS:
- Results are calibrated probabilities, not legal determinations.
- Missing modalities must not be treated as evidence of fakery.
- Every result must include model versions, quality limitations, and uncertainty.
- Results with insufficient quality must return decision="inconclusive".
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ── Enumerations ──────────────────────────────────────────────────────────────


class Decision(str, Enum):
    HIGH_RISK = "high_risk"
    MEDIUM_RISK = "medium_risk"
    LOW_RISK = "low_risk"
    INCONCLUSIVE = "inconclusive"


class ModalityType(str, Enum):
    AUDIO = "audio"
    VIDEO = "video"
    SYNCHRONIZATION = "sync"
    FREQUENCY = "frequency"
    BLINK = "blink"
    TEXTURE = "texture"
    FACE_BOUNDARY = "face_boundary"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    IN_REVIEW = "in_review"
    REVIEWED = "reviewed"
    ESCALATED = "escalated"


# ── Model-level results ───────────────────────────────────────────────────────


class ModelInferenceResult(BaseModel):
    """Output from a single model on a single modality."""
    model_name: str
    model_version: str
    model_hash: str | None = None          # SHA-256 of model weights file
    modality: ModalityType
    synthetic_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    inference_latency_ms: float | None = None
    feature_version: str | None = None
    limitations: list[str] = Field(default_factory=list)
    raw_scores: dict[str, float] = Field(default_factory=dict)


class ModalityResult(BaseModel):
    """Aggregated result for one modality (e.g., audio or video)."""
    modality: ModalityType
    available: bool = True
    models: list[ModelInferenceResult] = Field(default_factory=list)
    ensemble_score: float | None = Field(default=None, ge=0.0, le=1.0)
    ensemble_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    top_features: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @field_validator("ensemble_score", "ensemble_confidence")
    @classmethod
    def scores_only_when_available(cls, v: float | None, info: Any) -> float | None:
        # Scores should be None when modality is not available
        return v


class SyncResult(BaseModel):
    """Audio-visual synchronization analysis result."""
    available: bool = False
    phoneme_viseme_offset_ms: float | None = None
    mouth_motion_delay_ms: float | None = None
    temporal_drift_score: float | None = None   # 0.0–1.0, higher = more drift
    sync_confidence: float | None = None
    face_occluded: bool = False
    multiple_faces: bool = False
    off_camera_speaker: bool = False
    not_speech: bool = False
    sync_score: float | None = Field(default=None, ge=0.0, le=1.0)
    limitations: list[str] = Field(default_factory=list)


class BlinkAnalysis(BaseModel):
    """
    Blink analysis using Eye Aspect Ratio (EAR).

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    NOTE: Blink rate alone is NOT a reliable deepfake indicator.
    Blink frequency varies with fatigue, attention, health, camera angle,
    frame rate, and individual behavior. This is one signal among many.
    """
    available: bool = False
    observation_duration_s: float | None = None
    blink_count: int | None = None
    blinks_per_minute: float | None = None
    mean_blink_duration_ms: float | None = None
    blink_regularity_score: float | None = None  # 0.0–1.0
    left_ear_mean: float | None = None
    right_ear_mean: float | None = None
    landmark_confidence: float | None = None
    anomaly_score: float | None = Field(default=None, ge=0.0, le=1.0)
    limitations: list[str] = Field(default_factory=list)


class QualityReport(BaseModel):
    """Quality gate result from the quality assessment service."""
    is_sufficient: bool
    resolution: str | None = None
    frame_rate: float | None = None
    bitrate_kbps: int | None = None
    snr_db: float | None = None
    face_size_px: int | None = None
    face_visibility_pct: float | None = None
    landmark_confidence: float | None = None
    speech_activity_ratio: float | None = None
    codec: str | None = None
    compression_severity: str | None = None
    insufficiency_reasons: list[str] = Field(default_factory=list)
    recommended_action: str | None = None


# ── Primary Fusion Result ─────────────────────────────────────────────────────


class AnalysisResult(BaseModel):
    """
    Final fused analysis result produced by the fusion service.

    Consumers MUST read:
    - `requires_human_review` — never act on high-risk results without review
    - `limitations` — understand what evidence is missing or degraded
    - `decision_rationale` — written explanation of score interpretation
    - `confidence` — model certainty, not evidence certainty
    """

    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_id: str
    case_id: str | None = None
    job_id: str | None = None

    # Core result — calibrated probability
    synthetic_media_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    decision: Decision = Decision.INCONCLUSIVE

    # Disclaimer embedded in every result
    legal_disclaimer: str = (
        "This result is a statistical estimate produced by automated software. "
        "It is NOT a legal determination, NOT court-admissible evidence, and "
        "MUST NOT be used to make irreversible identity or criminal conduct decisions "
        "without qualified human analyst review and legal assessment."
    )

    # Modality breakdown
    available_modalities: list[ModalityType] = Field(default_factory=list)
    missing_modalities: list[ModalityType] = Field(default_factory=list)
    modality_results: dict[str, ModalityResult] = Field(default_factory=dict)
    sync_result: SyncResult = Field(default_factory=SyncResult)
    blink_analysis: BlinkAnalysis = Field(default_factory=BlinkAnalysis)

    # Explanation pointers
    top_contributing_signals: list[str] = Field(default_factory=list)
    gradcam_artifact_path: str | None = None
    shap_artifact_path: str | None = None
    explanation_artifact_paths: list[str] = Field(default_factory=list)

    # Quality
    quality_report: QualityReport | None = None

    # Uncertainty and limitations
    model_uncertainty: float | None = Field(default=None, ge=0.0, le=1.0)
    limitations: list[str] = Field(default_factory=list)
    decision_rationale: str | None = None

    # Human review
    requires_human_review: bool = True
    review_status: ReviewStatus = ReviewStatus.PENDING
    analyst_conclusion: str | None = None
    analyst_id: str | None = None
    analyst_timestamp: datetime | None = None

    # Versioning for reproducibility
    fusion_version: str | None = None
    calibration_version: str | None = None
    software_version: str | None = None
    model_versions: dict[str, str] = Field(default_factory=dict)

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processing_duration_ms: float | None = None

    model_config = {"use_enum_values": True}

    @classmethod
    def inconclusive(
        cls,
        event_id: str,
        reason: str,
        quality_report: QualityReport | None = None,
    ) -> "AnalysisResult":
        """Factory for inconclusive results due to insufficient signal quality."""
        return cls(
            event_id=event_id,
            decision=Decision.INCONCLUSIVE,
            synthetic_media_probability=None,
            confidence=None,
            limitations=[reason],
            quality_report=quality_report,
            decision_rationale=(
                f"Analysis could not be completed: {reason}. "
                "Collect higher-quality media or request manual analyst review."
            ),
        )
