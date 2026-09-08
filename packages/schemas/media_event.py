"""
Common MediaEvent schema — the universal internal representation produced by
every ingestion adapter and consumed by every downstream service.

All ingestion adapters MUST produce a MediaEvent. The inference layer MUST NOT
depend directly on Chrome, SIP, RTP, uploaded files, or any particular frontend.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ── Enumerations ──────────────────────────────────────────────────────────────


class SourceType(str, Enum):
    PHOTO = "photo"
    RECORDED_VIDEO = "recorded_video"
    RECORDED_AUDIO = "recorded_audio"
    LIVE_AUDIO = "live_audio"
    LIVE_VIDEO = "live_video"


class ProcessingStatus(str, Enum):
    RECEIVED = "received"
    NORMALIZED = "normalized"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


class PrivacyLevel(str, Enum):
    """Controls how much raw media may be retained after analysis."""
    EPHEMERAL = "ephemeral"        # No retention. Live streams default.
    TRANSIENT = "transient"        # Retain only for active analysis window.
    AUTHORIZED = "authorized"      # Retain under explicit case authorization.
    EVIDENCE = "evidence"          # Full forensic evidence retention.


# ── Sub-models ────────────────────────────────────────────────────────────────


class QualityMetrics(BaseModel):
    """Pre-inference quality measurements. Populated by the Quality service."""
    resolution: str | None = None           # e.g. "1920x1080"
    frame_rate: float | None = None         # fps
    bitrate_kbps: int | None = None
    codec: str | None = None
    sample_rate_hz: int | None = None
    snr_db: float | None = None             # Signal-to-noise ratio
    clipping_detected: bool | None = None
    echo_detected: bool | None = None
    packet_loss_pct: float | None = None
    jitter_ms: float | None = None
    compression_severity: str | None = None # "light"|"medium"|"heavy"
    face_size_px: int | None = None         # largest detected face width
    face_visibility_pct: float | None = None
    motion_blur_score: float | None = None  # 0.0–1.0
    occlusion_pct: float | None = None
    lighting_quality: str | None = None     # "good"|"low"|"uneven"
    landmark_confidence: float | None = None  # 0.0–1.0
    speech_activity_ratio: float | None = None  # VAD ratio
    is_sufficient: bool = True
    insufficiency_reasons: list[str] = Field(default_factory=list)


class PrivacyPolicy(BaseModel):
    """Privacy constraints applied to this event."""
    level: PrivacyLevel = PrivacyLevel.TRANSIENT
    retention_hours: int | None = None       # None = session only
    pseudonymous_only: bool = True
    delete_on_complete: bool = False
    gdpr_applicable: bool = False
    jurisdiction: str | None = None


# ── Primary Schema ────────────────────────────────────────────────────────────


class MediaEvent(BaseModel):
    """
    Versioned common schema produced by every ingestion adapter.

    IMPORTANT: Never include raw biometric media in this schema.
    Raw media is referenced by storage path only when authorized.
    """

    schema_version: str = "1.0"

    # Identifiers
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    case_id: str | None = None
    job_id: str | None = None

    # Source
    source_type: SourceType
    source_adapter: str = Field(
        description="Adapter class name that produced this event, e.g. 'VideoAdapter'"
    )

    # Subject (always pseudonymous)
    subject_id: str | None = Field(
        default=None,
        description="Pseudonymous identifier. Never a real name or biometric ID.",
    )

    # Timestamps
    ingestion_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    acquisition_timestamp: datetime | None = None  # When media was originally captured

    # Media metadata — no raw media bytes
    media_hash: str | None = Field(
        default=None,
        description="SHA-256 of original media file. None for live streams.",
    )
    storage_path: str | None = Field(
        default=None,
        description=(
            "Authorized storage reference (S3 key or local path). "
            "Null unless retention policy permits."
        ),
    )
    codec: str | None = None
    container_format: str | None = None
    sample_rate_hz: int | None = None
    video_resolution: str | None = None
    duration_ms: int | None = None
    channels: int | None = None
    file_size_bytes: int | None = None

    # Quality assessment (populated after quality service runs)
    quality_metrics: QualityMetrics = Field(default_factory=QualityMetrics)

    # Privacy and authorization
    privacy_policy: PrivacyPolicy = Field(default_factory=PrivacyPolicy)
    consent_or_authorization_reference: str | None = Field(
        default=None,
        description=(
            "Reference to the legal authorization or consent record "
            "permitting this analysis."
        ),
    )

    # Versioning
    model_policy_version: str | None = None
    software_version: str | None = None

    # Processing state
    processing_status: ProcessingStatus = ProcessingStatus.RECEIVED

    # Extensible metadata
    extra_metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"use_enum_values": True}

    @field_validator("subject_id")
    @classmethod
    def subject_must_be_pseudonymous(cls, v: str | None) -> str | None:
        """Block obvious real identifiers. Not a complete check — use proper pseudonymization."""
        if v is not None:
            prohibited_patterns = ["@", "name:", "id:"]
            for p in prohibited_patterns:
                if p in v.lower():
                    raise ValueError(
                        f"subject_id must be pseudonymous. Found pattern '{p}'. "
                        "Use a UUID or hashed identifier."
                    )
        return v

    def is_live(self) -> bool:
        return self.source_type in (SourceType.LIVE_AUDIO, SourceType.LIVE_VIDEO)

    def mark_status(self, status: ProcessingStatus) -> "MediaEvent":
        return self.model_copy(update={"processing_status": status})
