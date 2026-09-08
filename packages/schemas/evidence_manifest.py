"""
Forensic evidence manifest schema.

Every evidence export must be cryptographically signed and include a complete
chain-of-custody record. Evidence packages are for investigative support only
and require qualified legal review before use in judicial proceedings.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ChainOfCustodyEventType(str, Enum):
    ACQUIRED = "acquired"
    INGESTED = "ingested"
    NORMALIZED = "normalized"
    QUALITY_ASSESSED = "quality_assessed"
    ANALYZED = "analyzed"
    REVIEWED = "reviewed"
    EXPORTED = "exported"
    ACCESSED = "accessed"
    TRANSFERRED = "transferred"
    DELETED = "deleted"


class ChainOfCustodyEvent(BaseModel):
    """Single immutable event in the chain of custody."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: ChainOfCustodyEventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    actor_id: str  # Pseudonymous operator/system identifier
    actor_role: str
    description: str
    system_component: str
    software_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DerivedFileRecord(BaseModel):
    """Record of a file derived from the original during analysis."""
    file_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    description: str
    sha256_hash: str
    size_bytes: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    artifact_type: str  # "frame_extract"|"audio_extract"|"gradcam"|"shap"|"report"


class EvidenceManifest(BaseModel):
    """
    Cryptographically signed forensic evidence manifest.

    LEGAL NOTICE: This manifest and its associated evidence package are intended
    for use by authorized investigators only. Admissibility as evidence in legal
    proceedings depends on applicable jurisdiction, procedure, authentication
    requirements, and qualified expert testimony. This software does not
    certify legal admissibility. Consult qualified legal and forensic personnel
    before use in judicial proceedings.

    The manifest is Ed25519-signed. Verify with:
        aura-verify --manifest manifest.json --pubkey aura_signing.pub
    """

    manifest_version: str = "1.0"
    manifest_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Case & event linkage
    case_id: str
    event_id: str
    analysis_id: str
    subject_id: str | None = None  # Always pseudonymous

    # Acquisition metadata
    acquisition_timestamp: datetime | None = None
    acquisition_timezone: str = "UTC"
    acquisition_source: str  # Adapter class name

    # Processing timestamps
    ingestion_timestamp: datetime
    processing_timestamp: datetime
    export_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Original media integrity
    original_media_hash: str  # SHA-256
    original_media_size_bytes: int | None = None

    # Derived files
    derived_files: list[DerivedFileRecord] = Field(default_factory=list)

    # Software provenance
    software_version: str
    software_git_commit: str | None = None

    # Model provenance
    model_versions: dict[str, str] = Field(default_factory=dict)
    model_hashes: dict[str, str] = Field(default_factory=dict)
    preprocessing_config: dict[str, Any] = Field(default_factory=dict)

    # Analysis results summary (full result in derived files)
    synthetic_media_probability: float | None = None
    decision: str | None = None
    calibration_version: str | None = None

    # Quality and limitations
    quality_report_summary: dict[str, Any] = Field(default_factory=dict)
    analysis_limitations: list[str] = Field(default_factory=list)

    # Chain of custody
    chain_of_custody: list[ChainOfCustodyEvent] = Field(default_factory=list)

    # Alert and operator history
    alert_history: list[dict[str, Any]] = Field(default_factory=list)
    operator_actions: list[dict[str, Any]] = Field(default_factory=list)

    # Signature (populated by signer)
    signing_key_id: str | None = None
    signature_algorithm: str | None = None
    manifest_signature: str | None = None  # Base64-encoded Ed25519 signature
    signed_at: datetime | None = None

    # Verification
    legal_disclaimer: str = (
        "This evidence package was produced by automated software (AURA). "
        "It is NOT automatically court-admissible. Admissibility requires "
        "compliance with applicable rules of evidence, qualified expert testimony, "
        "and proper authentication. Jurisdiction-specific requirements (e.g., "
        "Section 65B IT Act for India) must be assessed by qualified legal counsel."
    )
    verification_instructions: str = (
        "Verify this manifest using the AURA verification tool: "
        "python -m aura.evidence.verify --manifest manifest.json --pubkey signing.pub"
    )

    model_config = {"use_enum_values": True}

    def add_custody_event(
        self,
        event_type: ChainOfCustodyEventType,
        actor_id: str,
        actor_role: str,
        description: str,
        component: str,
        **metadata: Any,
    ) -> None:
        self.chain_of_custody.append(
            ChainOfCustodyEvent(
                event_type=event_type,
                actor_id=actor_id,
                actor_role=actor_role,
                description=description,
                system_component=component,
                metadata=metadata,
            )
        )
