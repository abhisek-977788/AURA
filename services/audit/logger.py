"""
Audit Event Types and Immutable Tamper-Evident Audit Logger for AURA.
Uses SHA-256 hash chaining (blockchain-style ledger) to prevent undetectable log tampering.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from packages.common.crypto import sha256_string
from packages.common.logging import get_logger

logger = get_logger("audit.logger")


class AuditEventType(str, Enum):
    MEDIA_UPLOADED = "media_uploaded"
    JOB_CREATED = "job_created"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    ANALYSIS_COMPLETED = "analysis_completed"
    ANALYSIS_REVIEWED = "analysis_reviewed"
    ALERT_TRIGGERED = "alert_triggered"
    EVIDENCE_EXPORTED = "evidence_exported"
    EVIDENCE_VERIFIED = "evidence_verified"
    CASE_CREATED = "case_created"
    CASE_ACCESSED = "case_accessed"
    USER_LOGIN = "user_login"
    SECURITY_VIOLATION = "security_violation"


class AuditChainLogger:
    """Computes tamper-evident SHA-256 hash chaining across consecutive entries."""

    def __init__(self, initial_seed: str = "AURA-AUDIT-GENESIS-CHAIN-001") -> None:
        self.last_hash = sha256_string(initial_seed)

    def create_audit_entry(
        self,
        event_type: AuditEventType | str,
        actor_id: str,
        actor_role: str,
        resource_type: str,
        resource_id: str,
        action: str,
        details: dict[str, Any] | None = None,
        ip_address: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        entry_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        evt_type = event_type.value if isinstance(event_type, AuditEventType) else str(event_type)

        # Hash current content + previous entry hash
        raw_payload = f"{self.last_hash}:{entry_id}:{evt_type}:{actor_id}:{resource_id}:{action}:{now.isoformat()}"
        current_hash = sha256_string(raw_payload)
        self.last_hash = current_hash

        return {
            "id": entry_id,
            "timestamp": now,
            "event_type": evt_type,
            "actor_id": actor_id,
            "actor_role": actor_role,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "action": action,
            "details_json": details or {},
            "ip_address": ip_address,
            "request_id": request_id,
            "hash_chain": current_hash,
        }
