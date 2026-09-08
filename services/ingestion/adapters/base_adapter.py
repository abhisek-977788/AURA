"""
Abstract Base Class for Media Ingestion Adapters.
Every adapter accepts heterogeneous inputs and produces the unified MediaEvent.
"""

from __future__ import annotations

import abc
import hashlib
from pathlib import Path
from typing import Any

from packages.common.crypto import sha256_bytes
from packages.schemas.media_event import MediaEvent, PrivacyPolicy, ProcessingStatus, SourceType


class BaseIngestAdapter(abc.ABC):
    """Unified adapter interface."""

    def __init__(self, adapter_name: str, source_type: SourceType) -> None:
        self.adapter_name = adapter_name
        self.source_type = source_type

    @abc.abstractmethod
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
        """Processes raw media into standardized MediaEvent."""
        pass

    def compute_hash(self, data: bytes) -> str:
        return sha256_bytes(data)
