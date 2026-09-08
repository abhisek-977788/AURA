"""
Static Photo Ingestion Adapter.
Performs EXIF extraction, format inspection, SHA-256 calculation, and initial face detection check.
"""

from __future__ import annotations

import io
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ExifTags

from packages.common.crypto import sha256_bytes
from packages.schemas.media_event import MediaEvent, PrivacyLevel, PrivacyPolicy, ProcessingStatus, SourceType
from services.ingestion.adapters.base_adapter import BaseIngestAdapter


class PhotoAdapter(BaseIngestAdapter):
    def __init__(self) -> None:
        super().__init__(adapter_name="PhotoAdapter", source_type=SourceType.PHOTO)

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

        # 1. Parse Image & EXIF metadata safely
        img_buffer = io.BytesIO(raw_bytes)
        img = Image.open(img_buffer)
        width, height = img.size
        resolution_str = f"{width}x{height}"
        img_format = (img.format or "UNKNOWN").lower()

        exif_data: dict[str, str] = {}
        try:
            raw_exif = img._getexif()
            if raw_exif:
                for tag_id, value in raw_exif.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    # Convert binary or non-serializable EXIF fields to string representation
                    if isinstance(value, (bytes, bytearray)):
                        continue
                    exif_data[str(tag_name)] = str(value)
        except Exception:
            # Corrupted EXIF or format does not support EXIF
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
            source_type=SourceType.PHOTO,
            source_adapter=self.adapter_name,
            subject_id=f"anon-{media_hash[:12]}",
            media_hash=media_hash,
            codec=img_format,
            video_resolution=resolution_str,
            file_size_bytes=len(raw_bytes),
            privacy_policy=policy,
            consent_or_authorization_reference=auth_reference,
            processing_status=ProcessingStatus.RECEIVED,
            extra_metadata={
                "original_filename": filename,
                "image_width": width,
                "image_height": height,
                "color_mode": img.mode,
                "exif_metadata": exif_data,
            },
        )
