"""
Media ingestion endpoints for Static Photos, Recorded Audio, and Recorded Video.
Validates MIME types, checks size limits, runs malware/polyglot detection,
persists Job, and publishes MediaEvent to Redis Streams.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import CaseModel, JobModel, get_db
from apps.api.dependencies import get_case_or_404
from apps.api.redis_client import publish_media_event, set_job_cache
from packages.common.config import get_settings
from packages.common.crypto import sha256_bytes
from packages.common.logging import get_logger
from packages.schemas.media_event import (
    MediaEvent,
    PrivacyLevel,
    PrivacyPolicy,
    ProcessingStatus,
    SourceType,
)
from packages.security.auth import TokenData, get_current_user
from packages.security.malware_scan import scan_bytes
from packages.security.rate_limit import rate_limit_dependency

logger = get_logger("api.media")
router = APIRouter(prefix="/v1/media", tags=["Media Ingestion"])

# Temporary ingest scratch folder for workers
INGEST_BUFFER_DIR = Path(tempfile.gettempdir()) / "aura_ingest"
INGEST_BUFFER_DIR.mkdir(parents=True, exist_ok=True)


async def _process_upload(
    file: UploadFile,
    case: CaseModel,
    source_type: SourceType,
    db: AsyncSession,
    user: TokenData,
    max_mb: int,
    allowed_mimes: set[str],
) -> dict:
    settings = get_settings()

    # 1. Read bytes and check size
    content = await file.read()
    file_size_bytes = len(content)
    max_bytes = max_mb * 1024 * 1024
    if file_size_bytes > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {max_mb} MB",
        )
    if file_size_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    # 2. Validate MIME type
    content_type = (file.content_type or "").lower()
    if not any(content_type.startswith(m) or content_type == m for m in allowed_mimes):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type '{content_type}'. Allowed: {sorted(list(allowed_mimes))}",
        )

    # 3. Malware & Polyglot Scan
    scan_res = await scan_bytes(content, filename=file.filename or "unknown")
    if not scan_res.clean:
        logger.error("Malware detected on upload", threat=scan_res.threat_name, user=user.subject)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Security scan rejected upload: detected {scan_res.threat_name}",
        )

    # 4. SHA-256 Hashing
    media_hash = sha256_bytes(content)

    # 5. Persist file to local staging buffer (ephemeral, deleted after pipeline)
    event_id = str(uuid.uuid4())
    ext = Path(file.filename or "").suffix or ".bin"
    buffered_path = INGEST_BUFFER_DIR / f"{event_id}{ext}"
    buffered_path.write_bytes(content)

    # 6. Create Job record in DB
    job_id = str(uuid.uuid4())
    job = JobModel(
        id=job_id,
        case_id=case.id,
        event_id=event_id,
        source_type=source_type.value,
        status="queued",
        progress_pct=5.0,
        media_hash=media_hash,
        file_size_bytes=file_size_bytes,
    )
    db.add(job)
    await db.commit()

    # 7. Construct MediaEvent
    event = MediaEvent(
        schema_version="1.0",
        event_id=event_id,
        case_id=case.id,
        job_id=job_id,
        source_type=source_type,
        source_adapter=f"{source_type.value.capitalize()}Adapter",
        subject_id=f"anon-{media_hash[:12]}",
        media_hash=media_hash,
        storage_path=str(buffered_path),
        file_size_bytes=file_size_bytes,
        consent_or_authorization_reference=case.consent_or_authorization_reference,
        privacy_policy=PrivacyPolicy(
            level=PrivacyLevel.TRANSIENT,
            retention_hours=24,
            pseudonymous_only=True,
        ),
        software_version=settings.SOFTWARE_VERSION,
        processing_status=ProcessingStatus.QUEUED,
        extra_metadata={
            "original_filename": file.filename,
            "mime_type": content_type,
            "staged_file_path": str(buffered_path),
        },
    )

    # 8. Publish to Redis stream & cache job
    await publish_media_event(event, stream_name="aura:ingestion")
    await set_job_cache(
        job_id,
        {
            "job_id": job_id,
            "case_id": case.id,
            "event_id": event_id,
            "status": "queued",
            "progress_pct": 5.0,
            "media_hash": media_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    return {
        "job_id": job_id,
        "event_id": event_id,
        "case_id": case.id,
        "source_type": source_type.value,
        "media_hash": media_hash,
        "status": "queued",
        "message": "Media accepted for deepfake forensic analysis",
    }


@router.post("/photos", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(rate_limit_dependency)])
async def upload_photo(
    case_id: str = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user: TokenData = Depends(get_current_user),
    case: CaseModel = Depends(get_case_or_404),
):
    return await _process_upload(
        file=file,
        case=case,
        source_type=SourceType.PHOTO,
        db=db,
        user=user,
        max_mb=50,
        allowed_mimes={"image/jpeg", "image/png", "image/webp", "image/tiff", "image/"},
    )


@router.post("/audio", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(rate_limit_dependency)])
async def upload_audio(
    case_id: str = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user: TokenData = Depends(get_current_user),
    case: CaseModel = Depends(get_case_or_404),
):
    return await _process_upload(
        file=file,
        case=case,
        source_type=SourceType.RECORDED_AUDIO,
        db=db,
        user=user,
        max_mb=100,
        allowed_mimes={"audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/flac", "audio/ogg", "audio/"},
    )


@router.post("/video", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(rate_limit_dependency)])
async def upload_video(
    case_id: str = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user: TokenData = Depends(get_current_user),
    case: CaseModel = Depends(get_case_or_404),
):
    return await _process_upload(
        file=file,
        case=case,
        source_type=SourceType.RECORDED_VIDEO,
        db=db,
        user=user,
        max_mb=500,
        allowed_mimes={"video/mp4", "video/x-matroska", "video/quicktime", "video/webm", "video/avi", "video/"},
    )
