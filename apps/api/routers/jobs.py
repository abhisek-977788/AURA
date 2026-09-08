"""
Job status polling endpoints for AURA asynchronous processing pipeline.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import AnalysisRecordModel, JobModel, get_db
from apps.api.dependencies import get_job_or_404
from apps.api.redis_client import get_job_cache
from packages.security.auth import TokenData, get_current_user

router = APIRouter(prefix="/v1/jobs", tags=["Job Management"])


@router.get("/{job_id}")
async def get_job_status(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    user: TokenData = Depends(get_current_user),
):
    # 1. Try Redis cache for ultra-low latency polling
    cached = await get_job_cache(job_id)
    if cached:
        return cached

    # 2. Query Postgres
    job = await get_job_or_404(job_id, db)
    
    # Check if analysis record exists
    stmt = select(AnalysisRecordModel.id).where(AnalysisRecordModel.job_id == job_id)
    res = await db.execute(stmt)
    analysis_id = res.scalar_one_or_none()

    return {
        "job_id": job.id,
        "case_id": job.case_id,
        "event_id": job.event_id,
        "source_type": job.source_type,
        "status": job.status,
        "progress_pct": job.progress_pct,
        "error_message": job.error_message,
        "media_hash": job.media_hash,
        "analysis_id": analysis_id,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }
