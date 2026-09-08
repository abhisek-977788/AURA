"""
FastAPI route dependencies for Case boundary enforcement and entity lookups.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from apps.api.database import get_db, CaseModel, JobModel, AnalysisRecordModel
from packages.security.auth import TokenData, get_current_user, UserRole


async def get_case_or_404(
    case_id: str,
    db: AsyncSession = Depends(get_db),
    user: TokenData = Depends(get_current_user),
) -> CaseModel:
    stmt = select(CaseModel).where(CaseModel.id == case_id)
    result = await db.execute(stmt)
    case = result.scalar_one_or_none()
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case '{case_id}' not found",
        )
    # Check authorization boundary
    if user.role != UserRole.ADMIN and "*" not in user.case_ids and case_id not in user.case_ids:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied to case '{case_id}'",
        )
    return case


async def get_job_or_404(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> JobModel:
    stmt = select(JobModel).where(JobModel.id == job_id)
    result = await db.execute(stmt)
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found",
        )
    return job


async def get_analysis_or_404(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
) -> AnalysisRecordModel:
    stmt = select(AnalysisRecordModel).where(AnalysisRecordModel.id == analysis_id)
    result = await db.execute(stmt)
    record = result.scalar_one_or_none()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis '{analysis_id}' not found",
        )
    return record
