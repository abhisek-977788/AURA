"""
Case management endpoints.
"""

from __future__ import annotations

import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import CaseModel, JobModel, get_db
from apps.api.dependencies import get_case_or_404
from apps.api.redis_client import get_redis
from packages.security.auth import TokenData, get_current_user, require_role, UserRole

router = APIRouter(prefix="/v1/cases", tags=["Case Management"])


class CreateCaseRequest(BaseModel):
    title: str = Field(min_length=3, max_length=255)
    description: Optional[str] = ""
    consent_or_authorization_reference: Optional[str] = Field(
        default=None,
        description="Search warrant, victim consent ref, court order, or lawful interception authorization",
    )


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_case(
    req: CreateCaseRequest,
    db: AsyncSession = Depends(get_db),
    user: TokenData = Depends(get_current_user),
):
    case_id = str(uuid.uuid4())
    case = CaseModel(
        id=case_id,
        title=req.title,
        description=req.description or "",
        consent_or_authorization_reference=req.consent_or_authorization_reference,
        created_by=user.subject,
    )
    db.add(case)
    await db.commit()
    await db.refresh(case)

    return {
        "id": case.id,
        "title": case.title,
        "description": case.description,
        "status": case.status,
        "created_by": case.created_by,
        "created_at": case.created_at.isoformat() if case.created_at else None,
    }


@router.get("")
async def list_cases(
    db: AsyncSession = Depends(get_db),
    user: TokenData = Depends(get_current_user),
):
    stmt = select(CaseModel).order_by(CaseModel.created_at.desc()).limit(50)
    res = await db.execute(stmt)
    cases = res.scalars().all()
    
    # Filter if user is restricted to specific case IDs
    if user.role != UserRole.ADMIN and "*" not in user.case_ids:
        cases = [c for c in cases if c.id in user.case_ids]

    return {
        "items": [
            {
                "id": c.id,
                "title": c.title,
                "description": c.description,
                "status": c.status,
                "created_by": c.created_by,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in cases
        ]
    }


@router.get("/{case_id}")
async def get_case(
    case_id: str,
    case: CaseModel = Depends(get_case_or_404),
    db: AsyncSession = Depends(get_db),
    user: TokenData = Depends(get_current_user),
):
    stmt = select(JobModel).where(JobModel.case_id == case.id)
    res = await db.execute(stmt)
    jobs = res.scalars().all()

    return {
        "id": case.id,
        "title": case.title,
        "description": case.description,
        "status": case.status,
        "consent_reference": case.consent_or_authorization_reference,
        "created_by": case.created_by,
        "created_at": case.created_at.isoformat() if case.created_at else None,
        "jobs": [
            {
                "id": j.id,
                "event_id": j.event_id,
                "source_type": j.source_type,
                "status": j.status,
                "media_hash": j.media_hash,
            }
            for j in jobs
        ],
    }


@router.post("/{case_id}/evidence/export", status_code=status.HTTP_202_ACCEPTED)
async def trigger_evidence_export(
    case_id: str,
    analysis_id: str,
    case: CaseModel = Depends(get_case_or_404),
    db: AsyncSession = Depends(get_db),
    user: TokenData = Depends(get_current_user),
):
    """Enqueues cryptographic evidence package generation."""
    r = await get_redis()
    payload = {
        "case_id": case_id,
        "analysis_id": analysis_id,
        "actor_id": user.subject,
        "actor_role": user.role.value,
    }
    await r.xadd("aura:evidence_requests", {"request": str(payload)})

    return {
        "message": "Evidence export initiated",
        "case_id": case_id,
        "analysis_id": analysis_id,
        "status": "queued",
    }
