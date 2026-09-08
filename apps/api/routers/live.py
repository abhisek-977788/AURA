"""
Live session ingestion endpoints for WebRTC browser streams and SIP/RTP media relays.
Enforces ephemeral retention: NO raw media is stored by default unless high-threat evidence capture is triggered.
"""

from __future__ import annotations

import uuid
from typing import Optional
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from apps.api.redis_client import get_redis
from packages.security.auth import TokenData, get_current_user

router = APIRouter(prefix="/v1/live-sessions", tags=["Live Stream Ingestion"])


class CreateLiveSessionRequest(BaseModel):
    case_id: str
    source_type: str = Field(description="'live_video' or 'live_audio'")
    consent_or_authorization_reference: str = Field(description="Statutory authorization or consent token")


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_live_session(
    req: CreateLiveSessionRequest,
    user: TokenData = Depends(get_current_user),
):
    session_id = str(uuid.uuid4())
    r = await get_redis()
    
    session_data = {
        "session_id": session_id,
        "case_id": req.case_id,
        "source_type": req.source_type,
        "auth_ref": req.consent_or_authorization_reference,
        "created_by": user.subject,
        "status": "active",
    }
    await r.hset(f"aura:live:{session_id}", mapping=session_data)
    await r.expire(f"aura:live:{session_id}", 7200)

    return {
        "session_id": session_id,
        "status": "active",
        "privacy_mode": "ephemeral_no_raw_storage",
        "target_latency_ms": 150,
    }


@router.post("/{session_id}/events")
async def ingest_live_event_chunk(
    session_id: str,
    request: Request,
    user: TokenData = Depends(get_current_user),
):
    """Receives an audio or video chunk/frame from client-side WebAssembly or SBC relay."""
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty event chunk")

    r = await get_redis()
    active = await r.exists(f"aura:live:{session_id}")
    if not active:
        raise HTTPException(status_code=404, detail="Live session not found or expired")

    # Dispatch to live sliding-window stream
    await r.xadd(
        f"aura:live:stream:{session_id}",
        {"chunk_size": str(len(body)), "actor": user.subject},
        maxlen=100,  # Cap sliding window in memory to prevent accumulation
    )

    return {"status": "accepted", "session_id": session_id}


@router.delete("/{session_id}")
async def terminate_live_session(
    session_id: str,
    user: TokenData = Depends(get_current_user),
):
    r = await get_redis()
    await r.delete(f"aura:live:{session_id}")
    await r.delete(f"aura:live:stream:{session_id}")
    return {"status": "terminated", "session_id": session_id}
