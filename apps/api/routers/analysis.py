"""
Analysis and explanations retrieval endpoints.
Returns calibrated probabilities, modality breakdowns, quality gates, and explainability artifacts.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import AnalysisRecordModel, get_db
from apps.api.dependencies import get_analysis_or_404
from packages.schemas.analysis_result import AnalysisResult
from packages.security.auth import TokenData, get_current_user

router = APIRouter(prefix="/v1/analysis", tags=["Analysis & Forensics"])


@router.get("/{analysis_id}")
async def get_analysis(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
    user: TokenData = Depends(get_current_user),
):
    record = await get_analysis_or_404(analysis_id, db)
    return record.result_json


@router.get("/{analysis_id}/explanations")
async def get_analysis_explanations(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
    user: TokenData = Depends(get_current_user),
):
    record = await get_analysis_or_404(analysis_id, db)
    data = record.result_json

    return {
        "analysis_id": analysis_id,
        "gradcam_artifact_url": data.get("gradcam_artifact_path"),
        "shap_artifact_url": data.get("shap_artifact_path"),
        "explanation_artifacts": data.get("explanation_artifact_paths", []),
        "top_contributing_signals": data.get("top_contributing_signals", []),
        "limitations": data.get("limitations", []),
        "disclaimer": data.get("legal_disclaimer"),
    }
