"""
Health and metrics endpoints.
"""

from __future__ import annotations

from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.database import get_db
from apps.api.redis_client import get_redis
from packages.common.config import get_settings

router = APIRouter(tags=["System"])


@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    settings = get_settings()
    status_report = {
        "status": "ok",
        "service": "aura-api",
        "version": settings.SOFTWARE_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dependencies": {},
    }

    # Test DB
    try:
        await db.execute(text("SELECT 1"))
        status_report["dependencies"]["postgres"] = "healthy"
    except Exception as e:
        status_report["dependencies"]["postgres"] = f"unhealthy: {str(e)}"
        status_report["status"] = "degraded"

    # Test Redis
    try:
        r = await get_redis()
        await r.ping()
        status_report["dependencies"]["redis"] = "healthy"
    except Exception as e:
        status_report["dependencies"]["redis"] = f"unhealthy: {str(e)}"
        status_report["status"] = "degraded"

    return status_report


@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    """Returns basic Prometheus metrics for Prometheus scrapers."""
    settings = get_settings()
    now_ts = datetime.now(timezone.utc).timestamp()
    metrics = [
        f"# HELP aura_service_info AURA Gateway build metadata",
        f"# TYPE aura_service_info gauge",
        f'aura_service_info{{version="{settings.SOFTWARE_VERSION}",env="{settings.ENVIRONMENT}"}} 1',
        f"# HELP aura_uptime_seconds Process alive timestamp",
        f"# TYPE aura_uptime_seconds counter",
        f"aura_uptime_seconds {now_ts}",
    ]
    return "\n".join(metrics) + "\n"
