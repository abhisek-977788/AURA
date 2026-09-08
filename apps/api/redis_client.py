"""
Redis Streams and PubSub client for AURA event-driven pipeline.
"""

from __future__ import annotations

import json
from typing import Any

import redis.asyncio as aioredis

from packages.common.config import get_settings
from packages.common.logging import get_logger
from packages.schemas.media_event import MediaEvent

logger = get_logger("api.redis")

_redis_client: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=3.0,
        )
    return _redis_client


async def publish_media_event(event: MediaEvent, stream_name: str = "aura:ingestion") -> str:
    """Serializes MediaEvent and appends (XADD) to Redis Stream."""
    r = await get_redis()
    payload = {"event_json": event.model_dump_json()}
    entry_id = await r.xadd(stream_name, payload)
    logger.info("Published MediaEvent to stream", stream=stream_name, event_id=event.event_id, entry_id=entry_id)
    return entry_id


async def set_job_cache(job_id: str, data: dict[str, Any], ttl_seconds: int = 86400) -> None:
    try:
        r = await get_redis()
        key = f"aura:job:{job_id}"
        await r.set(key, json.dumps(data, default=str), ex=ttl_seconds)
    except Exception as err:
        logger.warning("Failed to cache job in Redis", job_id=job_id, error=str(err))


async def get_job_cache(job_id: str) -> dict[str, Any] | None:
    try:
        r = await get_redis()
        key = f"aura:job:{job_id}"
        val = await r.get(key)
        return json.loads(val) if val else None
    except Exception:
        return None
