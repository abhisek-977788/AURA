"""
Redis Streams and PubSub client for AURA event-driven pipeline.
Includes in-memory fallback for standalone local development without external Redis.
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
_in_memory_cache: dict[str, str] = {}
_in_memory_streams: dict[str, list[dict[str, Any]]] = {}


async def get_redis() -> aioredis.Redis | None:
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        try:
            client = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=1.0,
            )
            await client.ping()
            _redis_client = client
        except Exception:
            _redis_client = None
    return _redis_client


async def publish_media_event(event: MediaEvent, stream_name: str = "aura:ingestion") -> str:
    """Serializes MediaEvent and appends (XADD) to Redis Stream, or in-memory fallback."""
    payload = {"event_json": event.model_dump_json()}
    try:
        r = await get_redis()
        if r is not None:
            entry_id = await r.xadd(stream_name, payload)
            logger.info("Published MediaEvent to stream", stream=stream_name, event_id=event.event_id, entry_id=entry_id)
            return entry_id
    except Exception:
        pass

    # In-memory fallback
    if stream_name not in _in_memory_streams:
        _in_memory_streams[stream_name] = []
    entry_id = f"mem-{len(_in_memory_streams[stream_name]) + 1}"
    _in_memory_streams[stream_name].append(payload)
    logger.info("Buffered MediaEvent in local stream", stream=stream_name, event_id=event.event_id, entry_id=entry_id)
    return entry_id


async def set_job_cache(job_id: str, data: dict[str, Any], ttl_seconds: int = 86400) -> None:
    val = json.dumps(data, default=str)
    _in_memory_cache[f"aura:job:{job_id}"] = val
    try:
        r = await get_redis()
        if r is not None:
            key = f"aura:job:{job_id}"
            await r.set(key, val, ex=ttl_seconds)
    except Exception:
        pass


async def get_job_cache(job_id: str) -> dict[str, Any] | None:
    try:
        r = await get_redis()
        if r is not None:
            key = f"aura:job:{job_id}"
            val = await r.get(key)
            if val:
                return json.loads(val)
    except Exception:
        pass

    val = _in_memory_cache.get(f"aura:job:{job_id}")
    return json.loads(val) if val else None
