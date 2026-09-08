"""
Redis-backed rate limiter for AURA API gateway.
Protects ingestion endpoints from Denial of Service and API flooding.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import HTTPException, Request, status
import redis.asyncio as aioredis

from packages.common.config import get_settings
from packages.common.logging import get_logger

logger = get_logger("security.rate_limit")


class RateLimiter:
    """Sliding-window or bucket rate limiter backed by Redis with local in-memory fallback."""

    def __init__(self, redis_url: str | None = None) -> None:
        self.redis_url = redis_url or get_settings().REDIS_URL
        self._redis: aioredis.Redis | None = None
        self._memory_cache: dict[str, list[float]] = {}

    async def _get_redis(self) -> aioredis.Redis | None:
        if self._redis is None:
            try:
                self._redis = aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=2.0,
                )
                await self._redis.ping()
            except Exception as err:
                logger.warning("Redis unavailable for rate limiting, falling back to in-memory limiter", error=str(err))
                self._redis = None
        return self._redis

    async def check(self, key: str, max_requests: int = 60, window_seconds: int = 60) -> bool:
        """Returns True if request is allowed, False if limit exceeded."""
        r = await self._get_redis()
        now = time.time()

        if r is not None:
            try:
                redis_key = f"aura:rate:{key}"
                pipe = r.pipeline()
                pipe.zremrangebyscore(redis_key, 0, now - window_seconds)
                pipe.zadd(redis_key, {str(now): now})
                pipe.zcard(redis_key)
                pipe.expire(redis_key, window_seconds + 5)
                results = await pipe.execute()
                count = results[2]
                return count <= max_requests
            except Exception as e:
                logger.warning("Redis rate limit check failed, using fallback", error=str(e))

        # In-memory fallback
        timestamps = self._memory_cache.get(key, [])
        cutoff = now - window_seconds
        valid_ts = [t for t in timestamps if t > cutoff]
        if len(valid_ts) >= max_requests:
            return False
        valid_ts.append(now)
        self._memory_cache[key] = valid_ts
        return True


_limiter = RateLimiter()


async def rate_limit_dependency(request: Request) -> None:
    settings = get_settings()
    client_ip = request.client.host if request.client else "unknown"
    path = request.url.path
    key = f"{client_ip}:{path}"

    allowed = await _limiter.check(
        key,
        max_requests=settings.RATE_LIMIT_PER_MINUTE,
        window_seconds=60,
    )
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please throttle requests.",
            headers={"Retry-After": "60"},
        )
