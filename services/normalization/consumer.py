"""
Normalization Service Consumer.
Listens to 'aura:normalization', runs Video and Audio normalizers,
and forwards normalized MediaEvent to 'aura:quality'.
"""

from __future__ import annotations

import asyncio
import os
import redis.asyncio as aioredis

from packages.common.config import get_settings
from packages.common.logging import configure_logging, get_logger
from packages.schemas.media_event import MediaEvent, SourceType
from services.normalization.video_normalizer import VideoNormalizer
from services.normalization.audio_normalizer import AudioNormalizer

logger = get_logger("service.normalization.consumer")


async def run_normalization_consumer():
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)
    logger.info("Starting AURA Normalization Consumer", version=settings.SOFTWARE_VERSION)

    r = aioredis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
    group_name = "aura_normalization_group"
    consumer_name = f"norm_worker_{os.getpid()}"

    video_norm = VideoNormalizer()
    audio_norm = AudioNormalizer()

    try:
        await r.xgroup_create("aura:normalization", group_name, id="0", mkstream=True)
    except Exception:
        pass

    while True:
        try:
            entries = await r.xreadgroup(
                groupname=group_name,
                consumername=consumer_name,
                streams={"aura:normalization": ">"},
                count=5,
                block=2000,
            )
            if not entries:
                await asyncio.sleep(0.1)
                continue

            for stream, messages in entries:
                for msg_id, data in messages:
                    event_json = data.get("event_json")
                    if not event_json:
                        await r.xack("aura:normalization", group_name, msg_id)
                        continue

                    event = MediaEvent.model_validate_json(event_json)

                    # 1. Normalize based on source type
                    if event.source_type == SourceType.RECORDED_VIDEO:
                        event = await video_norm.normalize(event)
                        # Normalize demuxed audio if produced
                        event = await audio_norm.normalize(event)
                    elif event.source_type == SourceType.RECORDED_AUDIO:
                        event = await audio_norm.normalize(event)

                    # 2. Forward to Quality Assessment Queue
                    await r.xadd("aura:quality", {"event_json": event.model_dump_json()})
                    await r.xack("aura:normalization", group_name, msg_id)
                    logger.info("Forwarded event to quality assessment", event_id=event.event_id)

        except asyncio.CancelledError:
            break
        except Exception as err:
            logger.error("Error in normalization consumer", error=str(err))
            await asyncio.sleep(1.0)


if __name__ == "__main__":
    asyncio.run(run_normalization_consumer())
