"""
Audio Inference Service Worker.
Listens to 'aura:audio_inference', runs audio models, and publishes results to 'aura:audio_results'.
"""

from __future__ import annotations

import asyncio
import os
import redis.asyncio as aioredis

from packages.common.config import get_settings
from packages.common.logging import configure_logging, get_logger
from packages.schemas.media_event import MediaEvent
from services.audio-inference.pipeline import AudioInferencePipeline

logger = get_logger("service.audio.worker")


async def run_audio_worker():
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)
    logger.info("Starting AURA Audio Inference Worker", version=settings.SOFTWARE_VERSION)

    r = aioredis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
    group_name = "aura_audio_infer_group"
    consumer_name = f"audio_worker_{os.getpid()}"

    pipeline = AudioInferencePipeline()

    try:
        await r.xgroup_create("aura:audio_inference", group_name, id="0", mkstream=True)
    except Exception:
        pass

    while True:
        try:
            entries = await r.xreadgroup(
                groupname=group_name,
                consumername=consumer_name,
                streams={"aura:audio_inference": ">"},
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
                        await r.xack("aura:audio_inference", group_name, msg_id)
                        continue

                    event = MediaEvent.model_validate_json(event_json)
                    logger.info("Running audio inference", event_id=event.event_id)

                    result = await pipeline.run(event)

                    # Forward to fusion queue
                    await r.xadd(
                        "aura:audio_results",
                        {
                            "event_id": event.event_id,
                            "case_id": event.case_id or "",
                            "job_id": event.job_id or "",
                            "result_json": result.model_dump_json(),
                        },
                    )
                    await r.xack("aura:audio_inference", group_name, msg_id)
                    logger.info("Audio inference completed", event_id=event.event_id, score=result.ensemble_score)

        except asyncio.CancelledError:
            break
        except Exception as err:
            logger.error("Error in audio inference worker loop", error=str(err))
            await asyncio.sleep(1.0)


if __name__ == "__main__":
    asyncio.run(run_audio_worker())
