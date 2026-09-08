"""
Video Inference Service Worker.
Listens to 'aura:video_inference', runs video models, and publishes results to 'aura:video_results'.
"""

from __future__ import annotations

import asyncio
import os
import redis.asyncio as aioredis

from packages.common.config import get_settings
from packages.common.logging import configure_logging, get_logger
from packages.schemas.media_event import MediaEvent
from services.video-inference.pipeline import VideoInferencePipeline

logger = get_logger("service.video.worker")


async def run_video_worker():
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)
    logger.info("Starting AURA Video Inference Worker", version=settings.SOFTWARE_VERSION)

    r = aioredis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
    group_name = "aura_video_infer_group"
    consumer_name = f"video_worker_{os.getpid()}"

    pipeline = VideoInferencePipeline()

    try:
        await r.xgroup_create("aura:video_inference", group_name, id="0", mkstream=True)
    except Exception:
        pass

    while True:
        try:
            entries = await r.xreadgroup(
                groupname=group_name,
                consumername=consumer_name,
                streams={"aura:video_inference": ">"},
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
                        await r.xack("aura:video_inference", group_name, msg_id)
                        continue

                    event = MediaEvent.model_validate_json(event_json)
                    logger.info("Running video inference", event_id=event.event_id)

                    modality_res, blink_res = await pipeline.run(event)

                    await r.xadd(
                        "aura:video_results",
                        {
                            "event_id": event.event_id,
                            "case_id": event.case_id or "",
                            "job_id": event.job_id or "",
                            "modality_json": modality_res.model_dump_json(),
                            "blink_json": blink_res.model_dump_json(),
                        },
                    )
                    await r.xack("aura:video_inference", group_name, msg_id)
                    logger.info("Video inference completed", event_id=event.event_id, score=modality_res.ensemble_score)

        except asyncio.CancelledError:
            break
        except Exception as err:
            logger.error("Error in video inference worker loop", error=str(err))
            await asyncio.sleep(1.0)


if __name__ == "__main__":
    asyncio.run(run_video_worker())
