"""
Ingestion Service Worker.
Consumes newly uploaded MediaEvents from Redis stream 'aura:ingestion',
runs appropriate adapter, and publishes to 'aura:normalization'.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import redis.asyncio as aioredis

from packages.common.config import get_settings
from packages.common.logging import configure_logging, get_logger
from packages.schemas.media_event import MediaEvent, ProcessingStatus

logger = get_logger("service.ingestion.worker")


async def run_ingestion_worker():
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)
    logger.info("Starting AURA Ingestion Worker", version=settings.SOFTWARE_VERSION)

    r = aioredis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
    group_name = "aura_ingestion_group"
    consumer_name = f"ingest_worker_{os.getpid()}"

    try:
        await r.xgroup_create("aura:ingestion", group_name, id="0", mkstream=True)
    except Exception:
        pass  # Group already exists

    while True:
        try:
            entries = await r.xreadgroup(
                groupname=group_name,
                consumername=consumer_name,
                streams={"aura:ingestion": ">"},
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
                        await r.xack("aura:ingestion", group_name, msg_id)
                        continue

                    event = MediaEvent.model_validate_json(event_json)
                    logger.info("Ingesting media event", event_id=event.event_id, source=event.source_type)

                    # Mark event status and forward to normalization queue
                    updated_event = event.mark_status(ProcessingStatus.NORMALIZED)
                    await r.xadd("aura:normalization", {"event_json": updated_event.model_dump_json()})
                    await r.xack("aura:ingestion", group_name, msg_id)

        except asyncio.CancelledError:
            break
        except Exception as err:
            logger.error("Error in ingestion worker loop", error=str(err))
            await asyncio.sleep(1.0)


if __name__ == "__main__":
    asyncio.run(run_ingestion_worker())
