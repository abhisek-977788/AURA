"""
Quality Assessment Service Consumer.
Listens to 'aura:quality', runs QualityAssessor, and gates media:
- If quality is sufficient: forwards to inference queues ('aura:audio_inference', 'aura:video_inference')
- If quality is insufficient: short-circuits directly to 'aura:analysis_complete' with decision="inconclusive"
"""

from __future__ import annotations

import asyncio
import os
import redis.asyncio as aioredis

from packages.common.config import get_settings
from packages.common.logging import configure_logging, get_logger
from packages.schemas.analysis_result import AnalysisResult
from packages.schemas.media_event import MediaEvent, SourceType
from services.quality.assessor import QualityAssessor

logger = get_logger("service.quality.consumer")


async def run_quality_consumer():
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)
    logger.info("Starting AURA Quality Consumer", version=settings.SOFTWARE_VERSION)

    r = aioredis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
    group_name = "aura_quality_group"
    consumer_name = f"quality_worker_{os.getpid()}"

    assessor = QualityAssessor()

    try:
        await r.xgroup_create("aura:quality", group_name, id="0", mkstream=True)
    except Exception:
        pass

    while True:
        try:
            entries = await r.xreadgroup(
                groupname=group_name,
                consumername=consumer_name,
                streams={"aura:quality": ">"},
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
                        await r.xack("aura:quality", group_name, msg_id)
                        continue

                    event = MediaEvent.model_validate_json(event_json)
                    event, quality_report = await assessor.assess(event)

                    if not quality_report.is_sufficient:
                        # Short-circuit: Do NOT force inference on degraded signals
                        logger.warning(
                            "Media failed quality gate. Generating inconclusive analysis",
                            event_id=event.event_id,
                            reasons=quality_report.insufficiency_reasons,
                        )
                        inconclusive_result = AnalysisResult.inconclusive(
                            event_id=event.event_id,
                            reason="; ".join(quality_report.insufficiency_reasons),
                            quality_report=quality_report,
                        )
                        inconclusive_result.case_id = event.case_id
                        inconclusive_result.job_id = event.job_id
                        await r.xadd("aura:analysis_complete", {"result_json": inconclusive_result.model_dump_json()})
                        await r.xack("aura:quality", group_name, msg_id)
                        continue

                    # Signal is sufficient: Route to modality inference queues
                    has_audio = event.source_type in (SourceType.RECORDED_AUDIO, SourceType.RECORDED_VIDEO, SourceType.LIVE_AUDIO)
                    has_video = event.source_type in (SourceType.RECORDED_VIDEO, SourceType.PHOTO, SourceType.LIVE_VIDEO)

                    if has_audio:
                        await r.xadd("aura:audio_inference", {"event_json": event.model_dump_json()})
                    if has_video:
                        await r.xadd("aura:video_inference", {"event_json": event.model_dump_json()})

                    await r.xack("aura:quality", group_name, msg_id)
                    logger.info("Quality gate passed. Dispatched to inference queues", event_id=event.event_id)

        except asyncio.CancelledError:
            break
        except Exception as err:
            logger.error("Error in quality consumer loop", error=str(err))
            await asyncio.sleep(1.0)


if __name__ == "__main__":
    asyncio.run(run_quality_consumer())
