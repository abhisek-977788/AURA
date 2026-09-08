"""
Score Fusion Service Worker.
Consumes results from 'aura:audio_results' and 'aura:video_results',
aggregates per event_id, runs multimodal fusion, updates Job & Analysis in Postgres,
and publishes completed AnalysisResult to 'aura:analysis_complete'.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
import redis.asyncio as aioredis
from sqlalchemy import select, update

from apps.api.database import AnalysisRecordModel, JobModel, get_session_maker
from packages.common.config import get_settings
from packages.common.logging import configure_logging, get_logger
from packages.schemas.analysis_result import (
    AnalysisResult,
    BlinkAnalysis,
    ModalityResult,
    SyncResult,
)
from services.fusion.fusion_engine import FusionEngine
from services.fusion.syncnet_adapter import SyncNetAdapter

logger = get_logger("service.fusion.worker")


async def run_fusion_worker():
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)
    logger.info("Starting AURA Fusion Worker", version=settings.SOFTWARE_VERSION)

    r = aioredis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
    fusion_engine = FusionEngine()
    syncnet = SyncNetAdapter()
    session_factory = get_session_maker()

    # Track pending event states
    pending_events: dict[str, dict] = {}

    while True:
        try:
            # Read from both audio and video streams
            for stream_name in ("aura:audio_results", "aura:video_results"):
                entries = await r.xread({stream_name: "$"}, count=5, block=1000)
                if not entries:
                    continue

                for stream, messages in entries:
                    for msg_id, data in messages:
                        event_id = data.get("event_id")
                        case_id = data.get("case_id")
                        job_id = data.get("job_id")
                        if not event_id:
                            continue

                        if event_id not in pending_events:
                            pending_events[event_id] = {
                                "case_id": case_id,
                                "job_id": job_id,
                                "audio_res": None,
                                "video_res": None,
                                "blink_res": None,
                                "timestamp": datetime.now(timezone.utc),
                            }

                        if "result_json" in data:
                            pending_events[event_id]["audio_res"] = ModalityResult.model_validate_json(data["result_json"])
                        if "modality_json" in data:
                            pending_events[event_id]["video_res"] = ModalityResult.model_validate_json(data["modality_json"])
                        if "blink_json" in data:
                            pending_events[event_id]["blink_res"] = BlinkAnalysis.model_validate_json(data["blink_json"])

            # Check ready events
            to_remove = []
            for event_id, state in pending_events.items():
                has_audio = state["audio_res"] is not None
                has_video = state["video_res"] is not None

                # If we have both or single modality available after processing
                if has_audio or has_video:
                    sync_res = await syncnet.analyze(has_audio=has_audio, has_video=has_video)
                    
                    analysis: AnalysisResult = await fusion_engine.fuse(
                        event_id=event_id,
                        case_id=state["case_id"],
                        job_id=state["job_id"],
                        audio_result=state["audio_res"],
                        video_result=state["video_res"],
                        sync_result=sync_res,
                        blink_analysis=state["blink_res"],
                    )

                    # 1. Update Database
                    job_id = state["job_id"]
                    if job_id:
                        async with session_factory() as session:
                            try:
                                record = AnalysisRecordModel(
                                    id=analysis.analysis_id,
                                    job_id=job_id,
                                    event_id=event_id,
                                    decision=analysis.decision.value,
                                    synthetic_probability=analysis.synthetic_media_probability,
                                    confidence=analysis.confidence,
                                    requires_human_review=analysis.requires_human_review,
                                    review_status=analysis.review_status.value,
                                    result_json=analysis.model_dump(),
                                )
                                session.add(record)
                                await session.execute(
                                    update(JobModel)
                                    .where(JobModel.id == job_id)
                                    .values(status="complete", progress_pct=100.0, completed_at=datetime.now(timezone.utc))
                                )
                                await session.commit()
                            except Exception as e:
                                logger.error("Failed to save analysis in DB", error=str(e), event_id=event_id)

                    # 2. Publish to completed stream
                    await r.xadd("aura:analysis_complete", {"result_json": analysis.model_dump_json()})
                    logger.info("Fusion completed and published", event_id=event_id, decision=analysis.decision)
                    to_remove.append(event_id)

            for e_id in to_remove:
                pending_events.pop(e_id, None)

            await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            break
        except Exception as err:
            logger.error("Error in fusion worker loop", error=str(err))
            await asyncio.sleep(1.0)


if __name__ == "__main__":
    asyncio.run(run_fusion_worker())
