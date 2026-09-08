"""
Alerts Service Worker.
Consumes from 'aura:analysis_complete', triggers alert policy, throttles flapping,
and enqueues automated evidence packaging if High Risk is determined.
"""

from __future__ import annotations

import asyncio
import os
import redis.asyncio as aioredis

from packages.common.config import get_settings
from packages.common.logging import configure_logging, get_logger
from packages.schemas.alert import AlertAction, AlertEvent, AlertLevel
from packages.schemas.analysis_result import AnalysisResult
from services.alerts.policy_engine import AlertPolicyEngine
from services.alerts.throttle import AlertThrottle

logger = get_logger("service.alerts.worker")


async def run_alerts_worker():
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)
    logger.info("Starting AURA Alerts Worker", version=settings.SOFTWARE_VERSION)

    r = aioredis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
    group_name = "aura_alerts_group"
    consumer_name = f"alerts_worker_{os.getpid()}"

    policy_engine = AlertPolicyEngine()
    throttle = AlertThrottle()

    try:
        await r.xgroup_create("aura:analysis_complete", group_name, id="0", mkstream=True)
    except Exception:
        pass

    while True:
        try:
            entries = await r.xreadgroup(
                groupname=group_name,
                consumername=consumer_name,
                streams={"aura:analysis_complete": ">"},
                count=5,
                block=2000,
            )
            if not entries:
                await asyncio.sleep(0.1)
                continue

            for stream, messages in entries:
                for msg_id, data in messages:
                    res_json = data.get("result_json")
                    if not res_json:
                        await r.xack("aura:analysis_complete", group_name, msg_id)
                        continue

                    analysis = AnalysisResult.model_validate_json(res_json)
                    alert: AlertEvent = policy_engine.evaluate(analysis)

                    # Check flapping / throttle
                    key = f"{alert.case_id or 'global'}:{alert.level.value}"
                    if not throttle.should_suppress(key):
                        logger.info(
                            "Triggered alert policy",
                            level=alert.level,
                            analysis_id=alert.analysis_id,
                            probability=alert.synthetic_media_probability,
                        )

                        # High Risk policy trigger: Auto-enqueue evidence packaging
                        if AlertAction.GENERATE_EVIDENCE_PACKAGE in alert.actions and alert.case_id:
                            req_payload = {
                                "case_id": alert.case_id,
                                "analysis_id": alert.analysis_id,
                                "actor_id": "system_policy_engine",
                                "actor_role": "system",
                            }
                            await r.xadd("aura:evidence_requests", {"request": str(req_payload)})
                            logger.info("Auto-enqueued evidence packaging for high-risk detection", case_id=alert.case_id)

                    await r.xack("aura:analysis_complete", group_name, msg_id)

        except asyncio.CancelledError:
            break
        except Exception as err:
            logger.error("Error in alerts worker loop", error=str(err))
            await asyncio.sleep(1.0)


if __name__ == "__main__":
    asyncio.run(run_alerts_worker())
