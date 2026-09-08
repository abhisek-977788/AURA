"""
Audit Log Service Worker.
Consumes audit entries from 'aura:audit_events' and persists to PostgreSQL.
"""

from __future__ import annotations

import asyncio
import json
import os
import redis.asyncio as aioredis

from apps.api.database import AuditLogModel, get_session_maker
from packages.common.config import get_settings
from packages.common.logging import configure_logging, get_logger
from services.audit.logger import AuditChainLogger

logger = get_logger("service.audit.worker")


async def run_audit_worker():
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)
    logger.info("Starting AURA Audit Worker", version=settings.SOFTWARE_VERSION)

    r = aioredis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
    group_name = "aura_audit_group"
    consumer_name = f"audit_worker_{os.getpid()}"

    chain_logger = AuditChainLogger()
    session_factory = get_session_maker()

    try:
        await r.xgroup_create("aura:audit_events", group_name, id="0", mkstream=True)
    except Exception:
        pass

    while True:
        try:
            entries = await r.xreadgroup(
                groupname=group_name,
                consumername=consumer_name,
                streams={"aura:audit_events": ">"},
                count=10,
                block=2000,
            )
            if not entries:
                await asyncio.sleep(0.1)
                continue

            for stream, messages in entries:
                for msg_id, data in messages:
                    entry = chain_logger.create_audit_entry(
                        event_type=data.get("event_type", "operation"),
                        actor_id=data.get("actor_id", "system"),
                        actor_role=data.get("actor_role", "system"),
                        resource_type=data.get("resource_type", "media"),
                        resource_id=data.get("resource_id", "unknown"),
                        action=data.get("action", "processed"),
                        details=json.loads(data.get("details_json", "{}")),
                    )

                    async with session_factory() as session:
                        try:
                            log_rec = AuditLogModel(**entry)
                            session.add(log_rec)
                            await session.commit()
                        except Exception as e:
                            logger.error("Failed to commit audit entry", error=str(e))

                    await r.xack("aura:audit_events", group_name, msg_id)

        except asyncio.CancelledError:
            break
        except Exception as err:
            logger.error("Error in audit worker loop", error=str(err))
            await asyncio.sleep(1.0)


if __name__ == "__main__":
    asyncio.run(run_audit_worker())
