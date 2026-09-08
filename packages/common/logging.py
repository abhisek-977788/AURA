"""
AURA structured logging module.

Design decisions:
- structlog is used instead of the stdlib logging module because it produces
  machine-parseable JSON natively, supports context-local bound loggers, and
  plays well with async code without thread-local hacks.
- Every log line carries 'request_id' and 'component' fields so that log
  aggregators (Loki, Splunk, Elasticsearch) can correlate all events from a
  single inbound request and filter by service component without grepping.
- The stdlib integration (stdlib=True) ensures that third-party libraries
  (SQLAlchemy, aioboto3, httpx) that use the standard logging module also emit
  JSON through the same pipeline.
- In production, set LOG_LEVEL=WARNING to reduce I/O pressure; in development
  LOG_LEVEL=DEBUG gives full visibility into model inference steps.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

# ---------------------------------------------------------------------------
# Module-level sentinel to prevent configure_logging() from being called more
# than once per process (idempotency for test runners that import multiple
# modules).
# ---------------------------------------------------------------------------
_configured: bool = False


def configure_logging(log_level: str = "INFO") -> None:
    """
    Initialise structlog with a JSON renderer and stdlib bridge.

    Should be called once at application startup (e.g. in the FastAPI lifespan
    handler or service __main__ block). Safe to call multiple times — subsequent
    calls are no-ops.

    Args:
        log_level: One of DEBUG, INFO, WARNING, ERROR, CRITICAL. Case-insensitive.
    """
    global _configured
    if _configured:
        return

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure stdlib root logger so third-party lib logs flow through structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )

    shared_processors: list[Any] = [
        # Inject log level as a string
        structlog.stdlib.add_log_level,
        # Add ISO-8601 timestamp
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        # Render exception info as a structured dict rather than a traceback string
        structlog.processors.format_exc_info,
        # Ensure 'request_id' and 'component' are always present (defaulting to
        # sentinel values so downstream filters always have these fields to work with)
        _ensure_request_context,
        # Render stack info if present
        structlog.processors.StackInfoRenderer(),
    ]

    structlog.configure(
        processors=shared_processors
        + [
            # Bridge to stdlib so log aggregators receive one unified stream
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Attach JSON formatter to the stdlib handler
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    # Remove default handlers added by basicConfig before attaching our own
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)

    _configured = True


def _ensure_request_context(
    logger: Any,  # noqa: ANN401 – structlog processor signature
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """
    Structlog processor that guarantees 'request_id' and 'component' are always
    present in every log event dictionary.

    This avoids KeyError exceptions in downstream log processors and ensures
    that log aggregation dashboards can always filter on these fields even when
    a log statement was emitted outside a request context (e.g. during startup).
    """
    event_dict.setdefault("request_id", "none")
    event_dict.setdefault("component", "aura")
    return event_dict


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Return a bound structlog logger pre-populated with the component name.

    Usage::

        log = get_logger(__name__)
        log.info("model_loaded", model_version="v2.3", latency_ms=124)

    The 'component' field will appear in every log line emitted by this logger,
    enabling Loki / Kibana dashboard filters like ``component="services.ingest"``.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A structlog BoundLogger with 'component' pre-bound.
    """
    return structlog.get_logger(name).bind(component=name)
