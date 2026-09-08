"""
AURA API Gateway — Main Application Entrypoint.
Exposes public REST interfaces for media ingestion, job polling, forensics review,
case management, evidence verification, and live sessions.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from apps.api.database import init_db
from apps.api.routers import analysis, cases, evidence, health, jobs, live, media
from packages.common.config import get_settings
from packages.common.logging import configure_logging, get_logger

logger = get_logger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)
    logger.info(
        "Starting AURA API Gateway",
        version=settings.SOFTWARE_VERSION,
        environment=settings.ENVIRONMENT,
    )
    # Ensure database tables exist (development/self-healing)
    try:
        await init_db()
        logger.info("Database schema verified")
    except Exception as e:
        logger.warning("Could not auto-migrate DB tables on startup", error=str(e))

    yield

    logger.info("Shutting down AURA API Gateway")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="AURA — Audio-Visual Unnatural Representation Analyzer",
        description=(
            "Privacy-preserving, real-time and batch deepfake detection platform for "
            "authorized forensic media analysis, cybercrime investigations, and fraud prevention.\n\n"
            "**LEGAL DISCLAIMER**: Automated detections are statistical estimates and "
            "do NOT constitute court-admissible evidence without qualified forensic examiner review."
        ),
        version=settings.SOFTWARE_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS
    origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins if origins else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request ID and timing middleware
    @app.middleware("http")
    async def request_tracing_middleware(request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        start_time = time.perf_counter()
        
        response: Response = await call_next(request)
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
        
        # Avoid logging health checks in production to keep logs clean
        if request.url.path != "/health":
            logger.info(
                "HTTP request processed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
                request_id=req_id,
            )
        return response

    # Global Exception Handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation failed",
                "details": exc.errors(),
                "path": request.url.path,
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        err_id = str(uuid.uuid4())
        logger.error("Unhandled internal exception", error=str(exc), error_id=err_id, path=request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error occurred",
                "error_id": err_id,
                "path": request.url.path,
            },
        )

    # Register Routers
    app.include_router(health.router)
    app.include_router(media.router)
    app.include_router(jobs.router)
    app.include_router(analysis.router)
    app.include_router(cases.router)
    app.include_router(evidence.router)
    app.include_router(live.router)

    return app


app = create_app()
