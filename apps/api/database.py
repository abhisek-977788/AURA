"""
Async SQLAlchemy models and database session provider for AURA.
PostgreSQL backend storing Cases, Jobs, Analyses, Evidence Packages, and Audit Logs.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship

from packages.common.config import get_settings


class Base(DeclarativeBase):
    pass


# ── Database Models ───────────────────────────────────────────────────────────


class CaseModel(Base):
    """Investigative or forensic case container."""
    __tablename__ = "cases"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(255), nullable=False)
    description = Column(Text, default="")
    status = Column(String(50), default="open")  # open, under_review, closed, archived
    created_by = Column(String(100), nullable=False, default="operator")
    consent_or_authorization_reference = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    jobs = relationship("JobModel", back_populates="case", cascade="all, delete-orphan")
    evidence_packages = relationship("EvidencePackageModel", back_populates="case", cascade="all, delete-orphan")


class JobModel(Base):
    """Asynchronous media processing job tracking."""
    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id = Column(String(36), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    event_id = Column(String(36), nullable=False, unique=True, index=True)
    source_type = Column(String(50), nullable=False)  # photo, recorded_video, recorded_audio, live_audio, live_video
    status = Column(String(50), default="queued")    # queued, processing, complete, failed
    progress_pct = Column(Float, default=0.0)
    error_message = Column(Text, nullable=True)
    
    media_hash = Column(String(64), nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime(timezone=True), nullable=True)

    case = relationship("CaseModel", back_populates="jobs")
    analysis = relationship("AnalysisRecordModel", uselist=False, back_populates="job", cascade="all, delete-orphan")


class AnalysisRecordModel(Base):
    """Persisted analysis and score-fusion result."""
    __tablename__ = "analysis_records"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String(36), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    event_id = Column(String(36), nullable=False, index=True)
    
    decision = Column(String(50), nullable=False)  # high_risk, medium_risk, low_risk, inconclusive
    synthetic_probability = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    requires_human_review = Column(Boolean, default=True)
    review_status = Column(String(50), default="pending")  # pending, in_review, reviewed, escalated
    analyst_conclusion = Column(Text, nullable=True)
    
    result_json = Column(JSON, nullable=False)  # Full AnalysisResult schema serialized
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    job = relationship("JobModel", back_populates="analysis")


class EvidencePackageModel(Base):
    """Exported cryptographically signed forensic evidence package."""
    __tablename__ = "evidence_packages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id = Column(String(36), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    analysis_id = Column(String(36), nullable=False, index=True)
    manifest_id = Column(String(36), nullable=False, unique=True)
    
    storage_key = Column(String(512), nullable=False)  # S3 zip location
    signature_algorithm = Column(String(50), default="Ed25519")
    manifest_signature = Column(Text, nullable=False)
    original_media_hash = Column(String(64), nullable=False)
    
    exported_by = Column(String(100), nullable=False)
    exported_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    case = relationship("CaseModel", back_populates="evidence_packages")


class AuditLogModel(Base):
    """Immutable audit trail of all operations and decisions."""
    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    event_type = Column(String(100), nullable=False, index=True)
    actor_id = Column(String(100), nullable=False)
    actor_role = Column(String(50), nullable=False)
    resource_type = Column(String(50), nullable=False)
    resource_id = Column(String(36), nullable=False)
    
    action = Column(String(255), nullable=False)
    details_json = Column(JSON, default=dict)
    ip_address = Column(String(45), nullable=True)
    request_id = Column(String(64), nullable=True)
    hash_chain = Column(String(64), nullable=True)  # SHA256(prev_hash + entry) for tamper-evidence


# ── Database Session Factory ──────────────────────────────────────────────────

_engine = None
_async_session_maker = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.DATABASE_URL,
            echo=False,
            future=True,
            pool_pre_ping=True,
        )
    return _engine


def get_session_maker():
    global _async_session_maker
    if _async_session_maker is None:
        engine = get_engine()
        _async_session_maker = async_sessionmaker(
            engine, expire_on_commit=False, class_=AsyncSession
        )
    return _async_session_maker


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    session_factory = get_session_maker()
    async with session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Initializes database tables (used in development or tests)."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
