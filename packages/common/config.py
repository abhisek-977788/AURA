"""
AURA configuration module.

Design decisions:
- All config values are read from environment variables or a .env file, never
  hard-coded in application code. Defaults are safe for local development but
  MUST be overridden in staging/production via secrets management (Vault,
  AWS Secrets Manager, Kubernetes Secrets, etc.).
- A single AuraSettings instance is created per process via lru_cache so that
  startup I/O (reading .env, validating types) only happens once.
- case_sensitive=False so operators can use any casing in their .env files.
- pydantic-settings v2 is used for automatic env-var coercion and validation;
  this catches misconfigured deployments at startup rather than at runtime.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class AuraSettings(BaseSettings):
    """
    Central configuration for all AURA services.

    Values are resolved in priority order:
      1. Explicit environment variables
      2. .env file in the working directory
      3. Defaults below

    Never commit secrets to .env files that are checked into source control.
    Use a secrets manager in production; mount secrets as environment variables
    or files, and point the relevant *_PATH settings at those mounted files.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",  # silently ignore unknown env vars to avoid deploy friction
    )

    # -------------------------------------------------------------------------
    # Infrastructure URLs
    # -------------------------------------------------------------------------
    DATABASE_URL: str = "postgresql+asyncpg://aura:aura@localhost:5432/aura"
    REDIS_URL: str = "redis://localhost:6379/0"

    # -------------------------------------------------------------------------
    # S3-compatible object storage (MinIO in dev, S3 / GCS in production)
    # -------------------------------------------------------------------------
    S3_ENDPOINT: str = "http://localhost:9000"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"
    S3_BUCKET: str = "aura-evidence"

    # -------------------------------------------------------------------------
    # Authentication / JWT
    # -------------------------------------------------------------------------
    SECRET_KEY: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60

    # -------------------------------------------------------------------------
    # Observability
    # -------------------------------------------------------------------------
    LOG_LEVEL: str = "INFO"

    # -------------------------------------------------------------------------
    # Application identity
    # -------------------------------------------------------------------------
    SOFTWARE_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"  # development | staging | production

    # -------------------------------------------------------------------------
    # Ingestion limits
    # -------------------------------------------------------------------------
    MAX_UPLOAD_SIZE_MB: int = 500

    # -------------------------------------------------------------------------
    # Evidence signing (Ed25519 key, mounted as a Kubernetes Secret or Docker
    # secret at runtime). In development you may generate a test key with:
    #   python -c "from packages.security.signing import generate_keypair; ..."
    # -------------------------------------------------------------------------
    EVIDENCE_SIGNING_KEY_PATH: str = "/run/secrets/signing.key"

    # -------------------------------------------------------------------------
    # Anti-malware (ClamAV sidecar)
    # -------------------------------------------------------------------------
    CLAMAV_HOST: str = "localhost"
    CLAMAV_PORT: int = 3310

    # -------------------------------------------------------------------------
    # Rate limiting
    # -------------------------------------------------------------------------
    RATE_LIMIT_PER_MINUTE: int = 60
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:8000,*"


@lru_cache(maxsize=1)
def get_settings() -> AuraSettings:
    """
    Return the singleton AuraSettings instance.

    Using lru_cache ensures:
    - .env is read exactly once at startup.
    - All services in the same process share the same config object.
    - Tests can call get_settings.cache_clear() to reset state between runs.
    """
    return AuraSettings()
