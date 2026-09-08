"""
AURA S3-compatible object storage client.

Design decisions:
- aioboto3 is used (not boto3) because AURA is fully async. Blocking S3 calls
  in an async FastAPI service would stall the event loop, causing all concurrent
  requests to queue behind a single upload.
- AuraStorage is an async context manager. This ensures that the underlying
  aiohttp session is properly closed after use, preventing resource leaks in
  long-running workers.
- Raw biometric media is NEVER stored by this client unless the caller has
  already applied the appropriate anonymisation/redaction pipeline. The storage
  layer itself is policy-agnostic; enforcement happens in the ingest service.
- Presigned URLs are used for evidence downloads. This avoids streaming large
  files through the AURA API tier and allows investigators to download directly
  from object storage with time-limited, audit-logged credentials.
- ensure_bucket() is idempotent and safe to call on every service startup,
  making local dev and CI environments self-bootstrapping.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path
from types import TracebackType
from typing import Any, Self

try:
    import aioboto3
    from botocore.config import Config
except ImportError:
    aioboto3 = None
    Config = None

from packages.common.config import get_settings
from packages.common.logging import get_logger

log = get_logger(__name__)


class AuraStorage:
    """
    Async S3-compatible storage client wrapping aioboto3.

    Usage::

        async with AuraStorage() as storage:
            url = await storage.upload_bytes(
                bucket="aura-evidence",
                key="cases/abc123/audio.wav",
                data=raw_bytes,
                content_type="audio/wav",
            )

    The client reads its endpoint, credentials, and bucket from AuraSettings
    so no configuration is needed at the call site.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._session: aioboto3.Session = aioboto3.Session(
            aws_access_key_id=self._settings.S3_ACCESS_KEY,
            aws_secret_access_key=self._settings.S3_SECRET_KEY,
        )
        # _client is populated in __aenter__
        self._client: Any = None
        self._ctx: Any = None

    async def __aenter__(self) -> Self:
        self._ctx = self._session.client(
            "s3",
            endpoint_url=self._settings.S3_ENDPOINT,
            config=Config(
                # Disable checksum validation for MinIO compatibility
                signature_version="s3v4",
            ),
        )
        self._client = await self._ctx.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._ctx is not None:
            await self._ctx.__aexit__(exc_type, exc_val, exc_tb)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def upload_bytes(
        self,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """
        Upload raw bytes to object storage.

        Args:
            bucket: Target bucket name.
            key: Object key (path within the bucket).
            data: Bytes to upload.
            content_type: MIME type of the object.

        Returns:
            Canonical S3 URL of the uploaded object,
            e.g. 's3://aura-evidence/cases/abc/audio.wav'.
        """
        await self._client.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        url = f"s3://{bucket}/{key}"
        log.info("storage.uploaded_bytes", bucket=bucket, key=key, size=len(data))
        return url

    async def upload_file(self, bucket: str, key: str, path: Path) -> str:
        """
        Upload a file from the local filesystem to object storage.

        Streams the file directly without loading it entirely into memory,
        making it safe for large video evidence files.

        Args:
            bucket: Target bucket name.
            key: Object key (path within the bucket).
            path: Local path to the file to upload.

        Returns:
            Canonical S3 URL of the uploaded object.
        """
        content_type, _ = mimetypes.guess_type(str(path))
        content_type = content_type or "application/octet-stream"

        await self._client.upload_file(
            Filename=str(path),
            Bucket=bucket,
            Key=key,
            ExtraArgs={"ContentType": content_type},
        )
        url = f"s3://{bucket}/{key}"
        log.info(
            "storage.uploaded_file",
            bucket=bucket,
            key=key,
            path=str(path),
            content_type=content_type,
        )
        return url

    async def download_bytes(self, bucket: str, key: str) -> bytes:
        """
        Download an object from storage and return its raw bytes.

        Args:
            bucket: Source bucket name.
            key: Object key.

        Returns:
            Raw bytes of the object body.

        Raises:
            botocore.exceptions.ClientError: If the object does not exist or
                access is denied.
        """
        response = await self._client.get_object(Bucket=bucket, Key=key)
        body: bytes = await response["Body"].read()
        log.info("storage.downloaded_bytes", bucket=bucket, key=key, size=len(body))
        return body

    async def get_presigned_url(
        self,
        bucket: str,
        key: str,
        expires: int = 3600,
    ) -> str:
        """
        Generate a time-limited presigned GET URL for an object.

        Presigned URLs allow investigators to download evidence directly from
        object storage without routing the payload through the AURA API tier,
        reducing bandwidth costs and API latency.

        Args:
            bucket: Bucket name.
            key: Object key.
            expires: URL expiry in seconds (default: 1 hour).

        Returns:
            HTTPS presigned URL string.
        """
        url: str = await self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires,
        )
        log.info(
            "storage.presigned_url_generated",
            bucket=bucket,
            key=key,
            expires_in=expires,
        )
        return url

    async def delete_object(self, bucket: str, key: str) -> None:
        """
        Permanently delete an object from storage.

        This operation is irreversible. In production, enable S3 Object Lock or
        MinIO WORM mode on the evidence bucket to prevent accidental or malicious
        deletion of court evidence.

        Args:
            bucket: Bucket name.
            key: Object key to delete.
        """
        await self._client.delete_object(Bucket=bucket, Key=key)
        log.warning("storage.object_deleted", bucket=bucket, key=key)

    async def ensure_bucket(self, bucket: str) -> None:
        """
        Create the bucket if it does not already exist.

        Idempotent — safe to call on every service startup. The
        BucketAlreadyOwnedByYou error is suppressed because it is the expected
        steady-state response in development and CI environments.

        Args:
            bucket: Bucket name to create.
        """
        try:
            await self._client.create_bucket(Bucket=bucket)
            log.info("storage.bucket_created", bucket=bucket)
        except self._client.exceptions.BucketAlreadyOwnedByYou:
            log.debug("storage.bucket_already_exists", bucket=bucket)
        except Exception as exc:
            # Re-raise unexpected errors (permissions, network failures)
            log.error("storage.bucket_ensure_failed", bucket=bucket, error=str(exc))
            raise
