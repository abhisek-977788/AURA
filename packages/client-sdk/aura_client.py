"""
AURA client-sdk — lightweight Python client for the AURA API.

Usage:
    from aura_client import AuraClient
    
    client = AuraClient(base_url="http://localhost:8000", api_key="...")
    
    # Upload a video
    job = await client.upload_video("case-123", open("video.mp4", "rb"))
    
    # Wait for completion
    analysis = await client.wait_for_analysis(job.job_id)
    
    # Export evidence
    evidence = await client.export_evidence(job.case_id, analysis.analysis_id)
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import IO, Any

import httpx


class AuraClientError(Exception):
    pass


class AuraClient:
    """
    Async AURA API client.

    All methods raise AuraClientError on API errors.
    Authentication uses a Bearer token (JWT).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._headers: dict[str, str] = {}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "AuraClient":
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers,
            timeout=self._timeout,
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()

    def _get_client(self) -> httpx.AsyncClient:
        if not self._client:
            raise RuntimeError("Use AuraClient as async context manager")
        return self._client

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        client = self._get_client()
        response = await client.request(method, path, **kwargs)
        if not response.is_success:
            raise AuraClientError(
                f"API error {response.status_code}: {response.text}"
            )
        return response.json()

    # ── Health ──────────────────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        return await self._request("GET", "/health")

    # ── Cases ───────────────────────────────────────────────────────────────

    async def create_case(
        self,
        title: str,
        description: str = "",
        consent_reference: str | None = None,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/v1/cases",
            json={
                "title": title,
                "description": description,
                "consent_or_authorization_reference": consent_reference,
            },
        )

    async def get_case(self, case_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/v1/cases/{case_id}")

    async def list_cases(self) -> list[dict[str, Any]]:
        result = await self._request("GET", "/v1/cases")
        return result.get("items", [])

    # ── Media uploads ────────────────────────────────────────────────────────

    async def upload_video(
        self,
        case_id: str,
        file: IO[bytes] | Path,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        return await self._upload("/v1/media/video", case_id, file, idempotency_key)

    async def upload_audio(
        self,
        case_id: str,
        file: IO[bytes] | Path,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        return await self._upload("/v1/media/audio", case_id, file, idempotency_key)

    async def upload_photo(
        self,
        case_id: str,
        file: IO[bytes] | Path,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        return await self._upload("/v1/media/photos", case_id, file, idempotency_key)

    async def _upload(
        self,
        endpoint: str,
        case_id: str,
        file: IO[bytes] | Path,
        idempotency_key: str | None,
    ) -> dict[str, Any]:
        if isinstance(file, Path):
            file_handle = open(file, "rb")
            filename = file.name
        else:
            file_handle = file
            filename = getattr(file, "name", "upload")

        headers: dict[str, str] = {}
        if idempotency_key:
            headers["X-Idempotency-Key"] = idempotency_key

        try:
            return await self._request(
                "POST",
                endpoint,
                files={"file": (filename, file_handle)},
                data={"case_id": case_id},
                headers=headers,
            )
        finally:
            if isinstance(file, Path):
                file_handle.close()

    # ── Jobs ─────────────────────────────────────────────────────────────────

    async def get_job(self, job_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/v1/jobs/{job_id}")

    async def wait_for_job(
        self,
        job_id: str,
        poll_interval: float = 2.0,
        timeout: float = 600.0,
    ) -> dict[str, Any]:
        """Poll job status until complete or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = await self.get_job(job_id)
            if job["status"] in ("complete", "failed"):
                return job
            await asyncio.sleep(poll_interval)
        raise AuraClientError(f"Job {job_id} did not complete within {timeout}s")

    # ── Analysis ─────────────────────────────────────────────────────────────

    async def get_analysis(self, analysis_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/v1/analysis/{analysis_id}")

    async def get_explanations(self, analysis_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/v1/analysis/{analysis_id}/explanations")

    async def wait_for_analysis(
        self,
        job_id: str,
        poll_interval: float = 2.0,
        timeout: float = 600.0,
    ) -> dict[str, Any]:
        """Wait for job completion and return the analysis result."""
        job = await self.wait_for_job(job_id, poll_interval, timeout)
        if job["status"] == "failed":
            raise AuraClientError(f"Job failed: {job.get('error_message')}")
        analysis_id = job.get("analysis_id")
        if not analysis_id:
            raise AuraClientError("Job complete but no analysis_id found")
        return await self.get_analysis(analysis_id)

    # ── Evidence ──────────────────────────────────────────────────────────────

    async def export_evidence(
        self, case_id: str, analysis_id: str | None = None
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            f"/v1/cases/{case_id}/evidence/export",
            json={"analysis_id": analysis_id},
        )

    async def verify_evidence(self, evidence_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/v1/evidence/{evidence_id}/verify")

    # ── Live sessions ─────────────────────────────────────────────────────────

    async def create_live_session(
        self,
        case_id: str,
        source_type: str,
        authorization_reference: str,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/v1/live-sessions",
            json={
                "case_id": case_id,
                "source_type": source_type,
                "authorization_reference": authorization_reference,
            },
        )

    async def send_live_event(
        self, session_id: str, chunk: bytes, timestamp_ms: int
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            f"/v1/live-sessions/{session_id}/events",
            content=chunk,
            headers={
                "Content-Type": "application/octet-stream",
                "X-Timestamp-Ms": str(timestamp_ms),
            },
        )
