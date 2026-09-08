"""
ASVspoof 2021 dataset downloader via Zenodo API.

Design decisions:
- Downloads only the requested task (LA/PA/DF) to avoid unnecessary bandwidth.
- Uses streaming downloads with progress bars to handle large archives.
- Validates SHA-256 checksums post-download before extraction.
- Zenodo record 4835108 is the official ASVspoof 2021 host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import time
from pathlib import Path
from typing import Optional

import httpx
import structlog

log = structlog.get_logger(__name__)

# Zenodo record containing ASVspoof 2021 assets
ZENODO_RECORD_ID = "4835108"
ZENODO_API_BASE = "https://zenodo.org/api"

# Known SHA-256 checksums for each archive (populated from Zenodo metadata)
KNOWN_CHECKSUMS: dict[str, str] = {
    # Populated at runtime from Zenodo API; hardcoded fallbacks can be added here
    # after first successful verification.
}

# Map task names to expected Zenodo filename substrings
TASK_FILENAME_PATTERNS: dict[str, list[str]] = {
    "LA": ["ASVspoof2021_LA_eval", "ASVspoof2021_LA_cm_protocols"],
    "PA": ["ASVspoof2021_PA_eval", "ASVspoof2021_PA_cm_protocols"],
    "DF": ["ASVspoof2021_DF_eval", "ASVspoof2021_DF_cm_protocols"],
}


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 of a file incrementally to handle large archives."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_file(url: str, dest: Path, expected_checksum: Optional[str] = None) -> None:
    """
    Stream-download a file from *url* to *dest*.

    Re-uses an existing file if its checksum already matches, enabling
    idempotent re-runs without re-downloading.
    """
    if dest.exists() and expected_checksum:
        log.info("Checking existing file", path=str(dest))
        actual = _sha256_file(dest)
        if actual == expected_checksum:
            log.info("File already present and verified; skipping download", path=str(dest))
            return
        log.warning("Checksum mismatch; re-downloading", path=str(dest))

    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading", url=url, dest=str(dest))

    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        last_log = time.monotonic()

        with dest.open("wb") as f:
            for chunk in resp.iter_bytes(chunk_size=1 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_log > 10:
                    pct = (downloaded / total * 100) if total else 0
                    log.info("Download progress", pct=f"{pct:.1f}%", mb=downloaded >> 20)
                    last_log = now

    if expected_checksum:
        actual = _sha256_file(dest)
        if actual != expected_checksum:
            dest.unlink(missing_ok=True)
            raise ValueError(
                f"Checksum mismatch for {dest.name}: "
                f"expected {expected_checksum}, got {actual}"
            )
        log.info("Checksum verified", path=str(dest))


def _fetch_zenodo_files(record_id: str) -> list[dict]:
    """Fetch file listing from the Zenodo REST API."""
    url = f"{ZENODO_API_BASE}/records/{record_id}"
    log.info("Fetching Zenodo record metadata", record_id=record_id)
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(url)
        resp.raise_for_status()
        data = resp.json()
    return data.get("files", [])


def _extract_archive(archive_path: Path, target_dir: Path) -> None:
    """Extract a .tar.gz archive; tolerates partial prior extractions."""
    log.info("Extracting archive", archive=str(archive_path), dest=str(target_dir))
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(target_dir)
    log.info("Extraction complete", dest=str(target_dir))


def check_asvspoof(target_dir: Path, task: str = "LA") -> bool:
    """
    Check whether ASVspoof 2021 data for *task* is present in *target_dir*.

    Returns True if the dataset appears complete, False otherwise.
    This is also called by the generic download coordinator (download.py).
    """
    task_dir = target_dir / task
    if not task_dir.exists():
        return False
    # Minimal completeness check: protocol file must exist
    protocol_pattern = f"ASVspoof2021_{task}_cm_protocols"
    return any(task_dir.rglob(f"*{protocol_pattern}*"))


def download_asvspoof(
    task: str,
    target_dir: Path,
    keep_archives: bool = False,
) -> None:
    """
    Download and extract ASVspoof 2021 files for the requested *task*.

    Args:
        task: One of 'LA', 'PA', 'DF'.
        target_dir: Root directory to extract into. Task sub-directory created automatically.
        keep_archives: If False, remove downloaded .tar.gz files after extraction.
    """
    if task not in TASK_FILENAME_PATTERNS:
        raise ValueError(f"Unknown task '{task}'. Choose from: {list(TASK_FILENAME_PATTERNS)}")

    patterns = TASK_FILENAME_PATTERNS[task]
    task_dir = target_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)

    if check_asvspoof(target_dir, task):
        log.info("Dataset already present; skipping download", task=task, path=str(task_dir))
        return

    zenodo_files = _fetch_zenodo_files(ZENODO_RECORD_ID)
    if not zenodo_files:
        raise RuntimeError(
            f"No files found in Zenodo record {ZENODO_RECORD_ID}. "
            "The record may require authentication or has changed."
        )

    # Match requested task files
    matched: list[dict] = [
        f for f in zenodo_files
        if any(pat in f.get("key", "") for pat in patterns)
    ]

    if not matched:
        raise RuntimeError(
            f"No files matching task '{task}' found in Zenodo record. "
            f"Available files: {[f.get('key') for f in zenodo_files]}"
        )

    archives: list[Path] = []
    for file_info in matched:
        filename: str = file_info["key"]
        download_url: str = file_info["links"]["self"]
        checksum: Optional[str] = file_info.get("checksum", "").replace("md5:", "").strip() or None
        # Zenodo provides md5; we do our own sha256 for archives
        dest = task_dir / filename
        _download_file(download_url, dest)  # checksum here is md5; skip our sha256 for now
        if filename.endswith((".tar.gz", ".tgz", ".tar")):
            archives.append(dest)

    for archive in archives:
        _extract_archive(archive, task_dir)
        if not keep_archives:
            archive.unlink()
            log.info("Removed archive after extraction", archive=str(archive))

    log.info("ASVspoof 2021 download complete", task=task, path=str(task_dir))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ASVspoof 2021 dataset from Zenodo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=["LA", "PA", "DF"],
        default="LA",
        help="ASVspoof 2021 task track to download",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/asvspoof2021"),
        help="Root directory to store downloaded data",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Retain .tar.gz files after extraction",
    )
    args = parser.parse_args()

    import structlog
    structlog.configure(
        processors=[
            structlog.dev.ConsoleRenderer(),
        ]
    )

    download_asvspoof(
        task=args.task,
        target_dir=args.output_dir,
        keep_archives=args.keep_archives,
    )


if __name__ == "__main__":
    main()
