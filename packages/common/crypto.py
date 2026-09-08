"""
AURA cryptographic utility functions.

Design decisions:
- All digest functions use hashlib.sha256 from the Python standard library.
  We deliberately avoid third-party hashing wrappers to minimise the attack
  surface on the supply chain and to ensure FIPS-compatible behaviour when the
  OS ships with a FIPS-validated OpenSSL build.
- sha256_file() reads in 64 KiB chunks to avoid loading large media files
  (which can be hundreds of MB) entirely into memory. This is critical because
  AURA processes high-resolution video evidence.
- constant_time_compare() wraps hmac.compare_digest to prevent timing attacks
  when comparing tokens, HMAC tags, or other secrets. Never use == for this.
- generate_event_id() returns a UUID4 string. UUID4 is chosen because it is
  unpredictable (128 bits of entropy) and universally unique without
  coordination, making it safe to generate in distributed workers.
"""

from __future__ import annotations

import hashlib
import hmac
import uuid
from pathlib import Path

# Chunk size for streaming file hashing — 64 KiB balances memory use and I/O
_CHUNK_SIZE: int = 64 * 1024


def sha256_file(path: Path) -> str:
    """
    Compute the SHA-256 hex digest of a file on disk.

    Reads the file in streaming 64 KiB chunks so that arbitrarily large video
    evidence files can be hashed without exhausting process memory.

    Args:
        path: Absolute or relative path to the file.

    Returns:
        Lowercase hex-encoded SHA-256 digest string (64 characters).

    Raises:
        FileNotFoundError: If the path does not exist.
        PermissionError: If the process cannot read the file.
    """
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """
    Compute the SHA-256 hex digest of an in-memory byte string.

    Args:
        data: Raw bytes to hash (e.g. an in-memory video frame or audio buffer).

    Returns:
        Lowercase hex-encoded SHA-256 digest string (64 characters).
    """
    return hashlib.sha256(data).hexdigest()


def sha256_string(s: str) -> str:
    """
    Compute the SHA-256 hex digest of a UTF-8 encoded string.

    Useful for hashing JSON blobs (e.g. analysis results) before signing
    them into the evidence manifest.

    Args:
        s: The string to hash.

    Returns:
        Lowercase hex-encoded SHA-256 digest string (64 characters).
    """
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def generate_event_id() -> str:
    """
    Generate a globally unique event identifier.

    UUID4 is used because:
    - 122 bits of randomness makes collisions astronomically unlikely across
      distributed worker nodes without any coordination.
    - It contains no node/MAC address information (unlike UUID1), which avoids
      leaking host identity into stored evidence metadata.

    Returns:
        Hyphenated lowercase UUID4 string,
        e.g. '3fa85f64-5717-4562-b3fc-2c963f66afa6'.
    """
    return str(uuid.uuid4())


def constant_time_compare(a: str, b: str) -> bool:
    """
    Compare two strings in constant time to prevent timing side-channel attacks.

    Regular string comparison (==) short-circuits on the first differing
    character, leaking information about how many leading characters matched.
    This is exploitable when comparing tokens, HMAC signatures, or API keys.

    Uses hmac.compare_digest which is guaranteed to run in O(n) regardless of
    where the strings first differ.

    Args:
        a: First string.
        b: Second string.

    Returns:
        True if the strings are identical, False otherwise.
    """
    return hmac.compare_digest(
        a.encode("utf-8"),
        b.encode("utf-8"),
    )
