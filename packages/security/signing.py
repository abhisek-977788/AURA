"""
Cryptographic signing and verification for forensic evidence manifests.
Uses Ed25519 for high-speed, tamper-evident asymmetric signing.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519


def generate_keypair() -> tuple[bytes, bytes]:
    """Generates an Ed25519 keypair, returning (private_bytes_pem, public_bytes_pem)."""
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    priv_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return priv_pem, pub_pem


def canonicalize_manifest(manifest_dict: dict[str, Any]) -> bytes:
    """Produces deterministic canonical JSON (sorted keys, no extraneous whitespace, excluding signature)."""
    clean_dict = dict(manifest_dict)
    # Remove signature fields before canonical hashing/signing
    clean_dict.pop("manifest_signature", None)
    clean_dict.pop("signature_algorithm", None)
    clean_dict.pop("signed_at", None)
    clean_dict.pop("signing_key_id", None)

    return json.dumps(clean_dict, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def sign_manifest(manifest_json_or_dict: str | dict[str, Any], private_key_pem: bytes) -> str:
    """Signs a manifest dictionary or JSON string with an Ed25519 private key. Returns Base64 signature."""
    if isinstance(manifest_json_or_dict, str):
        data_dict = json.loads(manifest_json_or_dict)
    else:
        data_dict = manifest_json_or_dict

    canonical_bytes = canonicalize_manifest(data_dict)
    private_key = serialization.load_pem_private_key(private_key_pem, password=None)
    if not isinstance(private_key, ed25519.Ed25519PrivateKey):
        raise ValueError("Provided key is not an Ed25519 private key")

    raw_sig = private_key.sign(canonical_bytes)
    return base64.b64encode(raw_sig).decode("utf-8")


def verify_manifest(
    manifest_json_or_dict: str | dict[str, Any],
    signature_b64: str,
    public_key_pem: bytes,
) -> bool:
    """Verifies an Ed25519 signature against the canonical form of the manifest."""
    if isinstance(manifest_json_or_dict, str):
        data_dict = json.loads(manifest_json_or_dict)
    else:
        data_dict = manifest_json_or_dict

    canonical_bytes = canonicalize_manifest(data_dict)
    public_key = serialization.load_pem_public_key(public_key_pem)
    if not isinstance(public_key, ed25519.Ed25519PublicKey):
        raise ValueError("Provided key is not an Ed25519 public key")

    try:
        raw_sig = base64.b64decode(signature_b64)
        public_key.verify(raw_sig, canonical_bytes)
        return True
    except Exception:
        return False


def load_private_key(path: Path | str) -> bytes:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Signing private key not found at: {p}")
    return p.read_bytes()


def load_public_key(path: Path | str) -> bytes:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Signing public key not found at: {p}")
    return p.read_bytes()
