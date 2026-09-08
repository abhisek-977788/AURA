"""
Unit tests for Ed25519 manifest signing, verification, and tamper detection.
"""

from packages.security.signing import generate_keypair, sign_manifest, verify_manifest


def test_ed25519_sign_and_verify():
    priv_pem, pub_pem = generate_keypair()
    manifest_data = {
        "manifest_id": "man-12345",
        "case_id": "case-999",
        "original_media_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "synthetic_media_probability": 0.88,
    }

    sig_b64 = sign_manifest(manifest_data, priv_pem)
    assert len(sig_b64) > 0

    # Verification should succeed
    assert verify_manifest(manifest_data, sig_b64, pub_pem) is True

    # Tampered payload should fail verification
    tampered_data = dict(manifest_data)
    tampered_data["synthetic_media_probability"] = 0.12
    assert verify_manifest(tampered_data, sig_b64, pub_pem) is False
