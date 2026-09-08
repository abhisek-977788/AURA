"""
AURA Dataset Downloader.
Downloads benchmark and user-specified datasets via kagglehub:
1. birdy654/cifake-real-and-ai-generated-synthetic-images (Visual AI-generated vs Real)
2. birdy654/deep-voice-deepfake-voice-recognition (Synthetic Voice vs Real)
"""

from __future__ import annotations

import os
from pathlib import Path

DATASETS_DIR = Path("d:/AURA/datasets")
DATASETS_DIR.mkdir(parents=True, exist_ok=True)


def download_cifake() -> str:
    print("[AURA] Downloading CIFAKE Dataset (Real vs AI Generated Images)...")
    import kagglehub
    path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
    print("[AURA] CIFAKE downloaded to:", path)
    return path


def download_deep_voice() -> str:
    print("[AURA] Downloading Deep Voice Dataset (Real vs Synthetic Voices)...")
    import kagglehub
    path = kagglehub.dataset_download("birdy654/deep-voice-deepfake-voice-recognition")
    print("[AURA] Deep Voice downloaded to:", path)
    return path


if __name__ == "__main__":
    cifake_path = download_cifake()
    voice_path = download_deep_voice()
    print("\nDataset paths recorded:")
    print("  CIFAKE:", cifake_path)
    print("  DEEP_VOICE:", voice_path)
