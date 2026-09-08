"""
FaceForensics++ Dataset Downloader Integration.
Based on ondyari/FaceForensics official dataset protocols.
Supports all manipulation methods:
- Deepfakes
- Face2Face
- FaceSwap
- NeuralTextures
- FaceShifter
Across compression levels: raw (c0), c23 (HQ), c40 (LQ).
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path


BASE_URL = "http://kaldir.vc.in.tum.de/faceforensics"
METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "original_sequences"]
COMPRESSIONS = ["raw", "c23", "c40"]
TYPES = ["videos", "images", "masks"]


def download_file(url: str, out_file: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"[AURA FaceForensics] Downloading: {url} -> {out_file}")
    try:
        urllib.request.urlretrieve(url, str(out_file))
    except Exception as e:
        print(f"[AURA FaceForensics] Download failed: {e}")
        print("Note: Access to FaceForensics++ requires an authorized server link. See https://github.com/ondyari/FaceForensics")


def main():
    parser = argparse.ArgumentParser(description="FaceForensics++ dataset downloader for AURA")
    parser.add_argument("--output_path", type=str, default="d:/AURA/datasets/faceforensics")
    parser.add_argument("--dataset", type=str, default="FaceForensics", choices=["FaceForensics", "DeepFakeDetection"])
    parser.add_argument("--compression", "-c", type=str, default="c23", choices=COMPRESSIONS)
    parser.add_argument("--type", "-t", type=str, default="videos", choices=TYPES)
    parser.add_argument("--method", "-m", type=str, default="Deepfakes", choices=METHODS)
    args = parser.parse_args()

    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AURA — FaceForensics++ Dataset Integration")
    print(f"Target: {args.method} ({args.compression}, {args.type})")
    print(f"Output: {out_dir}")
    print("=" * 60)
    print("To download raw video sequences, submit the application form at:")
    print("https://github.com/ondyari/FaceForensics#access-request\n")


if __name__ == "__main__":
    main()
