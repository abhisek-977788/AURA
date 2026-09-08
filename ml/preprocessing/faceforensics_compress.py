"""
FaceForensics++ H.264 Compression Pipeline.
Adapted from ondyari/FaceForensics (dataset/compress.py).
Enables generating standardized benchmark compression levels:
- c0: Raw / Lossless (CRF 0)
- c23: High Quality (CRF 23 - standard H.264 web compression)
- c40: Low Quality (CRF 40 - heavy compression / degraded surveillance)
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Literal

from packages.common.logging import get_logger

logger = get_logger("ml.faceforensics.compress")


CompressionLevel = Literal["c0", "c23", "c40"]
CRF_MAP = {"c0": 0, "c23": 23, "c40": 40}


def compress_video(
    input_video_path: str | Path,
    output_video_path: str | Path,
    level: CompressionLevel = "c23",
    fps: int = 30,
) -> Path:
    """Compresses a video using libx264 with FaceForensics++ standardized CRF."""
    inp = Path(input_video_path)
    out = Path(output_video_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    crf = CRF_MAP.get(level, 23)
    codec = "libx264rgb" if crf == 0 else "libx264"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(inp),
        "-c:v", codec,
        "-crf", str(crf),
        "-vf", f"fps={fps}",
        str(out),
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        logger.info("Compressed video according to FaceForensics protocol", level=level, crf=crf, output=str(out))
    except Exception as e:
        logger.error("FFmpeg compression failed", error=str(e), file=str(inp))
        raise

    return out


def extract_frames(video_path: str | Path, output_dir: str | Path, fps: float = 2.0) -> list[Path]:
    """Extracts frames from video into PNG/JPG files (matches FaceForensics extract_compressed_videos.py)."""
    vid = Path(video_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pattern = str(out / "%04d.png")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(vid),
        "-vf", f"fps={fps}",
        "-start_number", "0",
        pattern,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return sorted(list(out.glob("*.png")))
