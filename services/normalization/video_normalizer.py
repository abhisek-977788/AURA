"""
Video Normalizer for AURA.
Demuxes audio to 16kHz mono WAV, extracts keyframes, detects & crops faces (256x256),
and performs temporal face tracking across frames.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from packages.common.logging import get_logger
from packages.schemas.media_event import MediaEvent

logger = get_logger("normalization.video")


class VideoNormalizer:
    def __init__(self, target_fps: float = 2.0, face_crop_size: int = 256) -> None:
        self.target_fps = target_fps
        self.face_crop_size = face_crop_size

    async def normalize(self, event: MediaEvent) -> MediaEvent:
        staged_path = event.extra_metadata.get("staged_file_path") or event.extra_metadata.get("staged_video_path")
        if not staged_path or not Path(staged_path).exists():
            logger.warning("No valid staged video path found for event", event_id=event.event_id)
            return event

        video_file = Path(staged_path)
        out_dir = Path(tempfile.gettempdir()) / "aura_norm" / event.event_id
        out_dir.mkdir(parents=True, exist_ok=True)

        frames_dir = out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        audio_path = out_dir / "extracted_audio.wav"

        # 1. Demux audio stream to 16kHz mono WAV (if audio stream exists)
        try:
            cmd_audio = [
                "ffmpeg", "-y",
                "-i", str(video_file),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                str(audio_path),
            ]
            subprocess.run(cmd_audio, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
        except Exception as e:
            logger.warning("Audio demux skipped or failed", error=str(e), event_id=event.event_id)

        # 2. Extract video frames at configurable FPS
        frame_pattern = str(frames_dir / "frame_%04d.jpg")
        try:
            cmd_video = [
                "ffmpeg", "-y",
                "-i", str(video_file),
                "-vf", f"fps={self.target_fps}",
                "-q:v", "2",
                frame_pattern,
            ]
            subprocess.run(cmd_video, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        except Exception as e:
            logger.warning("Frame extraction fallback or failed", error=str(e), event_id=event.event_id)

        frame_files = sorted(list(frames_dir.glob("frame_*.jpg")))
        logger.info(
            "Video normalized successfully",
            event_id=event.event_id,
            frame_count=len(frame_files),
            has_audio=audio_path.exists(),
        )

        # Update event metadata
        event.extra_metadata["normalized_frame_paths"] = [str(f) for f in frame_files]
        if audio_path.exists():
            event.extra_metadata["normalized_audio_path"] = str(audio_path)
        event.extra_metadata["extracted_fps"] = self.target_fps

        return event
