"""
Dataset Loader for Deep Voice Deepfake Voice Recognition.
Extracts audio waveforms and prepares acoustic features or raw waveforms for training.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import wave


class DeepVoiceDataset(Dataset):
    def __init__(self, root_dir: str | Path, target_samples: int = 32000) -> None:
        self.root_dir = Path(root_dir)
        self.target_samples = target_samples
        self.samples: list[tuple[Path, int]] = []

        # Look for real/fake subdirectories or label files
        for p in self.root_dir.glob("**/*.wav"):
            parent_lower = p.parent.name.lower()
            fname_lower = p.name.lower()
            if "fake" in parent_lower or "synthetic" in parent_lower or "fake" in fname_lower:
                label = 1
            else:
                label = 0
            self.samples.append((p, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        wav_path, label = self.samples[idx]
        waveform = self._load_wav(wav_path)
        return torch.from_numpy(waveform).float(), label

    def _load_wav(self, path: Path) -> np.ndarray:
        try:
            with wave.open(str(path), "rb") as wf:
                n_channels = wf.getnchannels()
                n_frames = wf.getnframes()
                data = wf.readframes(n_frames)
                audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                if n_channels > 1:
                    audio = audio.reshape(-1, n_channels)[:, 0]
                
                # Truncate or pad to target length
                if len(audio) < self.target_samples:
                    audio = np.pad(audio, (0, self.target_samples - len(audio)))
                else:
                    audio = audio[: self.target_samples]
                return audio
        except Exception:
            return np.zeros(self.target_samples, dtype=np.float32)
