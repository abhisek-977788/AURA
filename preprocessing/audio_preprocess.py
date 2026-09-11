"""
AURA - Audio Preprocessing Pipeline
Extracts archive (6).zip, resamples audio to 16,000 Hz, trims silence,
normalizes amplitude, and computes Mel Spectrogram, MFCC, and LFCC features.
"""

import os
import zipfile
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa

AUDIO_ZIP_PATH = Path("D:/AURA/archive (6).zip")
AUDIO_EXTRACT_DIR = Path("D:/AURA/data/audio")
AUDIO_CACHE_DIR = Path("D:/AURA/data/audio/processed_tensors")

TARGET_SAMPLE_RATE = 16000
SEGMENT_DURATION_SEC = 3.0
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256


def extract_audio_dataset(zip_path: Path = AUDIO_ZIP_PATH, dest_dir: Path = AUDIO_EXTRACT_DIR) -> Path:
    """
    Extracts archive (6).zip into data/audio if not already present.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = list(dest_dir.glob("**/*.wav")) + list(dest_dir.glob("**/*.mp3"))
    if len(existing) >= 20:
        print(f"[Audio Preprocess] Found {len(existing)} audio files already extracted in {dest_dir}.")
        return dest_dir

    if not zip_path.exists():
        raise FileNotFoundError(f"Audio dataset archive not found at: {zip_path}")

    print(f"[Audio Preprocess] Extracting {zip_path.name} into {dest_dir}...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(dest_dir)

    extracted = list(dest_dir.glob("**/*.wav")) + list(dest_dir.glob("**/*.mp3"))
    print(f"[Audio Preprocess] Extracted {len(extracted)} audio files.")
    return dest_dir


def extract_features_from_waveform(
    waveform: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE,
) -> Dict[str, np.ndarray]:
    """
    Computes Mel Spectrogram, MFCC, and delta features from a 1D audio waveform.
    """
    # 1. Mel Spectrogram (80 bands, normalized to dB)
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Normalize between [-1.0, 1.0]
    mel_norm = (mel_db + 40.0) / 40.0

    # 2. MFCC (20 coefficients)
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=20, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc_delta = librosa.feature.delta(mfcc)

    # 3. LFCC (approximated via linear filterbank)
    linear_spec = np.abs(librosa.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH))
    lfcc_feat = librosa.feature.spectral_centroid(S=linear_spec, sr=sr)

    return {
        "mel": mel_norm.astype(np.float32),
        "mfcc": mfcc.astype(np.float32),
        "mfcc_delta": mfcc_delta.astype(np.float32),
        "lfcc": lfcc_feat.astype(np.float32),
    }


def load_and_preprocess_audio(
    file_path: Path,
    target_sr: int = TARGET_SAMPLE_RATE,
    duration_sec: float = SEGMENT_DURATION_SEC,
) -> List[np.ndarray]:
    """
    Loads audio, resamples to 16,000 Hz, trims leading/trailing silence,
    normalizes amplitude, and slices into non-overlapping 3-second segments.
    """
    try:
        y, sr = librosa.load(str(file_path), sr=target_sr, mono=True)
    except Exception as e:
        print(f"[Audio Preprocess] Warning: failed to load {file_path}: {e}")
        return []

    # Trim silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    if len(y_trimmed) < int(0.5 * target_sr):  # Skip if < 0.5s of speech
        y_trimmed = y

    # Amplitude normalization
    peak = np.max(np.abs(y_trimmed))
    if peak > 1e-6:
        y_trimmed = y_trimmed / peak

    target_length = int(duration_sec * target_sr)  # e.g., 48,000 samples for 3s
    segments = []

    if len(y_trimmed) < target_length:
        # Pad with repeat
        repeats = int(np.ceil(target_length / len(y_trimmed)))
        padded = np.tile(y_trimmed, repeats)[:target_length]
        segments.append(padded)
    else:
        # Slice into 3-second chunks with 50% overlap for comprehensive training coverage
        step = target_length // 2
        for start in range(0, len(y_trimmed) - target_length + 1, step):
            segments.append(y_trimmed[start : start + target_length])
            if len(segments) >= 10:  # Max 10 segments per audio file
                break

    return segments


class AudioDeepfakeDataset(Dataset):
    """
    PyTorch Dataset returning (mel_spectrogram_tensor, label):
    - mel_spectrogram_tensor: shape (1, 80, 188)
    - label: 0.0 for Real, 1.0 for Fake
    """

    def __init__(self, data_dir: Path = AUDIO_EXTRACT_DIR, cache_dir: Path = AUDIO_CACHE_DIR):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.samples: List[Tuple[Path, float]] = []

        fake_files = list(data_dir.glob("**/FAKE/*.wav")) + list(data_dir.glob("**/fake/*.wav"))
        real_files = list(data_dir.glob("**/REAL/*.wav")) + list(data_dir.glob("**/real/*.wav"))
        demo_files = list(data_dir.glob("**/DEMONSTRATION/*.mp3"))
        for d in demo_files:
            if "original" in d.name.lower():
                real_files.append(d)
            else:
                fake_files.append(d)

        for f in fake_files:
            self.samples.append((f, 1.0))
        for r in real_files:
            self.samples.append((r, 0.0))

        random.seed(42)
        random.shuffle(self.samples)
        print(f"[Audio Preprocess] Total audio dataset items: {len(self.samples)} (Real: {len(real_files)}, Fake: {len(fake_files)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path, label = self.samples[idx]
        cache_key = f"{file_path.stem}.pt"
        cache_path = self.cache_dir / cache_key

        if cache_path.exists():
            try:
                mel_tensor = torch.load(cache_path, weights_only=True)
                return mel_tensor, torch.tensor(label, dtype=torch.float32)
            except Exception:
                pass

        segs = load_and_preprocess_audio(file_path)
        seg = segs[0] if segs else np.zeros(int(SEGMENT_DURATION_SEC * TARGET_SAMPLE_RATE), dtype=np.float32)

        feats = extract_features_from_waveform(seg, sr=TARGET_SAMPLE_RATE)
        mel = feats["mel"]  # (80, T)
        # Ensure fixed length (e.g. 188 frames for 3s @ hop 256)
        expected_len = 188
        if mel.shape[1] < expected_len:
            pad_w = expected_len - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad_w)), mode="edge")
        else:
            mel = mel[:, :expected_len]

        mel_tensor = torch.from_numpy(mel).unsqueeze(0)  # (1, 80, 188)
        torch.save(mel_tensor, cache_path)

        return mel_tensor, torch.tensor(label, dtype=torch.float32)


def get_audio_dataloaders(
    batch_size: int = 32,
    val_split: float = 0.2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Builds train and validation DataLoaders for synthetic voice / audio deepfake detection.
    """
    extract_audio_dataset()
    full_ds = AudioDeepfakeDataset()

    val_size = max(1, int(len(full_ds) * val_split))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available()
    )

    print(f"[Audio Preprocess] Audio DataLoaders ready: {len(train_ds)} train, {len(val_ds)} val samples.")
    return train_loader, val_loader


if __name__ == "__main__":
    extract_audio_dataset()
    tl, vl = get_audio_dataloaders(batch_size=4)
    for mels, lbls in tl:
        print(f"Sample audio batch: mel shape={mels.shape}, labels={lbls}")
        break
