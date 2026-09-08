"""
Dataset Loader for CIFAKE (Real vs AI-Generated Synthetic Images).
CIFAKE directory structure typically has:
  train/
    REAL/
    FAKE/
  test/
    REAL/
    FAKE/
"""

from __future__ import annotations

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CIFAKEDataset(Dataset):
    def __init__(self, root_dir: str | Path, split: str = "train", transform=None) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.samples: list[tuple[Path, int]] = []  # (path, label: 0 for real, 1 for fake)

        split_dir = self.root_dir / split
        if not split_dir.exists():
            # Search if dataset is direct or nested
            candidate = list(self.root_dir.glob(f"**/{split}"))
            if candidate:
                split_dir = candidate[0]
            else:
                split_dir = self.root_dir

        # Look for REAL and FAKE folders
        real_dir = split_dir / "REAL"
        fake_dir = split_dir / "FAKE"

        if real_dir.exists():
            for p in real_dir.glob("*.jpg"):
                self.samples.append((p, 0))
        if fake_dir.exists():
            for p in fake_dir.glob("*.jpg"):
                self.samples.append((p, 1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
