"""
Optimized Dataset Loader for CIFAKE (Real vs AI-Generated Synthetic Images).
Uses os.scandir for high-speed file discovery across 100,000+ images.
"""

from __future__ import annotations

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CIFAKEDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        transform=None,
        max_samples: int | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.samples: list[tuple[str, int]] = []

        split_dir = self.root_dir / split
        real_dir = split_dir / "REAL"
        fake_dir = split_dir / "FAKE"

        limit_per_class = max_samples // 2 if max_samples else None

        # Fast discovery using os.scandir
        if real_dir.exists():
            count = 0
            with os.scandir(str(real_dir)) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append((entry.path, 0))
                        count += 1
                        if limit_per_class and count >= limit_per_class:
                            break

        if fake_dir.exists():
            count = 0
            with os.scandir(str(fake_dir)) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append((entry.path, 1))
                        count += 1
                        if limit_per_class and count >= limit_per_class:
                            break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
