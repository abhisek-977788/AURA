"""
AURA - Image Preprocessing Pipeline
Handles automated extraction of Image.zip (140K Real and Fake Faces)
and PyTorch dataset generation with computer vision augmentations.
"""

import os
import zipfile
import random
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

IMAGE_ZIP_PATH = Path("D:/AURA/Image.zip")
IMAGE_EXTRACT_DIR = Path("D:/AURA/data/image")


def extract_image_dataset(zip_path: Path = IMAGE_ZIP_PATH, dest_dir: Path = IMAGE_EXTRACT_DIR) -> Path:
    """
    Extracts Image.zip into destination folder if not already extracted.
    Returns the root folder containing the train/valid/test subfolders.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset already extracted
    candidates = list(dest_dir.glob("**/train"))
    if candidates:
        dataset_root = candidates[0].parent
        print(f"[Image Preprocess] Dataset already extracted at: {dataset_root}")
        return dataset_root

    if not zip_path.exists():
        raise FileNotFoundError(f"Image dataset archive not found at: {zip_path}")

    print(f"[Image Preprocess] Extracting {zip_path.name} into {dest_dir}... (this may take a few moments)")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(dest_dir)
    
    candidates = list(dest_dir.glob("**/train"))
    if candidates:
        dataset_root = candidates[0].parent
        print(f"[Image Preprocess] Successfully extracted to: {dataset_root}")
        return dataset_root
    
    return dest_dir


def get_image_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns (train_transform, val_transform) with augmentations:
    - Random Horizontal Flip
    - Random Rotation (+-15 deg)
    - Color Jitter (Brightness / Contrast)
    - Random Resized Crop to 256x256
    - Gaussian Blur
    - Standard RGB Normalization
    """
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(256, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, eval_transform


class ReIndexedDataset(Dataset):
    """Wraps ImageFolder to ensure label 1.0 = Fake (Synthetic) and 0.0 = Real (Authentic)."""

    def __init__(self, base_ds, fake_is_one=True):
        self.base_ds = base_ds
        self.fake_is_one = fake_is_one
        self.classes = ["Real", "Fake"] if fake_is_one else ["Fake", "Real"]
        self.orig_fake_idx = base_ds.class_to_idx.get("fake", 0)

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        img, label = self.base_ds[idx]
        target = 1.0 if label == self.orig_fake_idx else 0.0
        return img, torch.tensor(target, dtype=torch.float32)


def get_image_dataloaders(
    data_dir: Optional[Path] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    sample_limit: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train, validation, and test DataLoaders for human face deepfake detection.
    Labels: 0 = Real, 1 = Fake (aligned with standard binary classification)
    """
    if data_dir is None:
        data_dir = extract_image_dataset()

    train_dir = data_dir / "train"
    valid_dir = data_dir / "valid"
    test_dir = data_dir / "test"

    train_tf, eval_tf = get_image_transforms()

    train_ds = ImageFolder(str(train_dir), transform=train_tf)
    val_ds = ImageFolder(str(valid_dir), transform=eval_tf)
    test_ds = ImageFolder(str(test_dir), transform=eval_tf)

    print(f"[Image Preprocess] Detected classes: {train_ds.class_to_idx}")

    train_ds = ReIndexedDataset(train_ds)
    val_ds = ReIndexedDataset(val_ds)
    test_ds = ReIndexedDataset(test_ds)

    if sample_limit is not None and sample_limit > 0:
        def get_balanced_indices(ds, limit):
            targets = [s[1] for s in ds.base_ds.samples]
            orig_fake_idx = ds.orig_fake_idx
            fake_idxs = [i for i, t in enumerate(targets) if t == orig_fake_idx]
            real_idxs = [i for i, t in enumerate(targets) if t != orig_fake_idx]
            half = limit // 2
            selected = fake_idxs[:half] + real_idxs[:half]
            random.seed(42)
            random.shuffle(selected)
            return selected

        train_indices = get_balanced_indices(train_ds, sample_limit)
        val_indices = get_balanced_indices(val_ds, max(20, sample_limit // 4))
        test_indices = get_balanced_indices(test_ds, max(20, sample_limit // 4))
        train_ds = Subset(train_ds, train_indices)
        val_ds = Subset(val_ds, val_indices)
        test_ds = Subset(test_ds, test_indices)
        print(f"[Image Preprocess] Stratified sample limit active: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    root = extract_image_dataset()
    train_l, val_l, test_l = get_image_dataloaders(root, batch_size=4, sample_limit=10)
    for imgs, targets in train_l:
        print(f"Loaded batch shape: {imgs.shape}, targets: {targets}")
        break
