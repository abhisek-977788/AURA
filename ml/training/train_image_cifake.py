"""
Training Pipeline for Visual Deepfake / Synthetic Image Detection using CIFAKE.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ml.datasets.cifake_loader import CIFAKEDataset


class SimpleMesoCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8x8
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def train(dataset_dir: str, epochs: int = 5, batch_size: int = 64, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AURA Training] Training on device: {device}")

    train_ds = CIFAKEDataset(dataset_dir, split="train")
    test_ds = CIFAKEDataset(dataset_dir, split="test")

    print(f"[AURA Training] Loaded {len(train_ds)} train samples, {len(test_ds)} test samples")

    if len(train_ds) == 0:
        print("[AURA Training] No images found. Run `python ml/datasets/download_datasets.py` first.")
        return

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = SimpleMesoCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / max(total, 1)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/max(total,1):.4f} Acc: {train_acc:.4f}")

    # Save model weights
    out_dir = Path("d:/AURA/models/video")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "cifake_mesonet.pt"
    torch.save(model.state_dict(), save_path)
    print(f"[AURA Training] Trained weights saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to CIFAKE directory")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    train(args.dataset, epochs=args.epochs)
