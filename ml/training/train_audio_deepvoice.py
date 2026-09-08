"""
Training Pipeline for Deep Voice Deepfake Voice Recognition.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ml.datasets.deepvoice_loader import DeepVoiceDataset


class Audio1DCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=8),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=32, stride=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=16, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) -> (B, 1, T)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        feat = self.conv(x).squeeze(-1)
        return self.fc(feat)


def train(dataset_dir: str, epochs: int = 5, batch_size: int = 32, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AURA Audio Training] Training on device: {device}")

    ds = DeepVoiceDataset(dataset_dir)
    print(f"[AURA Audio Training] Loaded {len(ds)} audio samples")

    if len(ds) == 0:
        print("[AURA Audio Training] No audio files found in directory.")
        return

    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = Audio1DCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for wavs, labels in train_loader:
            wavs, labels = wavs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(wavs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * wavs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / max(total, 1)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/max(total,1):.4f} Acc: {acc:.4f}")

    out_dir = Path("d:/AURA/models/audio")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "deepvoice_1dcnn.pt"
    torch.save(model.state_dict(), save_path)
    print(f"[AURA Audio Training] Weights saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to Deep Voice dataset")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    train(args.dataset, epochs=args.epochs)
