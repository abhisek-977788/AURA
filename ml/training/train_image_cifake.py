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


def train(dataset_dir: str, epochs: int = 5, batch_size: int = 128, lr: float = 1e-3, max_samples: int = 10000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AURA Training] Training device: {device}", flush=True)

    train_ds = CIFAKEDataset(dataset_dir, split="train", max_samples=max_samples)
    test_ds = CIFAKEDataset(dataset_dir, split="test", max_samples=max_samples // 4 if max_samples else None)

    print(f"[AURA Training] Loaded {len(train_ds)} train samples, {len(test_ds)} test samples", flush=True)

    if len(train_ds) == 0:
        print("[AURA Training] No images found. Check dataset path.", flush=True)
        return

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
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
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/max(total,1):.4f} Train Acc: {train_acc:.4f}", flush=True)

    # Evaluation on Test Set
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

    test_acc = test_correct / max(test_total, 1)
    print("\n" + "=" * 60, flush=True)
    print(f"[AURA CIFAKE Evaluation Report]", flush=True)
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)", flush=True)
    print("=" * 60, flush=True)

    # Save model weights
    out_dir = Path("d:/AURA/models/video")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "cifake_mesonet.pt"
    torch.save(model.state_dict(), save_path)
    print(f"[AURA Training] Trained weights saved to: {save_path}", flush=True)

    # Export to ONNX
    onnx_path = out_dir / "cifake_mesonet.onnx"
    dummy_input = torch.randn(1, 3, 64, 64).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input_image"],
        output_names=["logits"],
        dynamic_axes={"input_image": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=14,
    )
    print(f"[AURA Training] ONNX exported to: {onnx_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="C:/Users/ASUS/.cache/kagglehub/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/versions/3",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=10000)
    args = parser.parse_args()
    train(args.dataset, epochs=args.epochs, max_samples=args.max_samples)
