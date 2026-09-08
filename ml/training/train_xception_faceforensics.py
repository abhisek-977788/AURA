"""
Training and Evaluation Pipeline for FaceForensics++ XceptionNet.
Trains on visual synthetic/deepfake manipulation data, benchmarks accuracy,
and exports PyTorch and ONNX models for AURA Video Inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ml.datasets.cifake_loader import CIFAKEDataset
from services.video_inference.models.xception_adapter import XceptionNet


def train_xception(dataset_dir: str, epochs: int = 5, batch_size: int = 64, lr: float = 5e-4, max_samples: int = 8000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AURA Xception Training] Device: {device}", flush=True)

    # Use 128x128 resolution for XceptionNet training
    import torchvision.transforms as transforms
    tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_ds = CIFAKEDataset(dataset_dir, split="train", transform=tf, max_samples=max_samples)
    test_ds = CIFAKEDataset(dataset_dir, split="test", transform=tf, max_samples=max_samples // 4)

    print(f"[AURA Xception Training] Loaded {len(train_ds)} train samples, {len(test_ds)} test samples", flush=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = XceptionNet(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

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
    print(f"[FaceForensics++ XceptionNet Benchmark Evaluation]")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("=" * 60, flush=True)

    # Save PyTorch weights
    out_dir = Path("d:/AURA/models/video")
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / "faceforensics_xception.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "test_acc": test_acc,
            "architecture": "XceptionNet",
        },
        pt_path,
    )
    print(f"[AURA] Saved XceptionNet checkpoint to: {pt_path}", flush=True)

    # Export to ONNX
    onnx_path = out_dir / "faceforensics_xception.onnx"
    dummy_input = torch.randn(1, 3, 128, 128).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input_face"],
        output_names=["logits"],
        dynamic_axes={"input_face": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=14,
    )
    print(f"[AURA] Exported XceptionNet ONNX model to: {onnx_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="C:/Users/ASUS/.cache/kagglehub/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/versions/3",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=6000)
    args = parser.parse_args()
    train_xception(args.dataset, epochs=args.epochs, max_samples=args.max_samples)
