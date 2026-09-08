"""
Deep Voice Deepfake Acoustic Feature Classifier Training Pipeline.
Trains an MLP on the 26 spectral/chroma/MFCC acoustic features from DATASET-balanced.csv.
Evaluates ROC-AUC, EER, and saves PyTorch checkpoint and ONNX model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ml.evaluation.metrics import compute_eer, compute_brier_score


class DeepVoiceAcousticMLP(nn.Module):
    def __init__(self, input_dim: int = 26) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_deepvoice(csv_path: str | Path, epochs: int = 20, batch_size: int = 128, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AURA Deep Voice] Training device: {device}")

    # 1. Load CSV
    df = pd.read_csv(csv_path)
    print(f"[AURA Deep Voice] Loaded CSV with {len(df)} samples, columns: {list(df.columns)}")

    # 2. Extract features and binary labels
    # Real = 0, Fake = 1
    feature_cols = [c for c in df.columns if c.upper() != "LABEL"]
    X = df[feature_cols].values.astype(np.float32)
    y_raw = df["LABEL"].astype(str).str.upper().values
    y = np.array([1 if "FAKE" in lbl else 0 for lbl in y_raw], dtype=np.int64)

    # Standardize features
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-6
    X = (X - mean) / std

    # Train/test split (80/20)
    n = len(X)
    indices = np.random.permutation(n)
    split_idx = int(0.8 * n)
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    X_train, y_train = torch.tensor(X[train_idx]), torch.tensor(y[train_idx])
    X_test, y_test = torch.tensor(X[test_idx]), torch.tensor(y[test_idx])

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = DeepVoiceAcousticMLP(input_dim=len(feature_cols)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"[AURA Deep Voice] Training set: {len(X_train)}, Test set: {len(X_test)}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()

        train_acc = correct / len(X_train)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(X_train):.4f} Train Acc: {train_acc:.4f}")

    # Evaluation on Test Set
    model.eval()
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

    all_probs_np = np.array(all_probs)
    test_labels_np = y[test_idx]
    test_acc = float(np.mean(np.array(all_preds) == test_labels_np))
    eer, opt_th = compute_eer(test_labels_np, all_probs_np)
    brier = compute_brier_score(test_labels_np, all_probs_np)

    print("\n" + "=" * 60)
    print(f"[AURA Deep Voice Evaluation Report]")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Equal Error Rate (EER): {eer:.4f} ({eer*100:.2f}%) at threshold {opt_th:.3f}")
    print(f"  Brier Score: {brier:.4f}")
    print("=" * 60)

    # Save weights
    out_dir = Path("d:/AURA/models/audio")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "deepvoice_acoustic_mlp.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "features": feature_cols,
            "test_eer": eer,
            "test_acc": test_acc,
        },
        save_path,
    )
    print(f"[AURA Deep Voice] Model checkpoint saved to: {save_path}")

    # Export to ONNX
    onnx_path = out_dir / "deepvoice_acoustic_mlp.onnx"
    dummy_input = torch.randn(1, len(feature_cols)).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["acoustic_features"],
        output_names=["logits"],
        dynamic_axes={"acoustic_features": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=14,
    )
    print(f"[AURA Deep Voice] ONNX model exported to: {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="C:/Users/ASUS/.cache/kagglehub/datasets/birdy654/deep-voice-deepfake-voice-recognition/versions/2/KAGGLE/DATASET-balanced.csv",
    )
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    train_deepvoice(args.csv, epochs=args.epochs)
