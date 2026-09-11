"""
AURA - Audio Deepfake Detector Training Pipeline
Dataset: archive (6).zip (Cloned Speech & Voice Conversions)
Model: RawNet2 / Spectrogram-CNN Architecture
Loss: BCEWithLogitsLoss, Optimizer: AdamW, Scheduler: CosineAnnealingLR
Outputs: Metrics, ROC Curves, Confusion Matrix, Spectrogram Saliency Attribution.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from preprocessing.audio_preprocess import get_audio_dataloaders
from models.audio_model import build_audio_model
from training.utils import (
    calculate_metrics,
    log_metrics_to_csv,
    plot_loss_and_accuracy,
    plot_roc_curve,
    plot_confusion_matrix,
    EarlyStopping,
    GRADCAM_DIR,
    LOGS_DIR,
)

SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = SAVED_MODELS_DIR / "audio_detector.pth"


def plot_spectrogram_saliency(
    model: nn.Module,
    spec_tensor: torch.Tensor,
    save_path: Path = None,
    device: torch.device = None,
):
    """
    Computes input gradient saliency map over the audio Mel Spectrogram
    to visualize which frequencies/timeframes indicate synthetic voice synthesis.
    """
    if save_path is None:
        save_path = GRADCAM_DIR / "audio_spectrogram_saliency.png"

    model.eval()
    spec = spec_tensor.clone().detach().to(device).requires_grad_(True)
    if spec.dim() == 3:
        spec = spec.unsqueeze(0)

    logits = model(spec)
    prob = torch.sigmoid(logits).item()

    model.zero_grad()
    logits.backward()

    # Saliency: absolute value of gradient
    saliency = spec.grad.squeeze().abs().cpu().numpy()
    if saliency.max() > 0:
        saliency = saliency / saliency.max()

    raw_spec = spec.squeeze().detach().cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    # 1. Mel Spectrogram
    im1 = axes[0].imshow(raw_spec, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title(f"Input Mel Spectrogram (Predicted {prob*100:.1f}% Fake)", fontweight="bold")
    axes[0].set_ylabel("Mel Frequency Bands", fontweight="bold")
    fig.colorbar(im1, ax=axes[0], format="%+2.0f dB")

    # 2. Saliency Attribution Map
    im2 = axes[1].imshow(saliency, aspect="auto", origin="lower", cmap="inferno")
    axes[1].set_title("Frequency-Domain Saliency Attribution (Voice Artifacts)", fontweight="bold")
    axes[1].set_xlabel("Time Frames (Hop 256)", fontweight="bold")
    axes[1].set_ylabel("Mel Frequency Bands", fontweight="bold")
    fig.colorbar(im2, ax=axes[1])

    plt.suptitle("AURA - Explainable Audio Deepfake Saliency", fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Explainable AI] Saved audio spectrogram saliency map to: {save_path}")


def train_audio_pipeline(
    epochs: int = 25,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device_name: str = None,
):
    print("=" * 70)
    print("      AURA — AUDIO DEEPFAKE DETECTION TRAINING PIPELINE       ")
    print("=" * 70)

    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AURA Device] Using compute device: {device}")

    # 1. Audio Dataloaders
    print("\n[Step 1/5] Extracting & Preprocessing Audio from archive (6).zip...")
    train_loader, val_loader = get_audio_dataloaders(batch_size=batch_size)

    # 2. Build Audio Model
    print("\n[Step 2/5] Initializing RawNet2 / Spectrogram-CNN Architecture...")
    model = build_audio_model(device=device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    early_stopper = EarlyStopping(patience=5)

    train_loss_history = []
    val_loss_history = []
    val_acc_history = []
    best_val_loss = float("inf")

    # 3. Training Loop
    print(f"\n[Step 3/5] Starting Audio Model Training for {epochs} Epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]")

        for mels, labels in pbar:
            mels = mels.to(device)                 # (B, 1, 80, 188)
            labels = labels.to(device).unsqueeze(1) # (B, 1)

            optimizer.zero_grad()
            logits = model(mels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * mels.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for mels, labels in val_loader:
                mels = mels.to(device)
                labels = labels.to(device).unsqueeze(1)

                logits = model(mels)
                loss = criterion(logits, labels)
                val_loss += loss.item() * mels.size(0)

                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                val_preds.extend(probs)
                val_targets.extend(labels.cpu().numpy().flatten())

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_loss_history.append(epoch_val_loss)
        val_metrics = calculate_metrics(np.array(val_targets), np.array(val_preds))
        val_acc = val_metrics["accuracy"]
        val_acc_history.append(val_acc)

        scheduler.step()
        print(f"Epoch {epoch:02d} Summary: Train Loss = {epoch_train_loss:.4f} | Val Loss = {epoch_val_loss:.4f} | Val Acc = {val_acc*100:.2f}% | Val AUC = {val_metrics['roc_auc']:.4f}")

        # Best checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  --> Saved new best audio checkpoint to: {CHECKPOINT_PATH}")

        if early_stopper(epoch_val_loss):
            print(f"[Early Stopping] Validation loss plateaued at epoch {epoch}.")
            break

    # 4. Final Evaluation
    print("\n[Step 4/5] Evaluating Best Audio Model on Validation Set...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True, map_location=device))
    model.eval()

    eval_preds = []
    eval_targets = []
    sample_fake_mel = None

    with torch.no_grad():
        for mels, labels in val_loader:
            mels = mels.to(device)
            logits = model(mels)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            targets = labels.numpy().flatten()

            eval_preds.extend(probs)
            eval_targets.extend(targets)

            if sample_fake_mel is None:
                for idx, t in enumerate(targets):
                    if t == 1.0:
                        sample_fake_mel = mels[idx : idx + 1]
                        break

    audio_metrics = calculate_metrics(np.array(eval_targets), np.array(eval_preds))
    print("\n" + "=" * 50)
    print("         FINAL AUDIO MODEL METRICS REPORT         ")
    print("=" * 50)
    print(f"Accuracy:  {audio_metrics['accuracy']*100:.2f}%")
    print(f"Precision: {audio_metrics['precision']*100:.2f}%")
    print(f"Recall:    {audio_metrics['recall']*100:.2f}%")
    print(f"F1-Score:  {audio_metrics['f1']:.4f}")
    print(f"ROC-AUC:   {audio_metrics['roc_auc']:.4f}")
    print("Confusion Matrix:")
    print(np.array(audio_metrics["confusion_matrix"]))
    print("=" * 50)

    # 5. Visualizations & Saliency
    print("\n[Step 5/5] Generating Audio Visualizations & Spectrogram Saliency...")
    log_metrics_to_csv("Audio_RawNet2_CNN", audio_metrics)
    plot_loss_and_accuracy(train_loss_history, val_loss_history, val_acc_history, "Audio RawNet2-CNN")
    plot_roc_curve(np.array(eval_targets), np.array(eval_preds), "Audio RawNet2-CNN")
    plot_confusion_matrix(np.array(audio_metrics["confusion_matrix"]), "Audio RawNet2-CNN")

    if sample_fake_mel is not None:
        plot_spectrogram_saliency(
            model,
            sample_fake_mel,
            save_path=GRADCAM_DIR / "audio_spectrogram_saliency.png",
            device=device,
        )

    print("\n[AURA Success] Audio deepfake detection training pipeline completed successfully!")
    return audio_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AURA Audio Deepfake Detector")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs (default: 25)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 0.0001)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    args = parser.parse_args()

    train_audio_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device_name=args.device,
    )
