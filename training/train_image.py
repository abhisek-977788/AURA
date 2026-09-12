"""
AURA - Image Deepfake Detector Training Pipeline
Dataset: 140K Real and Fake Faces (Image.zip)
Model: EfficientNet-B0 with Transfer Learning
Loss: BCEWithLogitsLoss, Optimizer: AdamW, Scheduler: CosineAnnealingLR
Generates Metrics, Confusion Matrix, ROC curves, and Grad-CAM Visualizations.
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

from preprocessing.image_preprocess import get_image_dataloaders
from models.image_model import build_image_model
from training.utils import (
    calculate_metrics,
    log_metrics_to_csv,
    plot_loss_and_accuracy,
    plot_roc_curve,
    plot_confusion_matrix,
    generate_gradcam_heatmap,
    EarlyStopping,
    GRADCAM_DIR,
)

SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = SAVED_MODELS_DIR / "image_detector.pth"


def train_image_pipeline(
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    sample_limit: int = None,
    device_name: str = None,
    resume: bool = False,
    start_epoch: int = 1,
    best_val_loss_init: float = None,
):
    print("=" * 70)
    print("      AURA — IMAGE DEEPFAKE DETECTION TRAINING PIPELINE       ")
    print("=" * 70)

    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AURA Device] Using compute device: {device}")

    # 1. Dataloaders
    print("\n[Step 1/5] Loading and Preprocessing Image Dataset...")
    train_loader, val_loader, test_loader = get_image_dataloaders(
        batch_size=batch_size, sample_limit=sample_limit
    )

    # 2. Build Model
    print("\n[Step 2/5] Initializing EfficientNet-B0 Transfer Learning Architecture...")
    model = build_image_model(device=device)

    # Resume from saved best checkpoint if requested
    if resume and CHECKPOINT_PATH.exists():
        print(f"[Resume] Loading weights from: {CHECKPOINT_PATH}")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
        print(f"[Resume] Continuing from epoch {start_epoch} -> {epochs}")
    elif resume:
        print(f"[Resume] WARNING: No checkpoint at {CHECKPOINT_PATH}, starting fresh.")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    # Fast-forward scheduler state to match start_epoch
    for _ in range(start_epoch - 1):
        scheduler.step()
    early_stopper = EarlyStopping(patience=5)

    train_loss_history = []
    val_loss_history = []
    val_acc_history = []
    best_val_loss = best_val_loss_init if best_val_loss_init is not None else float("inf")
    if best_val_loss_init is not None:
        print(f"[Resume] Restored best_val_loss = {best_val_loss:.6f}")

    # 3. Training Loop
    print(f"\n[Step 3/5] Starting Training from Epoch {start_epoch} to {epochs}...")
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]")

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1)

                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)

                probs = torch.sigmoid(logits).cpu().numpy()
                all_val_preds.extend(probs.flatten())
                all_val_targets.extend(labels.cpu().numpy().flatten())

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_loss_history.append(epoch_val_loss)
        val_metrics = calculate_metrics(np.array(all_val_targets), np.array(all_val_preds))
        val_acc = val_metrics["accuracy"]
        val_acc_history.append(val_acc)

        scheduler.step()
        print(f"Epoch {epoch:02d} Summary: Train Loss = {epoch_train_loss:.4f} | Val Loss = {epoch_val_loss:.4f} | Val Acc = {val_acc*100:.2f}% | Val AUC = {val_metrics['roc_auc']:.4f}")

        # Checkpoint best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  --> Saved new best checkpoint to: {CHECKPOINT_PATH}")

        # Early stopping
        if early_stopper(epoch_val_loss):
            print(f"[Early Stopping] Validation loss plateaued at epoch {epoch}. Stopping early.")
            break

    # 4. Final Evaluation on Independent Test Set
    print("\n[Step 4/5] Evaluating Best Model on Test Set...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True, map_location=device))
    model.eval()

    test_preds = []
    test_targets = []
    sample_fake_tensor = None
    sample_fake_raw = None

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            targets = labels.numpy().flatten()

            test_preds.extend(probs)
            test_targets.extend(targets)

            # Keep a sample fake image for Grad-CAM
            if sample_fake_tensor is None:
                for idx, lbl in enumerate(targets):
                    if lbl == 1.0:
                        sample_fake_tensor = images[idx : idx + 1]
                        # De-normalize image for display
                        img_np = images[idx].cpu().numpy().transpose(1, 2, 0)
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        sample_fake_raw = np.clip((img_np * std + mean) * 255, 0, 255).astype(np.uint8)
                        break

    test_metrics = calculate_metrics(np.array(test_targets), np.array(test_preds))
    print("\n" + "=" * 50)
    print("         FINAL IMAGE TEST METRICS REPORT         ")
    print("=" * 50)
    print(f"Accuracy:  {test_metrics['accuracy']*100:.2f}%")
    print(f"Precision: {test_metrics['precision']*100:.2f}%")
    print(f"Recall:    {test_metrics['recall']*100:.2f}%")
    print(f"F1-Score:  {test_metrics['f1']:.4f}")
    print(f"ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print("Confusion Matrix:")
    print(np.array(test_metrics["confusion_matrix"]))
    print("=" * 50)

    # 5. Visualizations & Explainability
    print("\n[Step 5/5] Generating Visualizations & Explainable AI (Grad-CAM)...")
    log_metrics_to_csv("Image_EfficientNetB0", test_metrics)
    plot_loss_and_accuracy(train_loss_history, val_loss_history, val_acc_history, "Image EfficientNet-B0")
    plot_roc_curve(np.array(test_targets), np.array(test_preds), "Image EfficientNet-B0")
    plot_confusion_matrix(np.array(test_metrics["confusion_matrix"]), "Image EfficientNet-B0")

    if sample_fake_tensor is not None and sample_fake_raw is not None:
        gradcam_save = GRADCAM_DIR / "image_gradcam_fake_face.png"
        generate_gradcam_heatmap(
            model, sample_fake_tensor, sample_fake_raw, gradcam_save, title="Image Model: Grad-CAM on Fake Face"
        )
        print(f"[Explainable AI] Generated Grad-CAM heatmap saved at: {gradcam_save}")

    print("\n[AURA Success] Image deepfake detection training pipeline completed successfully!")
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AURA Image Deepfake Detector")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 0.0001)")
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional sample limit for quick smoke testing")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint")
    parser.add_argument("--start-epoch", type=int, default=1, help="Epoch to resume from (default: 1)")
    parser.add_argument("--best-val-loss", type=float, default=None, help="Known best val loss from previous run (for checkpoint saving logic)")
    args = parser.parse_args()

    train_image_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        sample_limit=args.sample_limit,
        device_name=args.device,
        resume=args.resume,
        start_epoch=args.start_epoch,
        best_val_loss_init=args.best_val_loss,
    )
