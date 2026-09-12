"""
AURA - Video Deepfake Detector Training Pipeline
Dataset: FaceForensics++ C23 (100 Real, 100 Fake balanced)
Model: EfficientNet-B0 Face Frame Evaluator + Temporal Pooling
Loss: BCEWithLogitsLoss, Optimizer: AdamW, Scheduler: CosineAnnealingLR
Outputs: Metrics, ROC Curves, Confusion Matrix, Per-Video Confidence Chart, Grad-CAM.
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

from preprocessing.video_preprocess import get_video_dataloaders
from models.video_model import build_video_model
from training.utils import (
    calculate_metrics,
    log_metrics_to_csv,
    plot_loss_and_accuracy,
    plot_roc_curve,
    plot_confusion_matrix,
    generate_gradcam_heatmap,
    EarlyStopping,
    GRADCAM_DIR,
    LOGS_DIR,
)

SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = SAVED_MODELS_DIR / "video_detector.pth"


def plot_per_video_confidence_chart(
    video_predictions: list,
    save_path: Path = None,
):
    """Generates a per-video confidence score distribution bar chart."""
    if save_path is None:
        save_path = LOGS_DIR / "video_confidence_distribution.png"

    real_confs = [v["confidence"] if not v["is_fake"] else 100.0 - v["confidence"] for v in video_predictions if not v["ground_truth_fake"]]
    fake_confs = [v["confidence"] if v["is_fake"] else 100.0 - v["confidence"] for v in video_predictions if v["ground_truth_fake"]]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 100, 20)
    ax.hist(real_confs, bins=bins, alpha=0.7, color="tab:green", label="Real Videos (Authentic)")
    ax.hist(fake_confs, bins=bins, alpha=0.7, color="tab:red", label="Fake Videos (Manipulated)")
    ax.set_xlabel("Predicted Fake Probability Score (%)", fontweight="bold")
    ax.set_ylabel("Number of Videos", fontweight="bold")
    ax.set_title("AURA - Per-Video Deepfake Confidence Distribution", fontweight="bold", pad=12)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved per-video confidence chart to: {save_path}")


def train_video_pipeline(
    epochs: int = 15,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    video_count: int = 100,
    device_name: str = None,
):
    print("=" * 70)
    print("      AURA — VIDEO DEEPFAKE DETECTION TRAINING PIPELINE       ")
    print("=" * 70)

    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AURA Device] Using compute device: {device}")

    # 1. Video Dataloaders
    print(f"\n[Step 1/5] Extracting & Preprocessing {video_count} Real & {video_count} Fake FaceForensics++ Videos...")
    train_loader, val_loader = get_video_dataloaders(
        batch_size=batch_size, real_count=video_count, fake_count=video_count
    )

    # 2. Build Video Model
    print("\n[Step 2/5] Initializing Video EfficientNet-B0 Architecture...")
    model = build_video_model(device=device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    early_stopper = EarlyStopping(patience=5)

    train_loss_history = []
    val_loss_history = []
    val_acc_history = []
    best_val_loss = float("inf")

    # 3. Training Loop
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\n[Step 3/5] Starting Video Model Training for {epochs} Epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]")

        for video_tensors, labels in pbar:
            video_tensors = video_tensors.to(device)  # (B, 10, 3, 224, 224)
            labels = labels.to(device).unsqueeze(1)    # (B, 1)

            optimizer.zero_grad()
            logits = model(video_tensors)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * video_tensors.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for video_tensors, labels in val_loader:
                video_tensors = video_tensors.to(device)
                labels = labels.to(device).unsqueeze(1)

                logits = model(video_tensors)
                loss = criterion(logits, labels)
                val_loss += loss.item() * video_tensors.size(0)

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
            print(f"  --> Saved new best video checkpoint to: {CHECKPOINT_PATH}")

        if early_stopper(epoch_val_loss):
            print(f"[Early Stopping] Validation loss reached plateau at epoch {epoch}.")
            break

    # 4. Final Validation Evaluation & Detailed Reporting
    print("\n[Step 4/5] Evaluating Best Video Model...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True, map_location=device))
    model.eval()

    eval_preds = []
    eval_targets = []
    video_summary_records = []
    sample_fake_frame_tensor = None
    sample_fake_frame_raw = None

    with torch.no_grad():
        for video_tensors, labels in val_loader:
            for i in range(video_tensors.size(0)):
                single_vid = video_tensors[i].to(device)  # (10, 3, 224, 224)
                lbl = float(labels[i].item())
                res = model.predict_video(single_vid)
                res["ground_truth_fake"] = (lbl == 1.0)
                video_summary_records.append(res)

                eval_preds.append(res["fake_probability"])
                eval_targets.append(lbl)

                if sample_fake_frame_tensor is None and lbl == 1.0:
                    # Capture the most suspicious frame from the video
                    fake_probs = res["frame_confidences"]
                    max_idx = int(np.argmax(fake_probs))
                    sample_fake_frame_tensor = single_vid[max_idx : max_idx + 1]  # (1, 3, 224, 224)
                    raw_np = single_vid[max_idx].cpu().numpy().transpose(1, 2, 0)
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    sample_fake_frame_raw = np.clip((raw_np * std + mean) * 255, 0, 255).astype(np.uint8)

    video_metrics = calculate_metrics(np.array(eval_targets), np.array(eval_preds))
    print("\n" + "=" * 50)
    print("         FINAL VIDEO MODEL METRICS REPORT         ")
    print("=" * 50)
    print(f"Accuracy:  {video_metrics['accuracy']*100:.2f}%")
    print(f"Precision: {video_metrics['precision']*100:.2f}%")
    print(f"Recall:    {video_metrics['recall']*100:.2f}%")
    print(f"F1-Score:  {video_metrics['f1']:.4f}")
    print(f"ROC-AUC:   {video_metrics['roc_auc']:.4f}")
    print("Confusion Matrix:")
    print(np.array(video_metrics["confusion_matrix"]))
    print("=" * 50)

    # 5. Visualizations & Explainability
    print("\n[Step 5/5] Generating Video Visualizations & Grad-CAM...")
    log_metrics_to_csv("Video_EfficientNetB0", video_metrics)
    plot_loss_and_accuracy(train_loss_history, val_loss_history, val_acc_history, "Video EfficientNet-B0")
    plot_roc_curve(np.array(eval_targets), np.array(eval_preds), "Video EfficientNet-B0")
    plot_confusion_matrix(np.array(video_metrics["confusion_matrix"]), "Video EfficientNet-B0")
    plot_per_video_confidence_chart(video_summary_records)

    if sample_fake_frame_tensor is not None and sample_fake_frame_raw is not None:
        gradcam_save = GRADCAM_DIR / "video_gradcam_fake_frame.png"

        class FrameWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return self.m.forward_frames(x)

            def get_activations_gradient(self):
                return self.m.get_activations_gradient()

            def get_activations(self):
                return self.m.get_activations()

        frame_wrapper = FrameWrapper(model)
        generate_gradcam_heatmap(
            frame_wrapper,
            sample_fake_frame_tensor,
            sample_fake_frame_raw,
            gradcam_save,
            title="Video Model: Grad-CAM on Suspicious Fake Frame",
        )
        print(f"[Explainable AI] Generated video frame Grad-CAM heatmap saved at: {gradcam_save}")

    print("\n[AURA Success] Video deepfake detection training pipeline completed successfully!")
    return video_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AURA Video Deepfake Detector")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--video-count", type=int, default=100, help="Number of real and fake videos to use (default: 100)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 0.0001)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    args = parser.parse_args()

    train_video_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        video_count=args.video_count,
        device_name=args.device,
    )
