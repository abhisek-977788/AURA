"""
AURA - Training Utilities & Evaluation Engine
Provides metric computation, visualization generation (Loss curves, ROC curves,
Confusion Matrices, Grad-CAM heatmaps), and CSV logging.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
import cv2

RESULTS_DIR = Path("D:/AURA/results")
METRICS_CSV_PATH = RESULTS_DIR / "metrics.csv"
CONFUSION_MATRIX_DIR = RESULTS_DIR / "confusion_matrix"
ROC_CURVES_DIR = RESULTS_DIR / "roc_curves"
GRADCAM_DIR = RESULTS_DIR / "gradcam"
LOGS_DIR = RESULTS_DIR / "training_logs"

# Ensure output directories exist
for p in [RESULTS_DIR, CONFUSION_MATRIX_DIR, ROC_CURVES_DIR, GRADCAM_DIR, LOGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def calculate_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Computes comprehensive binary classification metrics.
    """
    y_true = np.array(y_true).astype(int)
    y_pred_probs = np.array(y_pred_probs).astype(float)
    y_pred = (y_pred_probs >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_pred_probs)
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(y_true, y_pred, labels=[0, 1], target_names=["Real", "Fake"], output_dict=True, zero_division=0)

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def log_metrics_to_csv(model_name: str, metrics: Dict[str, Any], csv_path: Path = METRICS_CSV_PATH) -> None:
    """Appends run evaluation metrics to results/metrics.csv."""
    record = {
        "model_name": model_name,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
    }
    df = pd.DataFrame([record])
    if not csv_path.exists():
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode="a", header=False, index=False)
    print(f"[Metrics Logger] Saved metrics for {model_name} to {csv_path}")


def plot_loss_and_accuracy(
    train_losses: List[float],
    val_losses: List[float],
    val_accs: List[float],
    model_name: str,
    save_path: Optional[Path] = None,
) -> None:
    """Plots and saves training/validation loss curves and accuracy progression."""
    if save_path is None:
        save_path = LOGS_DIR / f"{model_name.lower().replace(' ', '_')}_training_curves.png"

    epochs = range(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Loss axis
    color = "tab:red"
    ax1.set_xlabel("Epochs", fontweight="bold")
    ax1.set_ylabel("Loss", color=color, fontweight="bold")
    ax1.plot(epochs, train_losses, "r--", label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_losses, "r-", label="Val Loss", linewidth=2)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Accuracy axis
    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Validation Accuracy", color=color, fontweight="bold")
    ax2.plot(epochs, val_accs, "b-o", label="Val Accuracy", linewidth=2, markersize=4)
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title(f"AURA - {model_name} Training Progression", fontweight="bold", pad=12)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved training curves to: {save_path}")


def plot_roc_curve(y_true: np.ndarray, y_pred_probs: np.ndarray, model_name: str, save_path: Optional[Path] = None) -> None:
    """Plots and saves the ROC Curve with AUC score."""
    if save_path is None:
        save_path = ROC_CURVES_DIR / f"{model_name.lower().replace(' ', '_')}_roc.png"

    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    auc = roc_auc_score(y_true, y_pred_probs)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontweight="bold")
    plt.ylabel("True Positive Rate", fontweight="bold")
    plt.title(f"ROC Curve - {model_name}", fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved ROC curve to: {save_path}")


def plot_confusion_matrix(cm: np.ndarray, model_name: str, save_path: Optional[Path] = None) -> None:
    """Plots and saves the confusion matrix heatmap."""
    if save_path is None:
        save_path = CONFUSION_MATRIX_DIR / f"{model_name.lower().replace(' ', '_')}_cm.png"

    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
        cbar=False,
    )
    plt.xlabel("Predicted Class", fontweight="bold")
    plt.ylabel("True Class", fontweight="bold")
    plt.title(f"Confusion Matrix - {model_name}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved confusion matrix to: {save_path}")


def generate_gradcam_heatmap(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    original_image_rgb: np.ndarray,
    save_path: Path,
    title: str = "Grad-CAM Explainability",
) -> np.ndarray:
    """
    Computes Grad-CAM on the model's hooked convolutional layer and generates
    an overlaid visualization for Explainable AI.
    - input_tensor: (1, 3, H, W) requires_grad=True
    - original_image_rgb: (H, W, 3) uint8 RGB array
    """
    model.eval()
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    logits = model(input_tensor)
    prob = torch.sigmoid(logits).item()

    # Target: Fake class logit
    model.zero_grad()
    logits.backward()

    gradients = model.get_activations_gradient()
    activations = model.get_activations()

    if gradients is None or activations is None:
        print("[Grad-CAM] Warning: hooks did not capture gradients/activations.")
        return original_image_rgb

    # Global average pooling over gradients
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # Weight activations by gradients
    for i in range(activations.size(1)):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze().detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)  # ReLU
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)

    h, w, _ = original_image_rgb.shape
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = np.uint8(0.6 * original_image_rgb + 0.4 * heatmap_rgb)

    # Save comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original_image_rgb)
    axes[0].set_title(f"Input Face ({prob*100:.1f}% Fake)", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM Manipulation Saliency", fontweight="bold")
    axes[1].axis("off")

    plt.suptitle(title, fontweight="bold")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

    return overlay


class EarlyStopping:
    """Early stops training when validation loss stops improving."""

    def __init__(self, patience: int = 5, delta: float = 1e-4):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop
