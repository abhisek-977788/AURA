"""
AURA - Standalone Image Deepfake Inference Script
Input: Path to face image (JPG, PNG, WebP)
Output: Prediction (Real/Fake), Confidence %, and Grad-CAM Heatmap
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from models.image_model import build_image_model, ImageDeepfakeDetector
from training.utils import generate_gradcam_heatmap, GRADCAM_DIR

CHECKPOINT_PATH = PROJECT_ROOT / "saved_models" / "image_detector.pth"


def predict_image(image_path: str, model_path: str = None, save_gradcam: bool = True):
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ImageDeepfakeDetector(pretrained=False).to(device)
    ckpt = Path(model_path) if model_path else CHECKPOINT_PATH

    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
        print(f"[Model Loaded] Checkpoint: {ckpt.name}")
    else:
        print(f"[Warning] Checkpoint not found at {ckpt}. Running with initialized pretrained weights.")
        model = build_image_model(device)

    model.eval()

    # Preprocessing
    raw_img = Image.open(path).convert("RGB")
    raw_img_resized = raw_img.resize((256, 256))
    raw_np = np.array(raw_img_resized)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(raw_img_resized).unsqueeze(0).to(device)

    # Forward
    with torch.no_grad():
        logits = model(tensor)
        fake_prob = float(torch.sigmoid(logits).item())

    is_fake = fake_prob > 0.5
    prediction = "Fake" if is_fake else "Real"
    confidence = (fake_prob if is_fake else (1.0 - fake_prob)) * 100.0

    print("\n" + "=" * 45)
    print("        AURA IMAGE PREDICTION REPORT        ")
    print("=" * 45)
    print(f"File:        {path.name}")
    print(f"Prediction:  {prediction.upper()}")
    print(f"Confidence:  {confidence:.2f}%")
    print(f"Probability: {fake_prob:.4f} (1.0 = Synthetic)")
    print("=" * 45)

    if save_gradcam:
        out_path = GRADCAM_DIR / f"{path.stem}_gradcam.png"
        generate_gradcam_heatmap(
            model,
            tensor,
            raw_np,
            out_path,
            title=f"AURA Grad-CAM: {path.name} ({prediction} - {confidence:.1f}%)",
        )
        print(f"[Grad-CAM] Visual explanation saved to: {out_path}")

    return {
        "prediction": prediction,
        "is_fake": is_fake,
        "confidence": round(confidence, 2),
        "fake_probability": round(fake_prob, 4),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict if a face image is Real or Deepfake")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", "-w", type=str, default=None, help="Optional model checkpoint path")
    parser.add_argument("--no-gradcam", action="store_true", help="Disable Grad-CAM heatmap generation")
    args = parser.parse_args()

    predict_image(args.input, model_path=args.weights, save_gradcam=not args.no_gradcam)
