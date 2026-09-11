"""
AURA - Standalone Video Deepfake Inference Script
Input: Path to MP4 video
Output: Frame-by-frame confidence timeline, final video prediction (Real/Fake Video), Confidence %
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
import matplotlib.pyplot as plt

from models.video_model import build_video_model, VideoDeepfakeDetector
from preprocessing.video_preprocess import extract_evenly_spaced_frames, detect_and_crop_face
from training.utils import LOGS_DIR

CHECKPOINT_PATH = PROJECT_ROOT / "saved_models" / "video_detector.pth"


def plot_frame_confidence_timeline(frame_confidences: list, video_name: str, save_path: Path):
    """Plots frame-by-frame deepfake confidence progression across video."""
    plt.figure(figsize=(7, 3.8))
    frames = list(range(1, len(frame_confidences) + 1))
    plt.plot(frames, frame_confidences, marker="o", color="crimson", linewidth=2, label="Frame Fake Probability (%)")
    plt.axhline(50.0, color="gray", linestyle="--", alpha=0.7, label="Decision Threshold (50%)")
    plt.fill_between(frames, frame_confidences, 50.0, where=[c >= 50.0 for c in frame_confidences],
                     color="red", alpha=0.15, interpolate=True)
    plt.fill_between(frames, frame_confidences, 50.0, where=[c < 50.0 for c in frame_confidences],
                     color="green", alpha=0.15, interpolate=True)

    plt.xlabel("Sampled Video Frame Index", fontweight="bold")
    plt.ylabel("Fake Confidence (%)", fontweight="bold")
    plt.ylim(0, 100)
    plt.title(f"AURA - Frame Confidence Timeline: {video_name}", fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def predict_video(video_path: str, model_path: str = None, num_frames: int = 10):
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found at: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = VideoDeepfakeDetector(pretrained=False).to(device)
    ckpt = Path(model_path) if model_path else CHECKPOINT_PATH

    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
        print(f"[Model Loaded] Checkpoint: {ckpt.name}")
    else:
        print(f"[Warning] Checkpoint not found at {ckpt}. Running with initialized pretrained weights.")
        model = build_video_model(device)

    model.eval()

    # Frame extraction & face detection
    print(f"[Video Analysis] Extracting {num_frames} frames and detecting faces from {path.name}...")
    raw_frames = extract_evenly_spaced_frames(path, num_frames=num_frames * 2)
    face_crops = []
    for f in raw_frames:
        crop = detect_and_crop_face(f)
        if crop is not None:
            face_crops.append(crop)
        if len(face_crops) == num_frames:
            break

    while len(face_crops) < num_frames:
        if len(face_crops) > 0:
            face_crops.append(face_crops[-1])
        else:
            face_crops.append(np.zeros((224, 224, 3), dtype=np.uint8))

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor_list = [tf(Image.fromarray(c)) for c in face_crops]
    video_tensor = torch.stack(tensor_list).unsqueeze(0).to(device)  # (1, 10, 3, 224, 224)

    # Predict
    result = model.predict_video(video_tensor)

    print("\n" + "=" * 50)
    print("        AURA VIDEO PREDICTION REPORT        ")
    print("=" * 50)
    print(f"File:               {path.name}")
    print(f"Prediction:         {result['prediction'].upper()}")
    print(f"Overall Confidence: {result['confidence']:.2f}%")
    print(f"Average Fake Prob:  {result['fake_probability']:.4f}")
    print("\nFrame-by-Frame Fake Confidences (%):")
    for idx, conf in enumerate(result["frame_confidences"], 1):
        bar = "#" * int(conf // 5)
        print(f"  Frame {idx:02d}: {conf:5.1f}% | {bar}")
    print("=" * 50)

    # Plot timeline
    timeline_path = LOGS_DIR / f"{path.stem}_timeline.png"
    plot_frame_confidence_timeline(result["frame_confidences"], path.name, timeline_path)
    print(f"[Timeline] Saved frame confidence graph to: {timeline_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict if a face video is Real or Deepfake")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input MP4 video")
    parser.add_argument("--weights", "-w", type=str, default=None, help="Optional model checkpoint path")
    args = parser.parse_args()

    predict_video(args.input, model_path=args.weights)
