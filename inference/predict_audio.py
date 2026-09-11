"""
AURA - Standalone Audio Deepfake Inference Script
Input: Path to audio file (WAV, MP3, FLAC)
Output: Real/Fake Audio, Confidence %, and Mel Spectrogram Plot
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt

from models.audio_model import build_audio_model, AudioDeepfakeDetector
from preprocessing.audio_preprocess import (
    load_and_preprocess_audio,
    extract_features_from_waveform,
    TARGET_SAMPLE_RATE,
)
from training.utils import LOGS_DIR

CHECKPOINT_PATH = PROJECT_ROOT / "saved_models" / "audio_detector.pth"


def plot_inference_spectrogram(mel_spec: np.ndarray, audio_name: str, pred_label: str, conf: float, save_path: Path):
    """Plots and saves the input audio Mel Spectrogram."""
    plt.figure(figsize=(8, 3.5))
    plt.imshow(mel_spec, aspect="auto", origin="lower", cmap="magma")
    plt.title(f"AURA Audio Spectrogram: {audio_name} ({pred_label} - {conf:.1f}%)", fontweight="bold")
    plt.ylabel("Mel Frequency Bands", fontweight="bold")
    plt.xlabel("Time Frames (Hop 256)", fontweight="bold")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def predict_audio(audio_path: str, model_path: str = None):
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found at: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = AudioDeepfakeDetector().to(device)
    ckpt = Path(model_path) if model_path else CHECKPOINT_PATH

    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
        print(f"[Model Loaded] Checkpoint: {ckpt.name}")
    else:
        print(f"[Warning] Checkpoint not found at {ckpt}. Running with initialized weights.")
        model = build_audio_model(device)

    model.eval()

    # Preprocess
    print(f"[Audio Analysis] Resampling 16kHz & extracting Mel Spectrogram from {path.name}...")
    segments = load_and_preprocess_audio(path)
    if not segments:
        raise ValueError(f"Could not extract audio samples from {path}")

    # Use first representative segment or average across segments
    seg_probs = []
    first_mel = None
    with torch.no_grad():
        for seg in segments:
            feats = extract_features_from_waveform(seg, sr=TARGET_SAMPLE_RATE)
            mel = feats["mel"]
            if first_mel is None:
                first_mel = mel

            expected_len = 188
            if mel.shape[1] < expected_len:
                mel = np.pad(mel, ((0, 0), (0, expected_len - mel.shape[1])), mode="edge")
            else:
                mel = mel[:, :expected_len]

            tensor = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 80, 188)
            logits = model(tensor)
            prob = float(torch.sigmoid(logits).item())
            seg_probs.append(prob)

    avg_fake_prob = float(np.mean(seg_probs))
    is_fake = avg_fake_prob > 0.5
    prediction = "Fake Audio" if is_fake else "Real Audio"
    confidence = (avg_fake_prob if is_fake else (1.0 - avg_fake_prob)) * 100.0

    print("\n" + "=" * 50)
    print("        AURA AUDIO PREDICTION REPORT        ")
    print("=" * 50)
    print(f"File:               {path.name}")
    print(f"Prediction:         {prediction.upper()}")
    print(f"Confidence:         {confidence:.2f}%")
    print(f"Fake Probability:   {avg_fake_prob:.4f}")
    print(f"Analyzed Segments:  {len(segments)} (3.0s windows)")
    print("=" * 50)

    # Plot spectrogram
    spec_path = LOGS_DIR / f"{path.stem}_spectrogram.png"
    plot_inference_spectrogram(first_mel, path.name, prediction, confidence, spec_path)
    print(f"[Spectrogram] Saved visualization to: {spec_path}")

    return {
        "prediction": prediction,
        "is_fake": is_fake,
        "confidence": round(confidence, 2),
        "fake_probability": round(avg_fake_prob, 4),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict if an audio voice clip is Real or Deepfake")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input audio (WAV, MP3, FLAC)")
    parser.add_argument("--weights", "-w", type=str, default=None, help="Optional model checkpoint path")
    args = parser.parse_args()

    predict_audio(args.input, model_path=args.weights)
