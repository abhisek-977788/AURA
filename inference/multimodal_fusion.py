"""
AURA - Multimodal Decision Fusion Engine
Combines predictions from Image, Video, and Audio deepfake detectors
using calibrated weighted score fusion:
Weights:
- Image: 0.30
- Video: 0.40
- Audio: 0.30

Outputs standard JSON report.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def fuse_predictions(
    image_fake_prob: float,
    video_fake_prob: float,
    audio_fake_prob: float,
    weight_image: float = 0.30,
    weight_video: float = 0.40,
    weight_audio: float = 0.30,
) -> Dict[str, Any]:
    """
    Computes weighted multimodal fusion decision.
    - Input probabilities: 0.0 (Authentic/Real) to 1.0 (Manipulated/Fake)
    - Returns JSON structure requested by AURA specification.
    """
    # Normalize weights if necessary
    total_weight = weight_image + weight_video + weight_audio
    w_img = weight_image / total_weight
    w_vid = weight_video / total_weight
    w_aud = weight_audio / total_weight

    # Individual decisions
    image_pred = "Fake" if image_fake_prob >= 0.5 else "Real"
    video_pred = "Fake" if video_fake_prob >= 0.5 else "Real"
    audio_pred = "Fake" if audio_fake_prob >= 0.5 else "Real"

    # Multimodal weighted probability
    fused_fake_prob = (
        (w_img * image_fake_prob) +
        (w_vid * video_fake_prob) +
        (w_aud * audio_fake_prob)
    )

    is_final_fake = fused_fake_prob >= 0.5
    final_pred = "Fake" if is_final_fake else "Real"
    final_confidence = (fused_fake_prob if is_final_fake else (1.0 - fused_fake_prob)) * 100.0

    result = {
        "image_prediction": image_pred,
        "video_prediction": video_pred,
        "audio_prediction": audio_pred,
        "final_prediction": final_pred,
        "confidence": round(final_confidence, 1),
        "fused_probability": round(fused_fake_prob, 4),
        "weights": {
            "image": weight_image,
            "video": weight_video,
            "audio": weight_audio,
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="AURA Multimodal Deepfake Decision Fusion")
    parser.add_argument("--image-prob", type=float, default=None, help="Image model fake probability (0.0 - 1.0)")
    parser.add_argument("--video-prob", type=float, default=None, help="Video model fake probability (0.0 - 1.0)")
    parser.add_argument("--audio-prob", type=float, default=None, help="Audio model fake probability (0.0 - 1.0)")

    # File based shortcuts
    parser.add_argument("--image-file", type=str, default=None, help="Path to image file")
    parser.add_argument("--video-file", type=str, default=None, help="Path to video file")
    parser.add_argument("--audio-file", type=str, default=None, help="Path to audio file")

    args = parser.parse_args()

    img_prob = args.image_prob
    vid_prob = args.video_prob
    aud_prob = args.audio_prob

    # If media files are passed, run inference directly
    if args.image_file:
        from inference.predict_image import predict_image
        res = predict_image(args.image_file, save_gradcam=False)
        img_prob = res["fake_probability"]

    if args.video_file:
        from inference.predict_video import predict_video
        res = predict_video(args.video_file)
        vid_prob = res["fake_probability"]

    if args.audio_file:
        from inference.predict_audio import predict_audio
        res = predict_audio(args.audio_file)
        aud_prob = res["fake_probability"]

    # Fallback to defaults if not provided
    if img_prob is None:
        img_prob = 0.95
    if vid_prob is None:
        vid_prob = 0.98
    if aud_prob is None:
        aud_prob = 0.05

    fusion_report = fuse_predictions(img_prob, vid_prob, aud_prob)

    print("\n" + "=" * 50)
    print("       AURA MULTIMODAL FUSION RESULT JSON        ")
    print("=" * 50)
    # Output exact requested JSON keys format
    formatted_output = {
        "image_prediction": fusion_report["image_prediction"],
        "video_prediction": fusion_report["video_prediction"],
        "audio_prediction": fusion_report["audio_prediction"],
        "final_prediction": fusion_report["final_prediction"],
        "confidence": fusion_report["confidence"],
    }
    print(json.dumps(formatted_output, indent=2))
    print("=" * 50)


if __name__ == "__main__":
    main()
