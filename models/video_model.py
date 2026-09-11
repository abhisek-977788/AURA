"""
AURA - Video Deepfake Detection Model
EfficientNet-B0 frame feature backbone with temporal confidence aggregation
and Grad-CAM hook support.
"""

from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class VideoDeepfakeDetector(nn.Module):
    """
    Video deepfake detector that evaluates N face frames per video,
    computes frame-level logits, and aggregates into a final video classification.
    """

    def __init__(self, pretrained: bool = True, dropout_rate: float = 0.3):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        in_features = backbone.classifier[1].in_features  # 1280

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
        )

        self.gradient = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradient = grad

    def get_activations_gradient(self):
        return self.gradient

    def get_activations(self):
        return self.activations

    def forward_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Processes a batch of individual face frames.
        Input: (B, 3, 224, 224)
        Output: (B, 1) frame logits
        """
        feats = self.features(frames)
        if frames.requires_grad:
            self.activations = feats
            feats.register_hook(self.activations_hook)

        pooled = self.avgpool(feats)
        flattened = torch.flatten(pooled, 1)
        logits = self.classifier(flattened)
        return logits

    def forward(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of videos.
        Input video_tensor: (B, N, 3, 224, 224) where N is number of frames (e.g. 10)
        Output: (B, 1) aggregated video logits (mean of frame logits)
        """
        b, n, c, h, w = video_tensor.shape
        flat_frames = video_tensor.view(b * n, c, h, w)
        frame_logits = self.forward_frames(flat_frames)
        # Reshape to (B, N) and compute mean logit across frames
        video_logits = frame_logits.view(b, n).mean(dim=1, keepdim=True)
        return video_logits

    def predict_video(self, video_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Inference helper: accepts video tensor (1, N, 3, 224, 224) or (N, 3, 224, 224)
        Returns:
        - frame_confidences: list of float % for each frame
        - prediction: 'Real Video' or 'Fake Video'
        - confidence: overall confidence %
        """
        self.eval()
        with torch.no_grad():
            if video_tensor.dim() == 4:
                video_tensor = video_tensor.unsqueeze(0)  # (1, N, 3, 224, 224)

            b, n, c, h, w = video_tensor.shape
            flat_frames = video_tensor.view(b * n, c, h, w)
            frame_logits = self.forward_frames(flat_frames)
            frame_probs = torch.sigmoid(frame_logits).view(n).cpu().numpy()

            avg_prob = float(frame_probs.mean())
            is_fake = avg_prob > 0.5
            prediction = "Fake Video" if is_fake else "Real Video"
            confidence = (avg_prob if is_fake else (1.0 - avg_prob)) * 100.0

            return {
                "prediction": prediction,
                "is_fake": is_fake,
                "confidence": round(confidence, 2),
                "fake_probability": round(avg_prob, 4),
                "frame_confidences": [round(float(p) * 100.0, 2) for p in frame_probs],
            }


def build_video_model(device: torch.device = None) -> VideoDeepfakeDetector:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoDeepfakeDetector(pretrained=True)
    return model.to(device)


if __name__ == "__main__":
    vm = build_video_model(torch.device("cpu"))
    sample_vid = torch.randn(2, 10, 3, 224, 224)
    out = vm(sample_vid)
    print("Video output shape:", out.shape)
    pred_res = vm.predict_video(sample_vid[0])
    print("Inference result:", pred_res)
