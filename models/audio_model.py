"""
AURA - Audio Deepfake Detection Model
RawNet2 / Spectrogram-CNN hybrid architecture with Residual blocks,
Batch Normalization, Max Pooling, and Spectrogram Saliency Attribution.
"""

from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock2D(nn.Module):
    """Residual convolutional block with Batch Normalization and LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += res
        return self.act(out)


class AudioDeepfakeDetector(nn.Module):
    """
    Spectrogram-CNN / RawNet2 style 2D convolutional deepfake voice classifier.
    Input: (B, 1, 80, T) Mel Spectrogram tensor
    Output: (B, 1) raw logit (Sigmoid > 0.5 indicates Fake, <= 0.5 indicates Real).
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32, dropout: float = 0.3):
        super().__init__()

        # Stem Conv
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=5, stride=(1, 1), padding=2, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # (B, 32, 40, T/2)
        )

        # Residual Feature Blocks
        self.res1 = ResidualBlock2D(base_channels, base_channels * 2, stride=2)   # (B, 64, 20, T/4)
        self.res2 = ResidualBlock2D(base_channels * 2, base_channels * 4, stride=2) # (B, 128, 10, T/8)
        self.res3 = ResidualBlock2D(base_channels * 4, base_channels * 8, stride=2) # (B, 256, 5, T/16)

        # Global Statistics / Adaptive Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(base_channels * 8, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input x: (B, 1, 80, T)
        Output: (B, 1) logit
        """
        out = self.stem(x)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        pooled = self.global_pool(out)
        flat = torch.flatten(pooled, 1)
        logits = self.classifier(flat)
        return logits

    def predict_audio(self, spec_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Inference helper: accepts (1, 1, 80, T) or (1, 80, T)
        Returns Real/Fake, confidence %, probability.
        """
        self.eval()
        with torch.no_grad():
            if spec_tensor.dim() == 3:
                spec_tensor = spec_tensor.unsqueeze(0)

            logits = self.forward(spec_tensor)
            prob = float(torch.sigmoid(logits).item())
            is_fake = prob > 0.5
            prediction = "Fake Audio" if is_fake else "Real Audio"
            confidence = (prob if is_fake else (1.0 - prob)) * 100.0

            return {
                "prediction": prediction,
                "is_fake": is_fake,
                "confidence": round(confidence, 2),
                "fake_probability": round(prob, 4),
            }


def build_audio_model(device: torch.device = None) -> AudioDeepfakeDetector:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioDeepfakeDetector()
    return model.to(device)


if __name__ == "__main__":
    am = build_audio_model(torch.device("cpu"))
    sample_spec = torch.randn(2, 1, 80, 188)
    logits = am(sample_spec)
    pred = am.predict_audio(sample_spec[0])
    print("Audio logits shape:", logits.shape)
    print("Audio pred:", pred)
