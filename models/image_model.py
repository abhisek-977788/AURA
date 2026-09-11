"""
AURA - Image Deepfake Detection Model
EfficientNet-B0 transfer learning backbone with custom classification head
and gradient hook support for Explainable AI (Grad-CAM).
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class ImageDeepfakeDetector(nn.Module):
    """
    Binary deepfake face classifier using pretrained EfficientNet-B0.
    Output: Single logit (Sigmoid > 0.5 indicates Fake, <= 0.5 indicates Real).
    """

    def __init__(self, pretrained: bool = True, dropout_rate: float = 0.3):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        # Retain feature extractor (convolutional layers)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        in_features = backbone.classifier[1].in_features  # 1280 for EfficientNet-B0

        # Custom binary classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
        )

        # Hook placeholders for Grad-CAM
        self.gradient = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradient = grad

    def get_activations_gradient(self):
        return self.gradient

    def get_activations(self):
        return self.activations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input x: (B, 3, 256, 256)
        Output: (B, 1) raw logits
        """
        feats = self.features(x)

        # Register hook on the last convolutional feature map for Grad-CAM
        if x.requires_grad:
            self.activations = feats
            h = feats.register_hook(self.activations_hook)

        pooled = self.avgpool(feats)
        flattened = torch.flatten(pooled, 1)
        logits = self.classifier(flattened)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns calibrated fake probability between 0.0 and 1.0."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


def build_image_model(device: torch.device = None) -> ImageDeepfakeDetector:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageDeepfakeDetector(pretrained=True)
    return model.to(device)


if __name__ == "__main__":
    m = build_image_model(torch.device("cpu"))
    sample_input = torch.randn(2, 3, 256, 256)
    logits = m(sample_input)
    probs = m.predict_proba(sample_input)
    print("Logits shape:", logits.shape, "Probs:", probs.squeeze().tolist())
