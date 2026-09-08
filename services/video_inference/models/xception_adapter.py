"""
XceptionNet Model Adapter for Facial Manipulation Detection.
Ported from the official FaceForensics++ architecture (Rössler et al., CVPR 2019).
Uses Depthwise Separable Convolutions with a custom binary output head.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from packages.schemas.analysis_result import ModelInferenceResult, ModalityType
from services.video_inference.models.mesonet_adapter import VideoModelAdapter


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1, padding: int = 0, bias: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters: int, out_filters: int, reps: int, strides: int = 1, start_with_relu: bool = True, grow_first: bool = True):
        super().__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for _ in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))

        self.rep = nn.Sequential(*rep)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skipbn(self.skip(inp))
        else:
            skip = inp
        x += skip
        return x


class XceptionNet(nn.Module):
    """FaceForensics++ core Xception architecture."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.entry_flow = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            Block(64, 128, 2, 2, start_with_relu=False, grow_first=True),
            Block(128, 256, 2, 2, start_with_relu=True, grow_first=True),
            Block(256, 728, 2, 2, start_with_relu=True, grow_first=True),
        )
        self.middle_flow = nn.Sequential(
            Block(728, 728, 3, 1, start_with_relu=True, grow_first=True),
            Block(728, 728, 3, 1, start_with_relu=True, grow_first=True),
            Block(728, 728, 3, 1, start_with_relu=True, grow_first=True),
            Block(728, 728, 3, 1, start_with_relu=True, grow_first=True),
        )
        self.exit_flow = nn.Sequential(
            Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False),
            SeparableConv2d(1024, 1536, 3, 1, 1),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            SeparableConv2d(1536, 2048, 3, 1, 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class XceptionAdapter(VideoModelAdapter):
    def __init__(self, weights_path: Path | str | None = None) -> None:
        super().__init__(model_name="FaceForensics-Xception", model_version="1.0.0")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = XceptionNet(num_classes=2).to(self.device)

        if weights_path and Path(weights_path).exists():
            try:
                ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
                state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
                self.model.load_state_dict(state_dict)
                self._is_loaded = True
            except Exception:
                self._is_loaded = False
        self.model.eval()

    async def predict(self, face_crops: list[np.ndarray]) -> ModelInferenceResult:
        start_time = time.perf_counter()

        if not face_crops:
            return ModelInferenceResult(
                model_name=self.model_name,
                model_version=self.model_version,
                modality=ModalityType.VIDEO,
                synthetic_score=0.5,
                confidence=0.0,
                limitations=["No face crops supplied for XceptionNet"],
            )

        # Batch preprocessing
        tensors = []
        for crop in face_crops[:8]:
            # Convert to float (C, H, W) normalized to [-1, 1]
            c = crop.astype(np.float32) / 255.0
            # Resize if needed
            if c.shape[:2] != (128, 128):
                # Basic crop resize
                import cv2
                c = cv2.resize(c, (128, 128))
            c = (c - 0.5) / 0.5
            tensors.append(np.transpose(c, (2, 0, 1)))

        batch = torch.tensor(np.array(tensors), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(batch)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

        mean_synth = float(np.mean(probs))
        latency = (time.perf_counter() - start_time) * 1000

        return ModelInferenceResult(
            model_name=self.model_name,
            model_version=self.model_version,
            modality=ModalityType.VIDEO,
            synthetic_score=round(mean_synth, 4),
            confidence=0.88 if self._is_loaded else 0.70,
            inference_latency_ms=round(latency, 2),
            raw_scores={"evaluated_faces": len(tensors)},
            limitations=[] if self._is_loaded else ["XceptionNet running on initialized backbone weights."],
        )
