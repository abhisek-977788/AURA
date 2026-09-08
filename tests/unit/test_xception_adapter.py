"""
Unit tests for FaceForensics XceptionNet model adapter.
"""

import pytest
import numpy as np
from services.video_inference.models.xception_adapter import XceptionAdapter, XceptionNet


def test_xception_architecture_forward():
    import torch
    model = XceptionNet(num_classes=2)
    model.eval()
    dummy = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, 2)


@pytest.mark.asyncio
async def test_xception_adapter_predict():
    adapter = XceptionAdapter()
    # Provide 2 dummy face crops
    face_crops = [
        np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
        np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
    ]
    res = await adapter.predict(face_crops)
    assert res.model_name == "FaceForensics-Xception"
    assert 0.0 <= res.synthetic_score <= 1.0
    assert res.confidence > 0.5
    assert res.inference_latency_ms is not None
