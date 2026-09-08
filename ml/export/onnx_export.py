"""
ONNX Model Export and Validation Pipeline for AURA.
Prepares trained PyTorch weights for low-latency server and browser WASM execution.
"""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn


def export_video_model_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, ...] = (1, 3, 64, 64),
) -> Path:
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    dummy_input = torch.randn(*input_shape, requires_grad=False)
    torch.onnx.export(
        model,
        dummy_input,
        str(out_p),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_tensor"],
        output_names=["synthetic_probabilities"],
        dynamic_axes={"input_tensor": {0: "batch_size"}, "synthetic_probabilities": {0: "batch_size"}},
    )
    print(f"[AURA ONNX] Exported model successfully to: {out_p}")
    return out_p


def export_audio_model_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, ...] = (1, 1, 32000),
) -> Path:
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    dummy_input = torch.randn(*input_shape, requires_grad=False)
    torch.onnx.export(
        model,
        dummy_input,
        str(out_p),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["waveform"],
        output_names=["logits"],
        dynamic_axes={"waveform": {0: "batch_size", 2: "time_steps"}},
    )
    print(f"[AURA ONNX] Exported audio model to: {out_p}")
    return out_p
