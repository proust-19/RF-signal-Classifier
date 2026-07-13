"""ONNX export and quantization for embedded deployment.

Supports full-precision and int8-quantized ONNX models
suitable for edge inference on satellite payloads.
"""

import torch
import torch.nn as nn
import numpy as np


def export_onnx(
    model: nn.Module,
    save_path: str,
    input_shape: tuple = (1, 2, 1024),
    opset_version: int = 13,
) -> dict:
    """Export PyTorch model to ONNX.

    Returns export metadata including model size.
    """
    model.eval()
    dummy = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        dummy,
        save_path,
        input_names=["iq_input"],
        output_names=["prediction"],
        dynamic_axes={"iq_input": {0: "batch"}, "prediction": {0: "batch"}},
        opset_version=opset_version,
    )

    import os

    size_mb = os.path.getsize(save_path) / (1024 * 1024)

    return {
        "path": save_path,
        "size_mb": round(size_mb, 2),
        "input_shape": input_shape,
        "opset": opset_version,
    }


def quantize_onnx(
    model_path: str,
    save_path: str,
    calibration_data: np.ndarray | None = None,
) -> dict:
    """Apply dynamic quantization to ONNX model for int8 inference.

    Reduces model size ~4x and speeds up inference on edge hardware.
    """
    try:
        import onnxruntime as ort
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        raise ImportError(
            "onnxruntime and onnxruntime.quantization required. "
            "Install: pip install onnxruntime"
        )

    quantize_dynamic(
        model_path,
        save_path,
        weight_type=QuantType.QInt8,
    )

    import os

    orig_size = os.path.getsize(model_path) / (1024 * 1024)
    quant_size = os.path.getsize(save_path) / (1024 * 1024)

    return {
        "path": save_path,
        "original_size_mb": round(orig_size, 2),
        "quantized_size_mb": round(quant_size, 2),
        "compression_ratio": round(orig_size / max(quant_size, 0.01), 1),
    }


def benchmark_onnx(model_path: str, n_runs: int = 100) -> dict:
    """Benchmark ONNX model inference latency."""
    import onnxruntime as ort
    import time

    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    dummy = np.random.randn(1, 2, 1024).astype(np.float32)

    # Warmup
    for _ in range(10):
        session.run(None, {input_name: dummy})

    # Benchmark
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        latencies.append(time.perf_counter() - t0)

    latencies = np.array(latencies) * 1000  # ms

    return {
        "mean_ms": round(float(latencies.mean()), 3),
        "std_ms": round(float(latencies.std()), 3),
        "p50_ms": round(float(np.percentile(latencies, 50)), 3),
        "p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "p99_ms": round(float(np.percentile(latencies, 99)), 3),
        "throughput_fps": round(1000 / float(latencies.mean()), 1),
    }


def export_with_quantization(
    model: nn.Module,
    onnx_path: str,
    quantized_path: str | None = None,
) -> dict:
    """Full export pipeline: FP32 ONNX + INT8 quantized."""
    if quantized_path is None:
        quantized_path = onnx_path.replace(".onnx", "_int8.onnx")

    fp32_info = export_onnx(model, onnx_path)
    quant_info = quantize_onnx(onnx_path, quantized_path)

    return {
        "fp32": fp32_info,
        "int8": quant_info,
    }
