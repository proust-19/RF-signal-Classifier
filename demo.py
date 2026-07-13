"""End-to-end demo: train -> export -> quantize -> benchmark."""

import time
import numpy as np

from src.dataset import generate_synthetic_signals, create_dataloaders
from src.model import create_model
from src.train import train
from src.export_onnx import export_with_quantization, benchmark_onnx


def main():
    print("=" * 60)
    print("RF Signal Classifier - Full Pipeline Demo")
    print("=" * 60)

    # 1. Generate synthetic data
    print("\n[1/5] Generating 10k synthetic RF signals...")
    t0 = time.time()
    signals, labels = generate_synthetic_signals(
        n_samples=10000,
        snr_db_range=(0, 20),
        signal_length=1024,
    )
    print(f"  Signals: {signals.shape}, Labels: {labels.shape}")
    print(f"  Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    print(f"  Generated in {time.time() - t0:.1f}s")

    # 2. Create dataloaders
    print("\n[2/5] Creating train/val splits...")
    train_loader, val_loader = create_dataloaders(signals, labels, batch_size=64)
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 3. Train
    print("\n[3/5] Training CNN (20 epochs)...")
    model = create_model("cnn", num_classes=6)
    results = train(
        model,
        train_loader,
        val_loader,
        epochs=20,
        lr=1e-3,
        save_dir="checkpoints",
    )
    print(f"  Best val accuracy: {results['best_val_acc']:.2%}")
    print(f"  Model params: {model.count_parameters():,}")

    # 4. Export ONNX + quantize
    import os

    os.makedirs("models", exist_ok=True)
    print("\n[4/5] Exporting ONNX + INT8 quantization...")
    export_info = export_with_quantization(
        model,
        "models/rf_classifier.onnx",
        "models/rf_classifier_int8.onnx",
    )
    print(f"  FP32: {export_info['fp32']['size_mb']} MB")
    print(f"  INT8: {export_info['int8']['quantized_size_mb']} MB")
    print(f"  Compression: {export_info['int8']['compression_ratio']}x")

    # 5. Benchmark
    print("\n[5/5] Benchmarking inference latency...")
    bench = benchmark_onnx("models/rf_classifier_int8.onnx", n_runs=200)
    print(f"  Mean: {bench['mean_ms']} ms")
    print(f"  P50:  {bench['p50_ms']} ms")
    print(f"  P95:  {bench['p95_ms']} ms")
    print(f"  P99:  {bench['p99_ms']} ms")
    print(f"  Throughput: {bench['throughput_fps']} fps")

    # Summary
    print("\n" + "=" * 60)
    print("DONE - All steps passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
