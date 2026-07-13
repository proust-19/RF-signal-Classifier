# RF Signal Classifier

Lightweight ML pipeline for RF signal classification optimized for embedded satellite payloads.

## Results

| Metric | Value |
|--------|-------|
| Validation accuracy | **95.95%** |
| Model parameters | **12,262** |
| FP32 ONNX size | **49.8 KB** |
| INT8 quantized size | **23.7 KB** |
| Inference latency | **1.9 ms** |
| Throughput | **515 fps** |

Tested on 10k synthetic I/Q samples across 6 modulation types, 25 epochs training on CPU.

## Supported Modulations

| Class | Type | Description |
|-------|------|-------------|
| 0 | BPSK | Binary Phase Shift Keying |
| 1 | QPSK | Quadrature Phase Shift Keying |
| 2 | 8PSK | 8-Phase Shift Keying |
| 3 | 16QAM | 16-Quadrature Amplitude Modulation |
| 4 | GFSK | Gaussian Frequency Shift Keying |
| 5 | AM-DSB | Amplitude Modulation - Double Sideband |

## Project Structure

```
rf-signal-classifier/
├── src/
│   ├── preprocessing.py    # DSP pipeline, feature extraction
│   ├── model.py            # CNN / CNN-LSTM classifiers
│   ├── dataset.py          # Dataset loading + synthetic RF generation
│   ├── train.py            # Training loop with cosine scheduling
│   ├── export_onnx.py      # ONNX export + int8 quantization + benchmarking
│   └── inference.py        # Lightweight inference engine
├── configs/
│   └── default.yaml
├── tests/
│   └── test_model.py
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt

# Train
python -c "from src.dataset import generate_synthetic_signals, create_dataloaders; from src.model import create_model; from src.train import train; sigs, labels = generate_synthetic_signals(10000); train_loader, val_loader = create_dataloaders(sigs, labels); model = create_model('cnn'); train(model, train_loader, val_loader)"

# Export to ONNX + quantize
python -c "from src.model import create_model; from src.export_onnx import export_with_quantization; model = create_model('cnn'); export_with_quantization(model, 'models/rf_classifier.onnx')"
```

## Key Features

- **Depthwise separable convolutions** - 12k params vs typical 100k+ for same task
- **ONNX export + int8 quantization** - 23.7 KB model for embedded deployment
- **Synthetic RF data generator** - BPSK, QPSK, 8PSK, 16QAM, GFSK, AM-DSB with configurable SNR
- **Sliding window preprocessing** - handles continuous I/Q streams
- **Embedded inference engine** - runs on onnxruntime only, no PyTorch at inference

## Architecture

1D CNN with depthwise separable convolutions:
- Conv1d(2, 32, k=7) → BN → ReLU → MaxPool
- DepthwiseSeparable(32, 64, k=5) → MaxPool
- DepthwiseSeparable(64, 128, k=3) → AdaptiveAvgPool
- Dropout(0.3) → Linear(128, 6)

## Tests

```bash
pytest tests/ -v
```

## References

- Shi et al., "Deep Learning for RF Signal Classification in Unknown and Dynamic Spectrum Environments" (DySPAN 2019)
- O'Shea et al., "Convolutional Radio Modulation Recognition Networks" (2016)

## Author

Purshotam Kumar - [proust-19](https://github.com/proust-19)
