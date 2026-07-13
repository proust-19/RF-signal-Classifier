# RF Signal Classifier

Lightweight ML pipeline for RF signal classification optimized for embedded satellite payloads.

## Overview

Classifies RF modulation types (BPSK, QPSK, 8PSK, 16QAM, GFSK, AM-DSB) from I/Q samples using compact CNN and CNN-LSTM architectures. Includes ONNX export with int8 quantization for edge deployment.

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
│   └── default.yaml        # Training & export configuration
├── tests/
│   └── test_model.py       # Unit tests
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt

# Train
python -c "from src.dataset import generate_synthetic_signals, create_dataloaders; from src.model import create_model; from src.train import train; sigs, labels = generate_synthetic_signals(10000); train_loader, val_loader = create_dataloaders(sigs, labels); model = create_model('cnn'); train(model, train_loader, val_loader)"

# Export to ONNX
python -c "from src.model import create_model; from src.export_onnx import export_with_quantization; model = create_model('cnn'); export_with_quantization(model, 'models/rf_classifier.onnx')"
```

## Supported Modulations

| Class | Type |
|-------|------|
| 0 | BPSK |
| 1 | QPSK |
| 2 | 8PSK |
| 3 | 16QAM |
| 4 | GFSK |
| 5 | AM-DSB |

## Key Features

- **Depthwise separable convolutions** for parameter-efficient inference
- **ONNX export** with int8 dynamic quantization (~4x compression)
- **Synthetic data generation** with configurable SNR
- **Sliding window** preprocessing for continuous I/Q streams
- **Embedded inference engine** using only onnxruntime

## Tests

```bash
pytest tests/ -v
```

## Author

Purshotam Kumar - [proust-19](https://github.com/proust-19)
