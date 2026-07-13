"""Tests for model architecture and preprocessing."""

import pytest
import torch
import numpy as np

from src.model import RFCNN, RFCNNLSTM, create_model
from src.preprocessing import (
    normalize_iq,
    compute_spectrogram,
    extract_statistical_features,
    sliding_window,
)


# ── Model Tests ──────────────────────────────────────────────────────────────


class TestRFCNN:
    def test_forward_pass(self):
        model = RFCNN(num_classes=6, in_channels=2)
        x = torch.randn(4, 2, 1024)
        out = model(x)
        assert out.shape == (4, 6)

    def test_parameter_count(self):
        model = RFCNN(num_classes=6)
        params = model.count_parameters()
        assert params > 0
        assert params < 500_000  # lightweight for embedded

    def test_single_sample(self):
        model = RFCNN(num_classes=6)
        x = torch.randn(1, 2, 1024)
        out = model(x)
        assert out.shape == (1, 6)


class TestRFCNNLSTM:
    def test_forward_pass(self):
        model = RFCNNLSTM(num_classes=6, in_channels=2)
        x = torch.randn(4, 2, 1024)
        out = model(x)
        assert out.shape == (4, 6)


class TestCreateModel:
    def test_cnn_factory(self):
        model = create_model("cnn", num_classes=6)
        assert isinstance(model, RFCNN)

    def test_cnn_lstm_factory(self):
        model = create_model("cnn_lstm", num_classes=6)
        assert isinstance(model, RFCNNLSTM)

    def test_invalid_arch(self):
        with pytest.raises(ValueError):
            create_model("invalid", num_classes=6)


# ── Preprocessing Tests ──────────────────────────────────────────────────────


class TestPreprocessing:
    def test_normalize_iq(self):
        iq = np.random.randn(1024) + 1j * np.random.randn(1024)
        iq = iq.astype(np.complex64)
        result = normalize_iq(iq)
        assert result.dtype == np.float32
        assert result.shape == (1024, 3)

    def test_compute_spectrogram(self):
        iq = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64)
        spec = compute_spectrogram(iq, nperseg=128, noverlap=64)
        assert spec.ndim == 2
        assert spec.dtype == np.float32

    def test_statistical_features(self):
        iq = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64)
        feat = extract_statistical_features(iq)
        assert feat.shape == (8,)
        assert feat.dtype == np.float32

    def test_sliding_window(self):
        iq = (np.random.randn(4096) + 1j * np.random.randn(4096)).astype(np.complex64)
        windows = sliding_window(iq, window_size=1024, hop_size=512)
        assert windows.ndim == 2
        assert windows.shape[1] == 1024
