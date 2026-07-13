"""Lightweight RF signal classifier models.

Provides both CNN and CNN-LSTM architectures optimized for
embedded deployment with configurable complexity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RFCNN(nn.Module):
    """1D CNN for RF signal classification.

    Optimized for embedded inference: uses depthwise separable
    convolutions, small kernel sizes, and batch normalization.
    """

    def __init__(
        self,
        num_classes: int = 6,
        in_channels: int = 2,
        base_filters: int = 32,
    ):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, base_filters, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            # Block 2 - depthwise separable
            DepthwiseSeparable1d(base_filters, base_filters * 2, kernel_size=5),
            nn.MaxPool1d(2),
            # Block 3
            DepthwiseSeparable1d(base_filters * 2, base_filters * 4, kernel_size=3),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(base_filters * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, channels, seq_len) - I/Q samples
        Returns:
            (batch, num_classes) logits
        """
        x = self.features(x)
        x = x.squeeze(-1)
        return self.classifier(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RFCNNLSTM(nn.Module):
    """CNN-LSTM hybrid for sequential RF signal classification.

    CNN extracts local features, LSTM captures temporal dependencies.
    """

    def __init__(
        self,
        num_classes: int = 6,
        in_channels: int = 2,
        cnn_filters: int = 32,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, cnn_filters, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(
                cnn_filters, cnn_filters * 2, kernel_size=5, padding=2, bias=False
            ),
            nn.BatchNorm1d(cnn_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_filters * 2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(lstm_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]
        return self.classifier(x)


class DepthwiseSeparable1d(nn.Module):
    """Depthwise separable 1D convolution for parameter efficiency."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_ch,
            in_ch,
            kernel_size,
            padding=kernel_size // 2,
            groups=in_ch,
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


def create_model(
    arch: str = "cnn",
    num_classes: int = 6,
    **kwargs,
) -> nn.Module:
    """Factory function for model creation."""
    models = {
        "cnn": RFCNN,
        "cnn_lstm": RFCNNLSTM,
    }
    if arch not in models:
        raise ValueError(
            f"Unknown architecture: {arch}. Choose from {list(models.keys())}"
        )
    return models[arch](num_classes=num_classes, **kwargs)
