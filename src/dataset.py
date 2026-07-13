"""Dataset handling and synthetic RF signal generation.

Supports real datasets (RadioML2016.10a) and synthetic signal
generation for training/testing.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from numpy.typing import NDArray


SIGNAL_TYPES = {
    0: "BPSK",
    1: "QPSK",
    2: "8PSK",
    3: "16QAM",
    4: "GFSK",
    5: "AM-DSB",
}


class RFDataset(Dataset):

    def __init__(
        self,
        signals: NDArray[np.complex64],
        labels: NDArray[np.int64],
        transform=None,
    ):
        self.signals = signals
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        iq = self.signals[idx]
        label = self.labels[idx]

        # Convert complex to 2-channel real (I, Q)
        x = np.stack([iq.real, iq.imag], axis=0).astype(np.float32)

        if self.transform:
            x = self.transform(x)

        return torch.from_numpy(x), torch.tensor(label, dtype=torch.long)


def generate_synthetic_signals(
    n_samples: int = 1000,
    snr_db_range: tuple = (0, 20),
    signal_length: int = 1024,
    seed: int = 42,
) -> tuple[NDArray[np.complex64], NDArray[np.int64]]:
    """Generate synthetic RF signals for each modulation type.

    Creates I/Q samples with realistic noise and modulation characteristics.
    """
    rng = np.random.default_rng(seed)
    signals = []
    labels = []

    samples_per_class = n_samples // len(SIGNAL_TYPES)

    for label, name in SIGNAL_TYPES.items():
        for _ in range(samples_per_class):
            snr_db = rng.uniform(*snr_db_range)
            sig = _generate_signal(name, signal_length, snr_db, rng)
            signals.append(sig)
            labels.append(label)

    return np.array(signals, dtype=np.complex64), np.array(labels, dtype=np.int64)


def _generate_signal(
    name: str,
    length: int,
    snr_db: float,
    rng: np.random.Generator,
) -> NDArray[np.complex64]:
    """Generate a single synthetic signal with given SNR."""
    t = np.arange(length) / 1e6  # 1 MHz sampling rate

    if name == "BPSK":
        bits = rng.choice([-1, 1], size=length)
        sig = bits * np.exp(1j * 2 * np.pi * 0.1 * t)
    elif name == "QPSK":
        symbols = np.array([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j]) / np.sqrt(2)
        bits = rng.integers(0, 4, size=length)
        sig = symbols[bits]
    elif name == "8PSK":
        angles = np.exp(1j * 2 * np.pi * np.arange(8) / 8)
        bits = rng.integers(0, 8, size=length)
        sig = angles[bits]
    elif name == "16QAM":
        real_parts = np.array([-3, -1, 1, 3]) / np.sqrt(10)
        imag_parts = np.array([-3, -1, 1, 3]) / np.sqrt(10)
        sig = (
            rng.choice(real_parts, size=length)
            + 1j * rng.choice(imag_parts, size=length)
        ).astype(np.complex64)
    elif name == "GFSK":
        bits = rng.choice([-1, 1], size=length)
        phase = np.cumsum(bits) * 0.3
        sig = np.exp(1j * phase)
    elif name == "AM-DSB":
        msg = rng.standard_normal(length)
        sig = (1 + 0.5 * msg) * np.exp(1j * 2 * np.pi * 0.1 * t)
    else:
        sig = rng.standard_normal(length) + 1j * rng.standard_normal(length)

    sig = sig.astype(np.complex64)

    # Add AWGN
    noise_power = 10 ** (-snr_db / 10)
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(length) + 1j * rng.standard_normal(length)
    )
    sig = sig + noise

    return sig / (np.max(np.abs(sig)) + 1e-8)


def create_dataloaders(
    signals: NDArray[np.complex64],
    labels: NDArray[np.int64],
    batch_size: int = 32,
    train_ratio: float = 0.8,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Split data and create train/val dataloaders."""
    n = len(labels)
    n_train = int(n * train_ratio)
    indices = np.random.permutation(n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_ds = RFDataset(signals[train_idx], labels[train_idx])
    val_ds = RFDataset(signals[val_idx], labels[val_idx])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
