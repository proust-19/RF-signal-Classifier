import numpy as np
from numpy.typing import NDArray


def normalize_iq(iq: NDArray[np.complex64]) -> NDArray[np.float32]:
    """Normalize I/Q data to zero-mean, unit-variance."""
    real = iq.real.astype(np.float32)
    imag = iq.imag.astype(np.float32)
    mag = np.sqrt(real**2 + imag**2)
    phase = np.arctan2(imag, real)
    return np.stack([real / (mag + 1e-8), imag / (mag + 1e-8), phase], axis=-1)


def compute_spectrogram(
    iq: NDArray[np.complex64],
    nperseg: int = 256,
    noverlap: int = 128,
) -> NDArray[np.float32]:
    """Compute spectrogram features from I/Q data.

    Returns log-magnitude spectrogram suitable for CNN input.
    """
    from scipy.signal import stft

    f, t, Zxx = stft(
        iq.astype(np.complex128),
        fs=1.0,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    mag = np.abs(Zxx).astype(np.float32)
    log_mag = np.log1p(mag)
    return log_mag


def extract_statistical_features(iq: NDArray[np.complex64]) -> NDArray[np.float32]:
    real = iq.real.astype(np.float32)
    imag = iq.imag.astype(np.float32)

    features = np.array(
        [
            np.mean(real),
            np.mean(imag),
            np.std(real),
            np.std(imag),
            np.mean(np.abs(iq)),
            np.std(np.abs(iq)),
            _zero_crossing_rate(real),
            _spectral_centroid(real),
        ],
        dtype=np.float32,
    )

    return features


def _zero_crossing_rate(signal: NDArray[np.float32]) -> float:
    """Compute zero-crossing rate."""
    signs = np.sign(signal)
    return float(np.sum(np.abs(np.diff(signs))) / (2 * len(signal)))


def _spectral_centroid(signal: NDArray[np.float32]) -> float:
    """Compute spectral centroid."""
    fft = np.fft.rfft(signal)
    magnitudes = np.abs(fft)
    freqs = np.arange(len(magnitudes))
    if np.sum(magnitudes) == 0:
        return 0.0
    return float(np.sum(freqs * magnitudes) / np.sum(magnitudes))


def sliding_window(
    iq: NDArray[np.complex64],
    window_size: int = 1024,
    hop_size: int = 512,
) -> NDArray[np.complex64]:
    """Split I/Q data into overlapping windows."""
    n_windows = (len(iq) - window_size) // hop_size + 1
    windows = np.stack(
        [iq[i * hop_size : i * hop_size + window_size] for i in range(n_windows)]
    )
    return windows
