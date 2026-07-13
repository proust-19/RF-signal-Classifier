import numpy as np
from numpy.typing import NDArray

from .preprocessing import normalize_iq, extract_statistical_features


class EmbeddedInferenceEngine:

    # Designed for resource-constrained satellite payloads.

    def __init__(self, model_path: str, use_onnx: bool = True):
        self.use_onnx = use_onnx
        if use_onnx:
            import onnxruntime as ort

            self.session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name

    def predict(
        self,
        iq: NDArray[np.complex64],
        window_size: int = 1024,
    ) -> dict:
        
        # Preprocess
        x = self._preprocess(iq, window_size)

        if self.use_onnx:
            probs = self.session.run(None, {self.input_name: x})[0]
        else:
            raise NotImplementedError(
                "Pure-numpy inference requires manual weight loading"
            )

        probs = np.exp(probs) / np.sum(np.exp(probs), axis=1, keepdims=True)  # softmax
        pred_class = int(np.argmax(probs[0]))
        confidence = float(probs[0][pred_class])

        return {
            "class_id": pred_class,
            "confidence": round(confidence, 4),
            "probabilities": probs[0].tolist(),
        }

    def _preprocess(
        self,
        iq: NDArray[np.complex64],
        window_size: int,
    ) -> NDArray[np.float32]:
        """Preprocess I/Q data for model input."""
        # Pad or truncate to expected length
        if len(iq) < window_size:
            iq = np.pad(iq, (0, window_size - len(iq)))
        elif len(iq) > window_size:
            iq = iq[:window_size]

        # Convert to (1, 2, window_size) - batch, channels, samples
        x = np.stack([iq.real, iq.imag], axis=0).astype(np.float32)
        x = x / (np.max(np.abs(x)) + 1e-8)
        return x[np.newaxis, ...]

    def predict_batch(
        self,
        iq_batch: list[NDArray[np.complex64]],
        window_size: int = 1024,
    ) -> list[dict]:
        """Batch prediction for multiple signals."""
        return [self.predict(iq, window_size) for iq in iq_batch]
