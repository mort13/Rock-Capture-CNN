"""
ONNX Runtime inference wrapper for the CRNN digit sequence model.

Drop-in replacement for CRNNPredictor — identical interface (is_loaded,
load_model, predict, predict_all) but uses onnxruntime instead of PyTorch.

Loads a .onnx file exported via crnn_export_onnx.py.

Usage:
    predictor = CRNNOnnxPredictor()
    predictor.load_model("crnn_model.onnx")
    text, conf = predictor.predict(roi_image, format_pattern="{1,3}.xx")
"""

import numpy as np
import cv2
from pathlib import Path

from digit_crnn.model import DigitCRNN
from digit_crnn.dataset import resize_pad, decode_ctc_greedy
from digit_crnn.predictor import _validate_format

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False


class CRNNOnnxPredictor:
    """
    ONNX Runtime inference wrapper for DigitCRNN.

    Matches the interface of CRNNPredictor exactly so it can be swapped in
    anywhere without other changes.
    """

    def __init__(self) -> None:
        self._session: "ort.InferenceSession | None" = None
        self._num_classes: int = DigitCRNN.NUM_CLASSES

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    def load_model(self, model_path: str | Path) -> bool:
        """Load a .onnx model file. Returns True on success."""
        if not _ORT_AVAILABLE:
            print("[CRNNOnnxPredictor] onnxruntime is not installed. "
                  "Run: pip install onnxruntime")
            return False
        try:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in ort.get_available_providers()
                else ["CPUExecutionProvider"]
            )
            self._session = ort.InferenceSession(str(model_path), providers=providers)
            print(f"[CRNNOnnxPredictor] Loaded: {model_path} "
                  f"(provider: {self._session.get_providers()[0]})")
            return True
        except Exception as e:
            print(f"[CRNNOnnxPredictor] Failed to load model: {e}")
            self._session = None
            return False

    # ── Preprocessing ──────────────────────────────────────────

    def _preprocess(self, image: np.ndarray) -> np.ndarray | None:
        """Return a (1, 1, 32, 256) float32 array ready for ONNX inference."""
        if image is None or image.size == 0:
            return None

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if int(gray.max()) - int(gray.min()) < 100:
            return None

        padded = resize_pad(gray, DigitCRNN.INPUT_W, DigitCRNN.INPUT_H)
        tensor = (padded.astype(np.float32) / 255.0)
        return tensor[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)

    # ── Inference ─────────────────────────────────────────────

    def predict(
        self,
        roi_image: np.ndarray,
        format_pattern: str = "",
    ) -> tuple[str, float]:
        """
        Decode a digit string from a ROI strip image.

        Returns (text, confidence) or ("", 0.0) on failure/validation reject.
        """
        if self._session is None:
            return "", 0.0

        processed = self._preprocess(roi_image)
        if processed is None:
            return "", 0.0

        # Run ONNX inference
        input_name = self._session.get_inputs()[0].name
        log_probs = self._session.run(None, {input_name: processed})[0]
        # log_probs shape: (T=64, 1, num_classes)

        probs = np.exp(log_probs[:, 0, :])           # (T, num_classes)
        best_indices = probs.argmax(axis=1)            # (T,)

        # Convert to torch-compatible decode — use int list
        import torch
        best_tensor = torch.from_numpy(best_indices.astype(np.int64))
        text = decode_ctc_greedy(best_tensor)

        # Confidence: mean max-prob over non-blank timesteps
        non_blank = best_indices != DigitCRNN.BLANK_IDX
        if not non_blank.any():
            return "", 0.0
        conf = float(probs[np.arange(len(probs)), best_indices][non_blank].mean())

        if not _validate_format(text, format_pattern):
            return "", 0.0

        text = text.replace('%', '')
        return text, conf

    def predict_all(
        self,
        roi_image: np.ndarray,
        format_pattern: str = "",
    ) -> list[tuple[str, float]]:
        text, conf = self.predict(roi_image, format_pattern)
        return [(text, conf)] if text else []
