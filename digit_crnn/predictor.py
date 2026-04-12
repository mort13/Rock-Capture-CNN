"""
Inference wrapper for the CRNN digit sequence model.

Accepts a raw ROI image (BGR or grayscale), applies preprocessing identical to
word_cnn.predictor (contrast check → autocrop → resize_pad), runs the CRNN,
and returns the decoded digit string via greedy CTC decoding.

Optional format_pattern validation rejects decoded strings that don't match
the expected format, returning ("", 0.0) rather than a silently wrong result.
"""

import re
import cv2
import torch
import numpy as np
from pathlib import Path

from digit_crnn.model import DigitCRNN
from digit_crnn.dataset import resize_pad, decode_ctc_greedy, IDX_TO_CHAR


# ── Format pattern → regex ────────────────────────────────────────────────────

_FORMAT_REGEXES: list[tuple[re.Pattern, re.Pattern]] = [
    # Decimal percent (MUST come before decimal-only and percent-only):
    # pattern contains '.' AND ends with '%'  e.g. {1,2}.{2}%
    (re.compile(r"\..*%\s*$"),
     re.compile(r"^\d{1,2}\.\d{2}%$")),
    # Decimal:  pattern contains '.' but does NOT end with '%'
    (re.compile(r"\.\s*(?:xx|\{[^}]+\}|x+)", re.IGNORECASE),
     re.compile(r"^\d{1,3}\.\d{2}$")),
    # Percent:  pattern ends with '%' (no '.' before it)
    (re.compile(r"^[^.]*%\s*$"),
     re.compile(r"^\d{1,2}%$")),
    # Integer:  pattern is only x/{} chars (no '.' or '%')
    (re.compile(r"^[\sx{},\d]+$"),
     re.compile(r"^\d{1,6}$")),
]


def _validate_format(text: str, format_pattern: str) -> bool:
    """
    Return True if *text* matches the regex implied by *format_pattern*.
    Returns True unconditionally when *format_pattern* is empty (no constraint).
    """
    if not format_pattern:
        return True
    for trigger, validator in _FORMAT_REGEXES:
        if trigger.search(format_pattern):
            return bool(validator.match(text))
    return True  # unknown pattern — allow through


# ── Predictor ─────────────────────────────────────────────────────────────────

class CRNNPredictor:
    """
    Wraps DigitCRNN for inference on digit-strip ROI images.

    Usage:
        predictor = CRNNPredictor()
        predictor.load_model("data/models/crnn_model.pth")
        text, conf = predictor.predict(roi_image, format_pattern="{1,3}.xx")
    """

    def __init__(self) -> None:
        self._model: DigitCRNN | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(self, model_path: str | Path) -> bool:
        """Load a trained CRNN checkpoint. Returns True on success."""
        try:
            checkpoint = torch.load(
                str(model_path), map_location=self._device, weights_only=False
            )
            num_classes = checkpoint.get("num_classes", DigitCRNN.NUM_CLASSES)
            self._model = DigitCRNN(num_classes=num_classes).to(self._device)
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._model.eval()
            return True
        except Exception as e:
            print(f"[CRNNPredictor] Failed to load model: {e}")
            self._model = None
            return False

    # ── Preprocessing ──────────────────────────────────────────

    def _preprocess(self, image: np.ndarray) -> np.ndarray | None:
        """
        Convert *image* to a 32×256 normalised grayscale strip ready for the model.

        Returns None when the image appears blank / noisy (low contrast).
        """
        if image is None or image.size == 0:
            return None

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Reject blank / rock-texture ROIs (no visible HUD text)
        if int(gray.max()) - int(gray.min()) < 100:
            return None

        return resize_pad(gray, DigitCRNN.INPUT_W, DigitCRNN.INPUT_H)

    # ── Inference ─────────────────────────────────────────────

    def predict(
        self,
        roi_image: np.ndarray,
        format_pattern: str = "",
    ) -> tuple[str, float]:
        """
        Decode a digit string from a ROI strip image.

        Args:
            roi_image:      BGR or grayscale numpy array from the pipeline.
            format_pattern: ROIDefinition.format_pattern — used for post-decode
                            validation.  Pass "" to skip validation.

        Returns:
            (text, confidence) where confidence is the mean max softmax probability
            over non-blank time-steps.  Returns ("", 0.0) when no model is loaded,
            the ROI is blank, or the decoded string fails format validation.
        """
        if self._model is None:
            return "", 0.0

        processed = self._preprocess(roi_image)
        if processed is None:
            return "", 0.0

        tensor = torch.from_numpy(processed).float() / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(self._device)  # (1, 1, H, W)

        with torch.no_grad():
            log_probs = self._model(tensor)  # (T, 1, C)
            probs = torch.exp(log_probs)     # (T, 1, C) — regular probabilities

        # Greedy decode
        best_indices = probs[:, 0, :].argmax(dim=1)  # (T,)
        text = decode_ctc_greedy(best_indices)

        # Confidence: mean of max-prob over non-blank time-steps
        max_probs = probs[:, 0, :].max(dim=1).values  # (T,)
        non_blank_mask = best_indices != DigitCRNN.BLANK_IDX
        if non_blank_mask.any():
            conf = float(max_probs[non_blank_mask].mean())
        else:
            return "", 0.0

        # Format validation
        if not _validate_format(text, format_pattern):
            return "", 0.0

        # Strip % character
        text = text.replace('%', '')
        return text, conf

    def predict_all(
        self,
        roi_image: np.ndarray,
        format_pattern: str = "",
    ) -> list[tuple[str, float]]:
        """
        Returns [(text, confidence)] — single-element list for interface parity
        with WordPredictor.predict_all.  Returns [] if no result available.
        """
        text, conf = self.predict(roi_image, format_pattern)
        return [(text, conf)] if text else []
