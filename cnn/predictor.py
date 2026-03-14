"""
CNN inference wrapper for Rock Capture CNN.
Loads a saved checkpoint and classifies segmented characters.
"""

import torch
import numpy as np

from cnn.model import DigitCNN
from core.segmenter import SegmentedChar


class Predictor:
    """
    Wraps DigitCNN for inference.
    Loads a saved checkpoint and classifies segmented characters.
    """

    def __init__(self):
        self._model: DigitCNN | None = None
        self._char_classes: str = ""
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def char_classes(self) -> str:
        return self._char_classes

    def load_model(self, model_path: str, char_classes: str = "") -> bool:
        """Load a trained model from a .pth checkpoint file."""
        try:
            checkpoint = torch.load(
                model_path, map_location=self._device, weights_only=False
            )
            self._char_classes = checkpoint.get("char_classes", char_classes)
            num_classes = checkpoint.get("num_classes", len(self._char_classes))

            self._model = DigitCNN(num_classes=num_classes).to(self._device)
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._model.eval()
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            self._model = None
            return False

    def predict_single(self, char_image: np.ndarray) -> tuple[str, float]:
        """
        Classify a single 28x28 normalized character image.

        Args:
            char_image: float32 array of shape (28, 28), values in [0, 1]

        Returns:
            (predicted_char, confidence)
        """
        if self._model is None:
            return ("?", 0.0)

        tensor = torch.from_numpy(char_image).float().unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, idx = torch.max(probs, dim=1)

        predicted_char = self._char_classes[idx.item()]
        return (predicted_char, confidence.item())

    def predict_sequence(
        self, characters: list[SegmentedChar], allowed_chars: str = ""
    ) -> list[tuple[str, float]]:
        """Classify a sequence of segmented characters (batch inference).

        Args:
            characters: segmented character images
            allowed_chars: if non-empty, only classes whose character is in this
                           string are considered — all others are masked to -inf
                           before argmax.  Empty = allow all model classes.

        Returns:
            List of (predicted_char, confidence) tuples.
        """
        if not characters or self._model is None:
            return []

        batch = torch.stack(
            [torch.from_numpy(c.image).float().unsqueeze(0) for c in characters]
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(batch)

            if allowed_chars:
                mask = torch.full(logits.shape, float("-inf"), device=self._device)
                for idx, ch in enumerate(self._char_classes):
                    if ch in allowed_chars:
                        mask[:, idx] = 0.0
                logits = logits + mask

            probs = torch.softmax(logits, dim=1)
            confidences, indices = torch.max(probs, dim=1)

        return [
            (self._char_classes[i.item()], c.item())
            for i, c in zip(indices, confidences)
        ]
