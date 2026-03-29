"""
Inference wrapper for the word-classification CNN.
Loads a saved checkpoint and classifies whole-word ROI images.
"""

import cv2
import torch
import numpy as np
from pathlib import Path

from word_cnn.model import WordCNN
from word_cnn.dataset import resize_pad, _autocrop_text


class WordPredictor:
    """
    Wraps WordCNN for inference.
    Loads a saved checkpoint and classifies whole-word ROI crops.
    """

    def __init__(self):
        self._model: WordCNN | None = None
        self._word_classes: list[str] = []
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.debug_dir: Path | None = None  # set to a Path to save debug images
        self._debug_counter: int = 0

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def word_classes(self) -> list[str]:
        return list(self._word_classes)

    def load_model(self, model_path: str) -> bool:
        """Load a trained word model from a .pth checkpoint."""
        try:
            checkpoint = torch.load(
                model_path, map_location=self._device, weights_only=False
            )
            self._word_classes = checkpoint.get("word_classes", [])
            num_classes = checkpoint.get("num_classes", len(self._word_classes))

            self._model = WordCNN(num_classes=num_classes).to(self._device)
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._model.eval()
            return True
        except Exception as e:
            print(f"Failed to load word model: {e}")
            self._model = None
            return False

    def predict_all(self, roi_image: np.ndarray) -> list[tuple[str, float]]:
        """Return every class sorted by confidence descending.

        Returns [(label, confidence), ...], empty list if no model loaded.
        """
        if self._model is None or not self._word_classes:
            return []

        if len(roi_image.shape) == 3:
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_image

        # Empty / noisy ROIs (rock texture, no label) have mid-gray pixels
        # everywhere (low contrast).  Real HUD labels are bright white text on
        # a near-black background → contrast ≥ ~150.  Skip classification for
        # anything that lacks a clear text-on-dark signal so we don't emit
        # random guesses for empty slots.
        if int(gray.max()) - int(gray.min()) < 100:
            return []

        cropped = _autocrop_text(gray)
        resized = resize_pad(gray, WordCNN.INPUT_W, WordCNN.INPUT_H)

        # Debug: save raw, cropped, and final canvas side by side
        if self.debug_dir is not None:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            self._debug_counter += 1
            ch, cw = cropped.shape
            nz = np.where(resized.max(axis=0) > 0)[0]
            tw = int(nz[-1]) + 1 if len(nz) else 0
            canvas_big = cv2.resize(resized, (WordCNN.INPUT_W * 4, WordCNN.INPUT_H * 4),
                                    interpolation=cv2.INTER_NEAREST)
            cv2.putText(canvas_big, f'raw={gray.shape[1]}x{gray.shape[0]} crop={cw}x{ch} tw={tw}px',
                        (2, 14), cv2.FONT_HERSHEY_PLAIN, 0.9, 255, 1)
            cv2.imwrite(str(self.debug_dir / f"live_{self._debug_counter:04d}.png"), canvas_big)

        tensor = torch.from_numpy(resized).float() / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(self._device)  # (1, 1, H, W)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        scores = [(self._word_classes[i], float(probs[i])) for i in range(len(self._word_classes))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def predict(self, roi_image: np.ndarray) -> tuple[str, float]:
        """Classify a whole-word ROI image. Returns (label, confidence)."""
        scores = self.predict_all(roi_image)
        return scores[0] if scores else ("?", 0.0)
