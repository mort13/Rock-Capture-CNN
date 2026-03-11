"""
Per-ROI image filter pipeline for Rock Capture CNN.
Applies a chain of image filters defined by FilterSettings.
"""

import cv2
import numpy as np

from core.profile import FilterSettings


class ImageFilterPipeline:
    """Applies a chain of image filters defined by FilterSettings."""

    @staticmethod
    def apply(image: np.ndarray, settings: FilterSettings) -> np.ndarray:
        """
        Apply all filters in sequence:
        1. Brightness
        2. Contrast
        3. Channel isolation (if not "none")
        4. Grayscale conversion
        5. Threshold (if enabled)
        6. Invert
        """
        img = image.copy()
        img = ImageFilterPipeline._apply_brightness(img, settings.brightness)
        img = ImageFilterPipeline._apply_contrast(img, settings.contrast)

        if settings.channel != "none":
            img = ImageFilterPipeline._isolate_channel(img, settings.channel)

        if settings.grayscale:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if settings.threshold_enabled:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(
                gray, settings.threshold, 255, cv2.THRESH_BINARY
            )
            img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        if settings.invert:
            img = cv2.bitwise_not(img)

        return img

    @staticmethod
    def _apply_brightness(img: np.ndarray, value: int) -> np.ndarray:
        if value == 0:
            return img
        return cv2.convertScaleAbs(img, alpha=1, beta=value)

    @staticmethod
    def _apply_contrast(img: np.ndarray, value: int) -> np.ndarray:
        if value == 0:
            return img
        alpha = 1.0 + (value / 100.0)
        return cv2.convertScaleAbs(img, alpha=alpha, beta=0)

    @staticmethod
    def _isolate_channel(img: np.ndarray, channel: str) -> np.ndarray:
        """Zero out all channels except the named one."""
        result = img.copy()
        channel_map = {"blue": 0, "green": 1, "red": 2}
        keep = channel_map.get(channel)
        if keep is not None:
            for i in range(3):
                if i != keep:
                    result[:, :, i] = 0
        return result
