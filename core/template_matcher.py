"""
Template-based word recognition for Rock Capture CNN.

Loads a directory of labelled template images and matches an ROI crop
against them using normalised cross-correlation (TM_CCOEFF_NORMED).
The filename (minus extension) is the label returned on a match.
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class TemplateMatch:
    """Result of a template-matching recognition."""
    label: str
    confidence: float


class WordTemplateMatcher:
    """
    Match a cropped ROI image against a set of named templates.

    Usage:
        matcher = WordTemplateMatcher()
        matcher.load_templates("data/templates/resources")
        result = matcher.match(filtered_roi)
        print(result.label, result.confidence)
    """

    def __init__(self, threshold: float = 0.5):
        self._templates: list[tuple[str, np.ndarray]] = []  # (label, gray image)
        self._threshold = threshold

    @property
    def is_loaded(self) -> bool:
        return len(self._templates) > 0

    def load_templates(self, directory: str | Path) -> int:
        """
        Load all .png / .jpg / .bmp images from *directory*.
        The stem of each filename becomes the label.
        Returns the number of templates loaded.
        """
        self._templates.clear()
        d = Path(directory)
        if not d.is_dir():
            return 0
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.PNG", "*.JPG", "*.BMP"):
            for img_path in sorted(d.glob(ext)):
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self._templates.append((img_path.stem, gray))
        return len(self._templates)

    def match(self, roi_image: np.ndarray) -> TemplateMatch:
        """
        Match *roi_image* against all loaded templates.

        The ROI is resized to each template's dimensions before matching
        so that minor size differences don't hurt correlation scores.

        Returns the best-matching label and confidence.
        If no template exceeds the threshold, label is "none" and confidence 0.
        """
        if not self._templates:
            return TemplateMatch(label="none", confidence=0.0)

        if len(roi_image.shape) == 3:
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_image

        best_label = "none"
        best_score = -1.0

        for label, tmpl in self._templates:
            th, tw = tmpl.shape[:2]
            rh, rw = gray.shape[:2]
            # Scale template down if it's larger than the ROI in any dimension
            if tw > rw or th > rh:
                scale = min(rw / tw, rh / th)
                tmpl_scaled = cv2.resize(tmpl, (max(1, int(tw * scale)), max(1, int(th * scale))), interpolation=cv2.INTER_AREA)
            else:
                tmpl_scaled = tmpl
            try:
                result = cv2.matchTemplate(gray, tmpl_scaled, cv2.TM_CCOEFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(result)
            except cv2.error:
                continue
            if score > best_score:
                best_score = score
                best_label = label

        if best_score >= self._threshold:
            return TemplateMatch(label=best_label, confidence=best_score)
        return TemplateMatch(label="none", confidence=best_score)
