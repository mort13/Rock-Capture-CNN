"""
Anchor template matching for Rock Capture CNN.
Finds a template image (anchor) within a captured frame using cv2.matchTemplate.
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class AnchorResult:
    """Result of anchor matching."""
    found: bool
    x: int = 0
    y: int = 0
    confidence: float = 0.0
    anchor_w: int = 0
    anchor_h: int = 0


class AnchorMatcher:
    """
    Finds a template image (anchor) within a captured frame
    using cv2.matchTemplate with TM_CCOEFF_NORMED.
    """

    def __init__(self):
        self._template: np.ndarray | None = None
        self._template_gray: np.ndarray | None = None
        self._threshold: float = 0.7

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = max(0.0, min(1.0, value))

    @property
    def is_loaded(self) -> bool:
        return self._template is not None

    @property
    def template_size(self) -> tuple[int, int]:
        """(width, height) of loaded template."""
        if self._template is None:
            return (0, 0)
        h, w = self._template.shape[:2]
        return (w, h)

    def load_template(self, path: str | Path) -> bool:
        """Load anchor template image from file. Returns True on success."""
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return False
        self._template = img
        self._template_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return True

    def find_anchor(self, frame: np.ndarray, anchor_roi: dict | None = None) -> AnchorResult:
        """
        Search for the anchor template in the given frame.

        Args:
            frame: BGR numpy array (the search region capture)
            anchor_roi: optional sub-region {"x", "y", "w", "h"} within
                        the frame to restrict template matching. Coordinates
                        returned are in the full frame's coordinate space.

        Returns:
            AnchorResult with position and confidence.
        """
        if self._template_gray is None:
            return AnchorResult(found=False)

        # Optionally crop to anchor ROI for faster matching
        roi_offset_x = 0
        roi_offset_y = 0
        search_area = frame
        if anchor_roi and all(k in anchor_roi for k in ("x", "y", "w", "h")):
            rx, ry = anchor_roi["x"], anchor_roi["y"]
            rw, rh = anchor_roi["w"], anchor_roi["h"]
            # Clamp to frame bounds
            rx = max(0, rx)
            ry = max(0, ry)
            rx2 = min(frame.shape[1], rx + rw)
            ry2 = min(frame.shape[0], ry + rh)
            search_area = frame[ry:ry2, rx:rx2]
            roi_offset_x = rx
            roi_offset_y = ry

        if search_area.size == 0:
            return AnchorResult(found=False)

        search_gray = cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY)
        th, tw = self._template_gray.shape[:2]

        if search_gray.shape[0] < th or search_gray.shape[1] < tw:
            return AnchorResult(found=False)

        result = cv2.matchTemplate(
            search_gray, self._template_gray, cv2.TM_CCOEFF_NORMED
        )
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= self._threshold:
            return AnchorResult(
                found=True,
                x=max_loc[0] + roi_offset_x,
                y=max_loc[1] + roi_offset_y,
                confidence=max_val,
                anchor_w=tw,
                anchor_h=th,
            )
        return AnchorResult(found=False, confidence=max_val)
