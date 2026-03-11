"""
Screen capture engine for Rock Capture CNN.
Captures a screen region at configurable FPS using QTimer + QScreen.
"""

import numpy as np
from PyQt6.QtCore import QObject, QTimer, QRect, pyqtSignal
from PyQt6.QtWidgets import QApplication

from utils.image_utils import qpixmap_to_numpy


class CaptureEngine(QObject):
    """
    Captures a screen region at a target FPS.
    Emits frame_captured(np.ndarray) each tick with a BGR numpy array.
    """
    frame_captured = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._grab_frame)
        self._search_region: QRect | None = None
        self._screen = None
        self._fps: int = 10

    def set_search_region(self, region: QRect, monitor_index: int = 0) -> None:
        """Set the screen region to capture."""
        screens = QApplication.screens()
        if monitor_index < len(screens):
            self._screen = screens[monitor_index]
        self._search_region = region

    def start(self, fps: int = 10) -> None:
        """Start capturing at the given FPS."""
        self._fps = fps
        interval_ms = max(1, 1000 // fps)
        self._timer.start(interval_ms)

    def stop(self) -> None:
        """Stop capturing."""
        self._timer.stop()

    @property
    def is_running(self) -> bool:
        return self._timer.isActive()

    def grab_single_frame(self) -> np.ndarray | None:
        """Capture and return a single frame without emitting signal."""
        if self._search_region is None or self._screen is None:
            return None
        pixmap = self._screen.grabWindow(
            0,
            self._search_region.x(),
            self._search_region.y(),
            self._search_region.width(),
            self._search_region.height(),
        )
        return qpixmap_to_numpy(pixmap)

    def _grab_frame(self) -> None:
        """Internal: grab frame and emit signal."""
        frame = self.grab_single_frame()
        if frame is not None:
            self.frame_captured.emit(frame)
