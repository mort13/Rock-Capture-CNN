"""
Dialog for selecting the anchor ROI on a captured frame.
Shows the current search region frame and lets the user draw a rectangle.
"""

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QDialogButtonBox,
)
from PyQt6.QtCore import Qt, QPoint, QRect
from PyQt6.QtGui import QPainter, QPen, QColor, QPixmap, QImage, QMouseEvent

from utils.image_utils import cv_to_qpixmap


class FrameROISelector(QLabel):
    """A QLabel that displays a frame image and lets the user draw a rectangle on it."""

    def __init__(self, frame: np.ndarray, parent=None):
        super().__init__(parent)
        self._frame = frame
        self._display_scale = 1.0
        self._start = QPoint()
        self._end = QPoint()
        self._selecting = False
        self.selected_rect: QRect | None = None

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._update_display()

    def _update_display(self) -> None:
        pixmap = cv_to_qpixmap(self._frame)
        # Scale down if too large for screen
        max_w, max_h = 900, 700
        if pixmap.width() > max_w or pixmap.height() > max_h:
            scaled = pixmap.scaled(
                max_w, max_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._display_scale = pixmap.width() / scaled.width()
            self.setPixmap(scaled)
        else:
            self._display_scale = 1.0
            self.setPixmap(pixmap)
        self.setFixedSize(self.pixmap().size())

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._start = event.pos()
            self._end = event.pos()
            self._selecting = True
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._selecting:
            self._end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._selecting:
            self._selecting = False
            self._end = event.pos()

            # Convert display coords to frame coords
            s = self._display_scale
            x1 = int(min(self._start.x(), self._end.x()) * s)
            y1 = int(min(self._start.y(), self._end.y()) * s)
            x2 = int(max(self._start.x(), self._end.x()) * s)
            y2 = int(max(self._start.y(), self._end.y()) * s)

            # Clamp to frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(self._frame.shape[1], x2)
            y2 = min(self._frame.shape[0], y2)

            w = x2 - x1
            h = y2 - y1
            if w > 5 and h > 5:
                self.selected_rect = QRect(x1, y1, w, h)
            self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if not self._selecting and self.selected_rect is None:
            return

        painter = QPainter(self)
        pen = QPen(QColor(255, 0, 255), 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)

        if self._selecting:
            x = min(self._start.x(), self._end.x())
            y = min(self._start.y(), self._end.y())
            w = abs(self._end.x() - self._start.x())
            h = abs(self._end.y() - self._start.y())
            painter.drawRect(x, y, w, h)
        elif self.selected_rect:
            s = self._display_scale
            painter.drawRect(
                int(self.selected_rect.x() / s),
                int(self.selected_rect.y() / s),
                int(self.selected_rect.width() / s),
                int(self.selected_rect.height() / s),
            )
        painter.end()


class AnchorROIDialog(QDialog):
    """
    Dialog that shows a captured frame and lets the user draw a rectangle
    to define the anchor search ROI.
    """

    def __init__(self, frame: np.ndarray, current_roi: dict | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Anchor ROI")
        self.setModal(True)

        layout = QVBoxLayout(self)

        info = QLabel(
            "Draw a rectangle around the area where the anchor template appears.\n"
            "This limits the template matching search area for better performance."
        )
        info.setStyleSheet("color: #aaa;")
        layout.addWidget(info)

        self.selector = FrameROISelector(frame)
        if current_roi and all(k in current_roi for k in ("x", "y", "w", "h")):
            self.selector.selected_rect = QRect(
                current_roi["x"], current_roi["y"],
                current_roi["w"], current_roi["h"],
            )
        layout.addWidget(self.selector)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_roi_dict(self) -> dict | None:
        """Return the selected ROI as {"x", "y", "w", "h"} or None."""
        r = self.selector.selected_rect
        if r and r.width() > 5 and r.height() > 5:
            return {"x": r.x(), "y": r.y(), "w": r.width(), "h": r.height()}
        return None
