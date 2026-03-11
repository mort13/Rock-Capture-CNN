"""
Fullscreen overlay widget for selecting a screen region.
"""

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRect, QPoint
from PyQt6.QtGui import QPainter, QPen, QColor


OVERLAY_ALPHA = 100
BORDER_WIDTH = 2
BORDER_COLOR = (0, 255, 0)


class ScreenCaptureOverlay(QWidget):
    """
    Transparent overlay for click-drag screen region selection.
    Shows a semi-transparent dark background with the selected area cut out.
    """

    def __init__(self, screen_geometry: QRect):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(screen_geometry)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self.start_point = QPoint()
        self.end_point = QPoint()
        self.is_selecting = False
        self.selected_rect: QRect | None = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_point = event.pos()
            self.end_point = event.pos()
            self.is_selecting = True
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_selecting:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_selecting = False
            self.end_point = event.pos()
            x = min(self.start_point.x(), self.end_point.x())
            y = min(self.start_point.y(), self.end_point.y())
            w = abs(self.end_point.x() - self.start_point.x())
            h = abs(self.end_point.y() - self.start_point.y())
            self.selected_rect = QRect(x, y, w, h)
            self.close()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, OVERLAY_ALPHA))

        if self.is_selecting or (self.start_point and self.end_point):
            x = min(self.start_point.x(), self.end_point.x())
            y = min(self.start_point.y(), self.end_point.y())
            w = abs(self.end_point.x() - self.start_point.x())
            h = abs(self.end_point.y() - self.start_point.y())

            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_Clear
            )
            painter.fillRect(x, y, w, h, Qt.GlobalColor.transparent)

            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_SourceOver
            )
            pen = QPen(QColor(*BORDER_COLOR), BORDER_WIDTH, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(x, y, w, h)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.selected_rect = None
            self.close()

    def get_selected_rect(self) -> QRect | None:
        if (
            self.selected_rect
            and self.selected_rect.width() > 0
            and self.selected_rect.height() > 0
        ):
            return self.selected_rect
        return None
