"""
Image conversion utilities for Rock Capture CNN.
QPixmap <-> numpy array conversions and scaling helpers.
"""

import cv2
import numpy as np
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt


def qpixmap_to_numpy(pixmap: QPixmap) -> np.ndarray:
    """Convert QPixmap to BGR numpy array."""
    qimage = pixmap.toImage()
    width = qimage.width()
    height = qimage.height()
    ptr = qimage.bits()
    ptr.setsize(height * width * 4)
    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)


def cv_to_qpixmap(cv_image: np.ndarray) -> QPixmap:
    """Convert BGR or grayscale numpy array to QPixmap."""
    if len(cv_image.shape) == 2:
        h, w = cv_image.shape
        bytes_per_line = w
        q_image = QImage(
            cv_image.data, w, h, bytes_per_line,
            QImage.Format.Format_Grayscale8
        )
    else:
        h, w, c = cv_image.shape
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        bytes_per_line = c * w
        q_image = QImage(
            rgb.data, w, h, bytes_per_line,
            QImage.Format.Format_RGB888
        )
    return QPixmap.fromImage(q_image)


def scale_pixmap_to_label(pixmap: QPixmap, label_size) -> QPixmap:
    """Scale pixmap to fit label, maintaining aspect ratio."""
    return pixmap.scaled(
        label_size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def numpy_grayscale_to_qpixmap(gray_image: np.ndarray, scale: int = 1) -> QPixmap:
    """Convert a grayscale numpy array to QPixmap, optionally scaling up."""
    if scale > 1:
        h, w = gray_image.shape[:2]
        gray_image = cv2.resize(
            gray_image, (w * scale, h * scale),
            interpolation=cv2.INTER_NEAREST,
        )
    h, w = gray_image.shape[:2]
    q_image = QImage(gray_image.data, w, h, w, QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(q_image)
