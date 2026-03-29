"""
Preview widget for Rock Capture CNN.
Large annotated frame preview + horizontal strip of per-ROI detail cards.
"""

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QScrollArea, QSplitter,
)
from PyQt6.QtCore import Qt

from utils.image_utils import cv_to_qpixmap, scale_pixmap_to_label, numpy_grayscale_to_qpixmap


class ROICard(QGroupBox):
    """Small card showing one ROI's details: raw, filtered, segmented, recognized text."""

    def __init__(self, name: str, parent=None):
        super().__init__(name, parent)
        self.setFixedWidth(220)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        layout.addWidget(QLabel("Raw:"))
        self.raw_label = QLabel()
        self.raw_label.setFixedHeight(40)
        self.raw_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.raw_label.setStyleSheet("background: #222;")
        layout.addWidget(self.raw_label)

        layout.addWidget(QLabel("Filtered:"))
        self.filtered_label = QLabel()
        self.filtered_label.setFixedHeight(40)
        self.filtered_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.filtered_label.setStyleSheet("background: #222;")
        layout.addWidget(self.filtered_label)

        self.seg_header = QLabel("Segmented:")
        layout.addWidget(self.seg_header)
        self.seg_label = QLabel()
        self.seg_label.setFixedHeight(40)
        self.seg_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.seg_label.setStyleSheet("background: #222;")
        self.seg_label.setScaledContents(True)
        layout.addWidget(self.seg_label)

        self.text_label = QLabel("--")
        self.text_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #0f0; padding: 2px;"
        )
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label.setToolTip("")
        layout.addWidget(self.text_label)

    @staticmethod
    def _draw_segmentation(filtered_image: np.ndarray, characters: list) -> np.ndarray:
        """Overlay cyan vertical cut-lines at each character boundary on the filtered image."""
        if len(filtered_image.shape) == 2:
            vis = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)
        else:
            vis = filtered_image.copy()
        h = vis.shape[0]
        for char in characters:
            x, _y, w, _h = char.bbox
            cv2.line(vis, (x, 0), (x, h - 1), (0, 255, 255), 1)
            cv2.line(vis, (x + w, 0), (x + w, h - 1), (0, 255, 255), 1)
        return vis

    @staticmethod
    def _draw_scores(word_scores: list, width: int = 212, height: int = 40) -> np.ndarray:
        """Render top-5 scores as a horizontal bar chart (BGR numpy image)."""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        top = word_scores[:5]
        if not top:
            return img
        bar_h = max(1, (height - 2) // len(top))
        for i, (lbl, conf) in enumerate(top):
            y = i * bar_h
            bar_w = max(1, round(conf * (width - 2)))
            color = (0, 200, 0) if i == 0 else (0, 120, 120)
            cv2.rectangle(img, (0, y + 1), (bar_w, y + bar_h - 1), color, -1)
            text = f"{lbl}  {conf*100:.0f}%"
            cv2.putText(img, text, (3, y + bar_h - 3),
                        cv2.FONT_HERSHEY_PLAIN, 0.7, (220, 220, 220), 1, cv2.LINE_AA)
        return img

    def update_content(self, raw_image: np.ndarray, filtered_image: np.ndarray, characters: list, text: str, word_scores: list | None = None, recognition_mode: str = "") -> None:
        raw_pix = cv_to_qpixmap(raw_image)
        scaled_raw = scale_pixmap_to_label(raw_pix, self.raw_label.size())
        self.raw_label.setPixmap(scaled_raw)

        filt_pix = cv_to_qpixmap(filtered_image)
        scaled_filt = scale_pixmap_to_label(filt_pix, self.filtered_label.size())
        self.filtered_label.setPixmap(scaled_filt)

        seg_img = self._draw_segmentation(filtered_image, characters)
        seg_pix = cv_to_qpixmap(seg_img)
        scaled_seg = scale_pixmap_to_label(seg_pix, self.seg_label.size())
        self.seg_label.setPixmap(scaled_seg)

        self.text_label.setText(text if text else "--")

        # show recognition mode tag in group box title
        base_title = self.title().split(" [")[0]
        if recognition_mode:
            tag = {"word_cnn": "[W-CNN]", "template": "[tmpl]", "cnn": "[CNN]"}.get(recognition_mode, f"[{recognition_mode}]")
            self.setTitle(f"{base_title} {tag}")

        if word_scores:
            tip = "\n".join(f"{lbl}: {c*100:.1f}%" for lbl, c in word_scores)
            self.text_label.setToolTip(tip)
            self.seg_header.setText("Scores:")
            chart = self._draw_scores(word_scores, width=212, height=40)
            chart_pix = cv_to_qpixmap(chart)
            self.seg_label.setPixmap(chart_pix)
        else:
            self.text_label.setToolTip("")
            self.seg_header.setText("Segmented:")
            seg_img = self._draw_segmentation(filtered_image, characters)
            seg_pix = cv_to_qpixmap(seg_img)
            self.seg_label.setPixmap(scale_pixmap_to_label(seg_pix, self.seg_label.size()))

class PreviewWidget(QWidget):
    """
    Left side of the main window.
    Contains a large annotated frame preview and a horizontal ROI detail strip.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Vertical)
        outer.addWidget(splitter)

        self.main_preview = QLabel("No capture")
        self.main_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_preview.setMinimumSize(640, 400)
        self.main_preview.setStyleSheet(
            "QLabel { background-color: #1e1e1e; border: 1px solid #444; color: #888; }"
        )
        splitter.addWidget(self.main_preview)

        roi_scroll = QScrollArea()
        roi_scroll.setWidgetResizable(True)
        roi_scroll.setStyleSheet("QScrollArea { border: 1px solid #444; }")
        self.roi_strip = QWidget()
        self.roi_strip_layout = QHBoxLayout(self.roi_strip)
        self.roi_strip_layout.setContentsMargins(4, 4, 4, 4)
        self.roi_strip_layout.addStretch()
        roi_scroll.setWidget(self.roi_strip)
        splitter.addWidget(roi_scroll)

        splitter.setSizes([600, 220])

        self._roi_cards: dict[str, ROICard] = {}

    def update_main_preview(self, annotated_frame: np.ndarray) -> None:
        """Display the annotated frame (BGR numpy array)."""
        pixmap = cv_to_qpixmap(annotated_frame)
        scaled = scale_pixmap_to_label(pixmap, self.main_preview.size())
        self.main_preview.setPixmap(scaled)

    def update_roi_previews(self, roi_results: list, profile_name: str = "") -> None:
        """Update per-ROI detail cards, qualified by profile name."""
        for result in roi_results:
            key = f"{profile_name}/{result.name}" if profile_name else result.name
            if key not in self._roi_cards:
                display = f"{profile_name} / {result.name}" if profile_name else result.name
                card = ROICard(display)
                self._roi_cards[key] = card
                idx = self.roi_strip_layout.count() - 1
                self.roi_strip_layout.insertWidget(idx, card)
            self._roi_cards[key].update_content(
                result.raw_image, result.filtered_image, result.characters,
                result.recognized_text,
                getattr(result, "word_scores", None) or None,
                getattr(result, "recognition_mode", ""),
            )

    def remove_card(self, name: str) -> None:
        """Remove a single ROI card by name."""
        card = self._roi_cards.pop(name, None)
        if card is not None:
            self.roi_strip_layout.removeWidget(card)
            card.deleteLater()

    def clear_previews(self) -> None:
        """Remove all ROI cards."""
        for card in self._roi_cards.values():
            self.roi_strip_layout.removeWidget(card)
            card.deleteLater()
        self._roi_cards.clear()
        self.main_preview.clear()
        self.main_preview.setText("No capture")
