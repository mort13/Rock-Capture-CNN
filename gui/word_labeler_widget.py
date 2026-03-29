"""
Built-in word / resource-name labeling tool for Rock Capture CNN.
Shows whole ROI crops from template-mode or word_cnn-mode ROIs
and lets the user assign a class label.
"""

import os
import subprocess
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox,
    QMessageBox, QLineEdit, QComboBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap


class WordLabelerWidget(QGroupBox):
    """
    Built-in word labeling tool for collecting training data
    for the word-classification CNN.

    Workflow:
    1. Toggle "Enable Word Labeler"
    2. Pipeline sends raw ROI images for template/word_cnn ROIs
    3. One image shown at a time, enlarged
    4. User selects label from combo or types a new one, then presses Enter
    5. Image saved to data/word_training_data/{label}/img_{ts}.png
    6. Space = skip, Backspace = undo
    """

    def __init__(self, data_dir: str | Path = "data/word_training_data", parent=None):
        super().__init__("Word Labeler", parent)
        self._data_dir = Path(data_dir)
        self._queue: list[tuple[np.ndarray, str]] = []  # (roi_image, roi_name)
        self._active = False
        self._last_saved: tuple[Path, str] | None = None

        layout = QVBoxLayout(self)

        self.enable_cb = QCheckBox("Enable Word Labeler")
        self.enable_cb.stateChanged.connect(self._on_toggle)
        layout.addWidget(self.enable_cb)

        self.roi_preview = QLabel("--")
        self.roi_preview.setFixedSize(260, 70)
        self.roi_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.roi_preview.setStyleSheet(
            "QLabel { background: #333; border: 2px solid #888; font-size: 12px; color: #aaa; }"
        )
        layout.addWidget(self.roi_preview, alignment=Qt.AlignmentFlag.AlignCenter)

        self.source_label = QLabel("Source: --")
        layout.addWidget(self.source_label)

        self.queue_label = QLabel("Queue: 0")
        layout.addWidget(self.queue_label)

        # Label input: combo with known classes + editable for new ones
        label_row = QHBoxLayout()
        label_row.addWidget(QLabel("Label:"))
        self.label_combo = QComboBox()
        self.label_combo.setEditable(True)
        self.label_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.label_combo.lineEdit().setPlaceholderText("type or select label")
        self.label_combo.lineEdit().returnPressed.connect(self._on_label_enter)
        label_row.addWidget(self.label_combo, stretch=1)
        self.label_btn = QPushButton("Save")
        self.label_btn.clicked.connect(self._on_label_enter)
        label_row.addWidget(self.label_btn)
        layout.addLayout(label_row)

        skip_row = QHBoxLayout()
        skip_btn = QPushButton("Skip (Space)")
        skip_btn.clicked.connect(self._on_skip)
        undo_btn = QPushButton("Undo")
        undo_btn.clicked.connect(self._on_undo)
        skip_row.addWidget(skip_btn)
        skip_row.addWidget(undo_btn)
        layout.addLayout(skip_row)

        self.instructions = QLabel(
            "Enter = save label, Space = skip\n"
            "Backspace = undo last"
        )
        self.instructions.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.instructions)

        self.counts_label = QLabel("Samples: --")
        self.counts_label.setWordWrap(True)
        self.counts_label.setStyleSheet("font-size: 10px;")
        layout.addWidget(self.counts_label)

        dataset_row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh Counts")
        refresh_btn.clicked.connect(self.refresh_counts)
        open_btn = QPushButton("Open Data Folder")
        open_btn.clicked.connect(self._on_open_folder)
        clear_btn = QPushButton("Clear Dataset")
        clear_btn.clicked.connect(self._on_clear_dataset)
        dataset_row.addWidget(refresh_btn)
        dataset_row.addWidget(open_btn)
        dataset_row.addWidget(clear_btn)
        layout.addLayout(dataset_row)

        self.feedback_label = QLabel("")
        self.feedback_label.setStyleSheet(
            "font-size: 12px; font-weight: bold; padding: 4px; "
            "background-color: #444; color: #0f0; border-radius: 3px;"
        )
        self.feedback_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.feedback_label)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def is_active(self) -> bool:
        return self._active

    def set_data_dir(self, path: Path) -> None:
        self._data_dir = path

    def queue_image(self, roi_image: np.ndarray, roi_name: str) -> None:
        """Add a raw ROI image to the labeling queue."""
        if not self._active:
            return
        self._queue.append((roi_image.copy(), roi_name))
        self.queue_label.setText(f"Queue: {len(self._queue)}")
        if len(self._queue) == 1:
            self._show_next()

    def keyPressEvent(self, event):
        if not self._active or not self._queue:
            return
        if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            self._on_label_enter()
        elif event.key() == Qt.Key.Key_Space:
            self._on_skip()
        elif event.key() == Qt.Key.Key_Backspace:
            self._on_undo()

    def _on_label_enter(self) -> None:
        label = self.label_combo.currentText().strip().lower().replace(" ", "_")
        if not label or not self._queue:
            return
        self._save_current(label)
        self._queue.pop(0)
        # Add to combo if new
        if self.label_combo.findText(label) < 0:
            self.label_combo.addItem(label)
        self.feedback_label.setText(f"Saved as '{label}'")
        self.feedback_label.setStyleSheet(
            "font-size: 12px; font-weight: bold; padding: 4px; "
            "background-color: #444; color: #0f0; border-radius: 3px;"
        )
        self._show_next()

    def _on_skip(self) -> None:
        if self._queue:
            self._queue.pop(0)
            self.feedback_label.setText("Skipped")
            self.feedback_label.setStyleSheet(
                "font-size: 12px; font-weight: bold; padding: 4px; "
                "background-color: #444; color: #ff0; border-radius: 3px;"
            )
            self._show_next()

    def _on_undo(self) -> None:
        if self._last_saved:
            path, _ = self._last_saved
            if path.exists():
                path.unlink()
            self._last_saved = None
            self.feedback_label.setText("Undo last")
            self.feedback_label.setStyleSheet(
                "font-size: 12px; font-weight: bold; padding: 4px; "
                "background-color: #444; color: #f80; border-radius: 3px;"
            )

    def _save_current(self, label: str) -> None:
        if not self._queue:
            return
        roi_image, _ = self._queue[0]
        save_dir = self._data_dir / label
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"img_{timestamp}.png"
        save_path = save_dir / filename
        # Save as-is (full colour or grayscale)
        cv2.imwrite(str(save_path), roi_image)
        self._last_saved = (save_path, label)

    def _show_next(self) -> None:
        if not self._queue:
            self.roi_preview.clear()
            self.roi_preview.setText("Queue empty")
            self.source_label.setText("Source: --")
            self.queue_label.setText("Queue: 0")
            return
        roi_image, roi_name = self._queue[0]
        self.source_label.setText(f"Source: {roi_name}")
        self.queue_label.setText(f"Queue: {len(self._queue)}")

        # Display: resize to fit the preview label
        display = roi_image.copy()
        h, w = display.shape[:2]
        # Scale to fit 256×64
        scale = min(256 / max(w, 1), 64 / max(h, 1))
        new_w, new_h = int(w * scale), int(h * scale)
        display = cv2.resize(display, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        if len(display.shape) == 2:
            qimg = QImage(display.data, new_w, new_h, new_w, QImage.Format.Format_Grayscale8)
        else:
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, new_w, new_h, new_w * 3, QImage.Format.Format_RGB888)
        self.roi_preview.setPixmap(QPixmap.fromImage(qimg))

    def _on_toggle(self, state: int) -> None:
        self._active = bool(state)
        if not self._active:
            self._queue.clear()
            self.roi_preview.clear()
            self.roi_preview.setText("--")
            self.source_label.setText("Source: --")
            self.queue_label.setText("Queue: 0")

    def refresh_counts(self) -> None:
        if not self._data_dir.exists():
            self.counts_label.setText("No word training data directory")
            return
        counts = []
        for d in sorted(self._data_dir.iterdir()):
            if d.is_dir():
                n = len(list(d.glob("*.png")))
                if n > 0:
                    counts.append(f"'{d.name}': {n}")
        if counts:
            self.counts_label.setText("Samples: " + ", ".join(counts))
        else:
            self.counts_label.setText("No samples yet")
        # Sync combo with known classes
        known = {self.label_combo.itemText(i) for i in range(self.label_combo.count())}
        for d in sorted(self._data_dir.iterdir()):
            if d.is_dir() and d.name not in known:
                self.label_combo.addItem(d.name)

    def _on_open_folder(self) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["explorer", str(self._data_dir.resolve())])

    def _on_clear_dataset(self) -> None:
        if not self._data_dir.exists():
            QMessageBox.information(self, "Clear Dataset", "No word training data directory found.")
            return
        png_files = list(self._data_dir.rglob("*.png"))
        if not png_files:
            QMessageBox.information(self, "Clear Dataset", "No samples to delete.")
            return
        reply = QMessageBox.question(
            self, "Clear Word Dataset",
            f"Delete all {len(png_files)} word samples?\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        deleted = 0
        for f in png_files:
            try:
                f.unlink()
                deleted += 1
            except OSError:
                pass
        self._last_saved = None
        self.refresh_counts()
        QMessageBox.information(self, "Clear Dataset", f"Deleted {deleted} word samples.")
