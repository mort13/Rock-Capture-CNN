"""
Built-in character labeling tool for Rock Capture CNN.
Keystroke-based labeling queue for segmented characters.
"""

import os
import subprocess
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox,
    QMessageBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent, QImage, QPixmap

from core.segmenter import SegmentedChar
from cnn.dataset import CharacterDataset


class LabelerWidget(QGroupBox):
    """
    Built-in character labeling tool.

    Workflow:
    1. User toggles "Enable Labeler Mode"
    2. Pipeline captures and segments characters from ROIs
    3. Segmented characters are queued here
    4. One character shown at a time, enlarged
    5. User presses key (0-9, ., -, %) to label
    6. Image saved to data/training_data/{class}/img_{timestamp}.png
    7. Space = skip, Backspace = undo last
    """

    def __init__(self, data_dir: str | Path = "data/training_data", parent=None):
        super().__init__("Labeler", parent)
        self._data_dir = Path(data_dir)
        self._queue: list[tuple[SegmentedChar, str]] = []
        self._active = False
        self._last_saved: tuple[Path, str] | None = None
        self._paused = False

        layout = QVBoxLayout(self)

        self.enable_cb = QCheckBox("Enable Labeler Mode")
        self.enable_cb.stateChanged.connect(self._on_toggle)
        layout.addWidget(self.enable_cb)

        self.pause_cb = QCheckBox("Pause queue (label existing)")
        self.pause_cb.setEnabled(False)
        layout.addWidget(self.pause_cb)

        self.char_preview = QLabel("--")
        self.char_preview.setFixedSize(140, 140)
        self.char_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.char_preview.setStyleSheet(
            "QLabel { background: #333; border: 2px solid #888; font-size: 12px; color: #aaa; }"
        )
        layout.addWidget(self.char_preview, alignment=Qt.AlignmentFlag.AlignCenter)

        self.source_label = QLabel("Source: --")
        layout.addWidget(self.source_label)

        self.queue_label = QLabel("Queue: 0")
        layout.addWidget(self.queue_label)

        self.instructions = QLabel(
            "Press 0-9 to label digit\n"
            "'.' for dot, '-' for dash\n"
            "ß or ` for percent (%)\n"
            "Space = skip, Backspace = undo last"
        )
        self.instructions.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.instructions)

        self.counts_label = QLabel("Samples: --")
        self.counts_label.setWordWrap(True)
        self.counts_label.setStyleSheet("font-size: 10px;")
        layout.addWidget(self.counts_label)

        refresh_btn = QPushButton("Refresh Counts")
        refresh_btn.clicked.connect(self.refresh_counts)
        layout.addWidget(refresh_btn)

        dataset_row = QHBoxLayout()
        open_folder_btn = QPushButton("Open Data Folder")
        open_folder_btn.setToolTip("Open the training data directory in Explorer")
        open_folder_btn.clicked.connect(self._on_open_folder)
        clear_btn = QPushButton("Clear Dataset")
        clear_btn.setToolTip("Delete all labeled samples (requires confirmation)")
        clear_btn.clicked.connect(self._on_clear_dataset)
        dataset_row.addWidget(open_folder_btn)
        dataset_row.addWidget(clear_btn)
        layout.addLayout(dataset_row)

        # Feedback label for last action
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

    def queue_characters(self, chars: list[SegmentedChar], roi_name: str) -> None:
        """Add segmented characters to the labeling queue."""
        if not self._active or self._paused:
            return
        for c in chars:
            self._queue.append((c, roi_name))
        self.queue_label.setText(f"Queue: {len(self._queue)}")
        if len(self._queue) == len(chars):
            self._show_next()

    # Keys that are remapped to a label character.
    # ß -> % so user doesn't need Shift+5
    _KEY_REMAP: dict[str, str] = {"ß": "%", "`": "%"}

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if not self._active or not self._queue:
            return

        key = event.text()
        label = self._KEY_REMAP.get(key, key)
        valid_chars = "0123456789.-%"

        if label in valid_chars:
            self._save_current(label)
            self._queue.pop(0)
            self.feedback_label.setText(f"Labeled as '{label}'")
            self.feedback_label.setStyleSheet(
                "font-size: 12px; font-weight: bold; padding: 4px; "
                "background-color: #444; color: #0f0; border-radius: 3px;"
            )
            self._show_next()
        elif event.key() == Qt.Key.Key_Space:
            self._queue.pop(0)
            self.feedback_label.setText("Skipped")
            self.feedback_label.setStyleSheet(
                "font-size: 12px; font-weight: bold; padding: 4px; "
                "background-color: #444; color: #ff0; border-radius: 3px;"
            )
            self._show_next()
        elif event.key() == Qt.Key.Key_Backspace:
            self._undo_last()
            self.feedback_label.setText("Undo last")
            self.feedback_label.setStyleSheet(
                "font-size: 12px; font-weight: bold; padding: 4px; "
                "background-color: #444; color: #f80; border-radius: 3px;"
            )

    def _save_current(self, label: str) -> None:
        if not self._queue:
            return
        char, roi_name = self._queue[0]

        dir_name = CharacterDataset.CHAR_DIR_MAP.get(label, label)
        save_dir = self._data_dir / dir_name
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"img_{timestamp}.png"
        save_path = save_dir / filename

        img_to_save = (char.image * 255).astype(np.uint8)
        cv2.imwrite(str(save_path), img_to_save)

        self._last_saved = (save_path, label)

    def _undo_last(self) -> None:
        if self._last_saved:
            path, _ = self._last_saved
            if path.exists():
                path.unlink()
            self._last_saved = None

    def _show_next(self) -> None:
        if not self._queue:
            self.char_preview.clear()
            self.char_preview.setText("Queue empty")
            self.source_label.setText("Source: --")
            self.queue_label.setText("Queue: 0")
            return

        char, roi_name = self._queue[0]
        self.source_label.setText(f"Source: {roi_name}")
        self.queue_label.setText(f"Queue: {len(self._queue)}")

        display_img = (char.image * 255).astype(np.uint8)
        display_img = cv2.resize(
            display_img, (128, 128), interpolation=cv2.INTER_NEAREST
        )
        h, w = display_img.shape[:2]
        q_image = QImage(display_img.data, w, h, w, QImage.Format.Format_Grayscale8)
        self.char_preview.setPixmap(QPixmap.fromImage(q_image))

    def _on_toggle(self, state: int) -> None:
        self._active = bool(state)
        self.pause_cb.setEnabled(self._active)
        if not self._active:
            self._queue.clear()
            self.char_preview.clear()
            self.char_preview.setText("--")
            self.source_label.setText("Source: --")
            self.queue_label.setText("Queue: 0")

    def refresh_counts(self) -> None:
        """Scan training data directory and show per-class counts."""
        if not self._data_dir.exists():
            self.counts_label.setText("No training data directory")
            return

        counts = []
        all_dirs = {**CharacterDataset.CHAR_DIR_MAP}
        for i in range(10):
            all_dirs[str(i)] = str(i)

        for char, dir_name in sorted(all_dirs.items()):
            class_dir = self._data_dir / dir_name
            if class_dir.exists():
                n = len(list(class_dir.glob("*.png")))
                if n > 0:
                    counts.append(f"'{char}': {n}")

        if counts:
            self.counts_label.setText("Samples: " + ", ".join(counts))
        else:
            self.counts_label.setText("No samples yet")

    def _on_open_folder(self) -> None:
        """Open the training data directory in Windows Explorer."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["explorer", str(self._data_dir.resolve())])

    def _on_clear_dataset(self) -> None:
        """Delete all .png files from all class subdirectories after confirmation."""
        if not self._data_dir.exists():
            QMessageBox.information(self, "Clear Dataset", "No training data directory found.")
            return

        png_files = list(self._data_dir.rglob("*.png"))
        if not png_files:
            QMessageBox.information(self, "Clear Dataset", "No samples to delete.")
            return

        reply = QMessageBox.question(
            self,
            "Clear Dataset",
            f"Delete all {len(png_files)} labeled samples?\nThis cannot be undone.",
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
        QMessageBox.information(self, "Clear Dataset", f"Deleted {deleted} samples.")
