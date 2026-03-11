"""
ROI editor widget for Rock Capture CNN.
List of ROI definitions with add/edit/remove functionality.
"""

from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidget, QListWidgetItem, QDialog, QFormLayout, QLineEdit, QSpinBox,
    QDialogButtonBox, QComboBox, QLabel,
)
from PyQt6.QtCore import pyqtSignal, Qt

from core.profile import ROIDefinition


_SEG_MODES = [
    ("Projection (recommended)", "projection"),
    ("Contour", "contour"),
    ("Fixed width", "fixed_width"),
]
_MODE_LABELS = [label for label, _ in _SEG_MODES]
_MODE_VALUES = [value for _, value in _SEG_MODES]


class ROIEditDialog(QDialog):
    """Dialog for adding or editing a single ROI definition."""

    def __init__(self, parent=None, roi: ROIDefinition | None = None):
        super().__init__(parent)
        self.setWindowTitle("Edit ROI" if roi else "Add ROI")
        layout = QFormLayout(self)

        self.name_edit = QLineEdit(roi.name if roi else "")

        self.x_spin = QSpinBox()
        self.x_spin.setRange(-4000, 4000)
        self.y_spin = QSpinBox()
        self.y_spin.setRange(-4000, 4000)
        self.w_spin = QSpinBox()
        self.w_spin.setRange(1, 4000)
        self.w_spin.setValue(80)
        self.h_spin = QSpinBox()
        self.h_spin.setRange(1, 4000)
        self.h_spin.setValue(24)

        if roi:
            self.x_spin.setValue(roi.x_offset)
            self.y_spin.setValue(roi.y_offset)
            self.w_spin.setValue(roi.width)
            self.h_spin.setValue(roi.height)

        layout.addRow("Name:", self.name_edit)
        layout.addRow("X Offset:", self.x_spin)
        layout.addRow("Y Offset:", self.y_spin)
        layout.addRow("Width:", self.w_spin)
        layout.addRow("Height:", self.h_spin)

        # Segmentation mode
        self.seg_combo = QComboBox()
        self.seg_combo.addItems(_MODE_LABELS)
        if roi and roi.seg_mode in _MODE_VALUES:
            self.seg_combo.setCurrentIndex(_MODE_VALUES.index(roi.seg_mode))
        self.seg_combo.currentIndexChanged.connect(self._on_seg_mode_changed)
        layout.addRow("Segmentation:", self.seg_combo)

        # Characters (fixed_width primary) — exact char count, no pixel rounding
        self.char_count_label = QLabel("Characters:")
        self.char_count_spin = QSpinBox()
        self.char_count_spin.setRange(0, 30)
        self.char_count_spin.setValue(roi.char_count if roi else 0)
        self.char_count_spin.setSpecialValueText("Auto")
        self.char_count_spin.setToolTip(
            "How many characters are in this ROI.\n"
            "When set, the ROI is divided into exactly this many equal slices\n"
            "— no pixel rounding issues.\n"
            "0 = determine from Char width below.\n"
            "Only used in Fixed width mode."
        )
        layout.addRow(self.char_count_label, self.char_count_spin)

        # Char width (fixed_width fallback when char_count == 0)
        self.char_w_label = QLabel("Char width (px):")
        self.char_w_spin = QSpinBox()
        self.char_w_spin.setRange(0, 500)
        self.char_w_spin.setValue(roi.char_width if roi else 0)
        self.char_w_spin.setSpecialValueText("Auto")
        self.char_w_spin.setToolTip(
            "Pixel width of one character.\n"
            "Used when Characters is set to Auto (0).\n"
            "0 = estimate from ROI height.\n"
            "Only used in Fixed width mode."
        )
        layout.addRow(self.char_w_label, self.char_w_spin)

        # Allowed characters for this ROI (filters CNN predictions)
        self.allowed_chars_edit = QLineEdit(roi.allowed_chars if roi else "")
        self.allowed_chars_edit.setPlaceholderText("empty = all model classes")
        self.allowed_chars_edit.setToolTip(
            "Only characters listed here will be predicted for this ROI.\n"
            "Example: '0123456789' for a digits-only field.\n"
            "Example: '0123456789%' for a percentage field.\n"
            "Example: '0123456789.' for a decimal field.\n"
            "Leave empty to allow all classes the model was trained on."
        )
        layout.addRow("Allowed chars:", self.allowed_chars_edit)

        # Format pattern
        self.format_pattern_edit = QLineEdit(roi.format_pattern if roi else "")
        self.format_pattern_edit.setPlaceholderText("e.g.  xx%  or  xxx.xx  (empty = off)")
        self.format_pattern_edit.setToolTip(
            "Describes the exact output format for this ROI.\n"
            "'x' = one character predicted by the CNN.\n"
            "Any other character is inserted literally (no CNN needed).\n\n"
            "Examples:\n"
            "  xx%    → 2 predicted digits then a literal %\n"
            "  xxx.xx → 3 digits, literal dot, 2 digits\n"
            "  xxxxx  → 5 predicted digits\n\n"
            "The number of 'x' positions automatically determines\n"
            "how many character crops the segmenter produces,\n"
            "overriding the 'Characters' count above."
        )
        layout.addRow("Format pattern:", self.format_pattern_edit)

        # Dot width — pixel width of the '.' glyph in the image for format patterns
        self.dot_width_spin = QSpinBox()
        self.dot_width_spin.setRange(0, 100)
        self.dot_width_spin.setValue(roi.dot_width if roi else 0)
        self.dot_width_spin.setSpecialValueText("Auto (char_width ÷ 4)")
        self.dot_width_spin.setToolTip(
            "Pixel width of the decimal-point glyph in the captured image.\n"
            "Only used when the format pattern contains '.' (e.g. '{1,3}.xx').\n"
            "The dot glyph is skipped over during segmentation and '.' is\n"
            "inserted in the output without any CNN prediction.\n"
            "0 = auto-estimate (char_width ÷ 4)."
        )
        layout.addRow("Dot width (px):", self.dot_width_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        self._on_seg_mode_changed(self.seg_combo.currentIndex())

    def _on_seg_mode_changed(self, index: int) -> None:
        is_fixed = _MODE_VALUES[index] == "fixed_width"
        for w in (self.char_count_label, self.char_count_spin,
                  self.char_w_label, self.char_w_spin):
            w.setEnabled(is_fixed)

    def get_roi(self) -> ROIDefinition:
        return ROIDefinition(
            name=self.name_edit.text() or "unnamed",
            x_offset=self.x_spin.value(),
            y_offset=self.y_spin.value(),
            width=self.w_spin.value(),
            height=self.h_spin.value(),
            seg_mode=_MODE_VALUES[self.seg_combo.currentIndex()],
            char_count=self.char_count_spin.value(),
            char_width=self.char_w_spin.value(),
            allowed_chars=self.allowed_chars_edit.text(),
            format_pattern=self.format_pattern_edit.text(),
            dot_width=self.dot_width_spin.value(),
        )


class ROIEditor(QGroupBox):
    """
    ROI list management widget.

    Signals:
        roi_selected(int): index of selected ROI
        rois_changed(): emitted when ROIs are added/removed/edited
    """
    roi_selected = pyqtSignal(int)
    rois_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("ROI Definitions", parent)
        layout = QVBoxLayout(self)

        self.roi_list = QListWidget()
        self.roi_list.currentRowChanged.connect(self._on_selection_changed)
        self.roi_list.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.roi_list)

        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("+")
        self.add_btn.setFixedWidth(40)
        self.add_btn.setToolTip("Add ROI")
        self.edit_btn = QPushButton("Edit")
        self.edit_btn.setToolTip("Edit selected ROI")
        self.remove_btn = QPushButton("-")
        self.remove_btn.setFixedWidth(40)
        self.remove_btn.setToolTip("Remove selected ROI")
        self.add_btn.clicked.connect(self._on_add)
        self.edit_btn.clicked.connect(self._on_edit)
        self.remove_btn.clicked.connect(self._on_remove)
        btn_row.addWidget(self.add_btn)
        btn_row.addWidget(self.edit_btn)
        btn_row.addWidget(self.remove_btn)
        layout.addLayout(btn_row)

        self._rois: list[ROIDefinition] = []
        self._refreshing = False

    @property
    def rois(self) -> list[ROIDefinition]:
        return list(self._rois)

    def load_rois(self, rois: list[ROIDefinition]) -> None:
        self._rois = list(rois)
        self._refresh_list()

    def _refresh_list(self) -> None:
        self._refreshing = True
        self.roi_list.clear()
        for roi in self._rois:
            pattern_hint = f"  {roi.format_pattern}" if roi.format_pattern else ""
            item = QListWidgetItem(
                f"{roi.name}  ({roi.x_offset}, {roi.y_offset}) "
                f"{roi.width}x{roi.height}  [{roi.seg_mode}]{pattern_hint}"
            )
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if roi.enabled else Qt.CheckState.Unchecked
            )
            self.roi_list.addItem(item)
        self._refreshing = False

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        if self._refreshing:
            return
        row = self.roi_list.row(item)
        if 0 <= row < len(self._rois):
            self._rois[row].enabled = (item.checkState() == Qt.CheckState.Checked)
            self.rois_changed.emit()

    def _on_add(self) -> None:
        dialog = ROIEditDialog(self)
        if dialog.exec():
            roi = dialog.get_roi()
            self._rois.append(roi)
            self._refresh_list()
            self.rois_changed.emit()

    def _on_edit(self) -> None:
        idx = self.roi_list.currentRow()
        if idx < 0 or idx >= len(self._rois):
            return
        dialog = ROIEditDialog(self, self._rois[idx])
        if dialog.exec():
            new_roi = dialog.get_roi()
            new_roi.filters = self._rois[idx].filters  # preserve filter settings
            new_roi.enabled = self._rois[idx].enabled  # preserve enabled state (list checkbox)
            self._rois[idx] = new_roi
            self._refresh_list()
            self.roi_list.setCurrentRow(idx)
            self.rois_changed.emit()

    def _on_remove(self) -> None:
        idx = self.roi_list.currentRow()
        if idx < 0 or idx >= len(self._rois):
            return
        self._rois.pop(idx)
        self._refresh_list()
        self.rois_changed.emit()

    def _on_selection_changed(self, index: int) -> None:
        if 0 <= index < len(self._rois):
            self.roi_selected.emit(index)

    def get_selected_roi(self) -> ROIDefinition | None:
        idx = self.roi_list.currentRow()
        if 0 <= idx < len(self._rois):
            return self._rois[idx]
        return None

    def update_selected_filters(self, filters) -> None:
        idx = self.roi_list.currentRow()
        if 0 <= idx < len(self._rois):
            self._rois[idx].filters = filters
