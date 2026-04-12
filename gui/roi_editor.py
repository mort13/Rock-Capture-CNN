"""
ROI editor widget for Rock Capture CNN.
List of ROI definitions with add/edit/remove functionality.
List of sub-anchor definitions with search region editing.
"""

from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidget, QListWidgetItem, QDialog, QFormLayout, QLineEdit, QSpinBox,
    QDialogButtonBox, QComboBox, QLabel, QFileDialog,
)
from PyQt6.QtCore import pyqtSignal, Qt
from pathlib import Path

from core.profile import ROIDefinition, AnchorPoint


_SEG_MODES = [
    ("Projection (recommended)", "projection"),
    ("Contour", "contour"),
    ("Fixed width", "fixed_width"),
]
_MODE_LABELS = [label for label, _ in _SEG_MODES]
_MODE_VALUES = [value for _, value in _SEG_MODES]


_RECOG_MODES = [
    ("CNN (digits/chars)", "cnn"),
    ("Template matching (words)", "template"),
    ("Word CNN (names)", "word_cnn"),
]
_RECOG_LABELS = [label for label, _ in _RECOG_MODES]
_RECOG_VALUES = [value for _, value in _RECOG_MODES]


class ROIEditDialog(QDialog):
    """Dialog for adding or editing a single ROI definition."""

    def __init__(self, parent=None, roi: ROIDefinition | None = None,
                 multi_anchor: bool = False,
                 sub_anchor_names: list[str] | None = None):
        super().__init__(parent)
        self.setWindowTitle("Edit ROI" if roi else "Add ROI")
        self._multi_anchor = multi_anchor
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

        self.csv_index_spin = QSpinBox()
        self.csv_index_spin.setRange(0, 999)
        self.csv_index_spin.setValue(roi.csv_index if roi else 0)
        self.csv_index_spin.setSpecialValueText("0 (unordered)")
        self.csv_index_spin.setToolTip(
            "Column order in the exported CSV.\n"
            "Columns are sorted ascending by this value.\n"
            "0 = placed after all explicitly-ordered columns."
        )
        layout.addRow("CSV column index:", self.csv_index_spin)

        # --- Multi-anchor reference position ---
        self.ref_x_spin = QSpinBox()
        self.ref_x_spin.setRange(-10000, 10000)
        self.ref_x_spin.setValue(int(roi.ref_x) if roi else 0)
        self.ref_x_spin.setToolTip(
            "Absolute X position in the reference frame.\n"
            "Used with multi-anchor resolution-independent mode."
        )
        self.ref_y_spin = QSpinBox()
        self.ref_y_spin.setRange(-10000, 10000)
        self.ref_y_spin.setValue(int(roi.ref_y) if roi else 0)
        self.ref_y_spin.setToolTip(
            "Absolute Y position in the reference frame.\n"
            "Used with multi-anchor resolution-independent mode."
        )
        self._ref_x_label = QLabel("Ref X:")
        self._ref_y_label = QLabel("Ref Y:")
        layout.addRow(self._ref_x_label, self.ref_x_spin)
        layout.addRow(self._ref_y_label, self.ref_y_spin)

        self.sub_anchor_combo = QComboBox()
        self.sub_anchor_combo.addItem("(none)")
        if sub_anchor_names:
            for sa_name in sub_anchor_names:
                self.sub_anchor_combo.addItem(sa_name)
        if roi and roi.sub_anchor:
            idx = self.sub_anchor_combo.findText(roi.sub_anchor)
            if idx >= 0:
                self.sub_anchor_combo.setCurrentIndex(idx)
        self._sub_anchor_label = QLabel("Sub-anchor:")
        layout.addRow(self._sub_anchor_label, self.sub_anchor_combo)

        # Show/hide multi-anchor vs legacy fields
        for w in (self._ref_x_label, self.ref_x_spin,
                  self._ref_y_label, self.ref_y_spin,
                  self._sub_anchor_label, self.sub_anchor_combo):
            w.setVisible(multi_anchor)

        # --- Legacy position fields ---
        self._x_label = QLabel("X Offset:")
        self._y_label = QLabel("Y Offset:")
        layout.addRow(self._x_label, self.x_spin)
        layout.addRow(self._y_label, self.y_spin)
        self._x_label.setVisible(not multi_anchor)
        self.x_spin.setVisible(not multi_anchor)
        self._y_label.setVisible(not multi_anchor)
        self.y_spin.setVisible(not multi_anchor)

        layout.addRow("Width:", self.w_spin)
        layout.addRow("Height:", self.h_spin)

        # Recognition mode
        self.recog_combo = QComboBox()
        self.recog_combo.addItems(_RECOG_LABELS)
        if roi and roi.recognition_mode in _RECOG_VALUES:
            self.recog_combo.setCurrentIndex(_RECOG_VALUES.index(roi.recognition_mode))
        self.recog_combo.currentIndexChanged.connect(self._on_recog_mode_changed)
        layout.addRow("Recognition:", self.recog_combo)

        # Template directory (only for template mode)
        self.template_dir_label = QLabel("Template dir:")
        self.template_dir_edit = QLineEdit(roi.template_dir if roi else "")
        self.template_dir_edit.setPlaceholderText("relative to data/  e.g. templates/resources")
        self.template_dir_edit.setToolTip(
            "Folder containing template images for word matching.\n"
            "Path is relative to the project's data/ directory.\n"
            "Each .png/.jpg file in that folder is one possible word;\n"
            "the filename (without extension) is the label returned."
        )
        layout.addRow(self.template_dir_label, self.template_dir_edit)

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

        self._on_recog_mode_changed(self.recog_combo.currentIndex())

    def _on_recog_mode_changed(self, index: int) -> None:
        is_template = _RECOG_VALUES[index] == "template"
        # Show template dir only in template mode
        self.template_dir_label.setVisible(is_template)
        self.template_dir_edit.setVisible(is_template)
        # Hide CNN-specific widgets in template mode
        cnn_widgets = [
            self.seg_combo,
            self.char_count_label, self.char_count_spin,
            self.char_w_label, self.char_w_spin,
            self.allowed_chars_edit,
            self.format_pattern_edit,
            self.dot_width_spin,
        ]
        for w in cnn_widgets:
            w.setVisible(not is_template)
        # Also hide the form labels for hidden widgets
        form = self.layout()
        for w in cnn_widgets:
            label = form.labelForField(w)
            if label:
                label.setVisible(not is_template)
        if not is_template:
            self._on_seg_mode_changed(self.seg_combo.currentIndex())

    def _on_seg_mode_changed(self, index: int) -> None:
        is_fixed = _MODE_VALUES[index] == "fixed_width"
        for w in (self.char_count_label, self.char_count_spin,
                  self.char_w_label, self.char_w_spin):
            w.setEnabled(is_fixed)

    def get_roi(self) -> ROIDefinition:
        sub_anchor = ""
        if self._multi_anchor and self.sub_anchor_combo.currentIndex() > 0:
            sub_anchor = self.sub_anchor_combo.currentText()
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
            recognition_mode=_RECOG_VALUES[self.recog_combo.currentIndex()],
            template_dir=self.template_dir_edit.text(),
            csv_index=self.csv_index_spin.value(),
            ref_x=float(self.ref_x_spin.value()),
            ref_y=float(self.ref_y_spin.value()),
            sub_anchor=sub_anchor,
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
        self._multi_anchor: bool = False
        self._sub_anchor_names: list[str] = []

    @property
    def multi_anchor(self) -> bool:
        return self._multi_anchor

    @multi_anchor.setter
    def multi_anchor(self, value: bool) -> None:
        self._multi_anchor = value

    @property
    def sub_anchor_names(self) -> list[str]:
        return self._sub_anchor_names

    @sub_anchor_names.setter
    def sub_anchor_names(self, names: list[str]) -> None:
        self._sub_anchor_names = list(names)

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
            idx_hint = f"#{roi.csv_index}  " if roi.csv_index > 0 else "#-  "
            if self._multi_anchor and (roi.ref_x or roi.ref_y):
                pos_hint = f"ref({roi.ref_x:.0f}, {roi.ref_y:.0f})"
                sa_hint = f" →{roi.sub_anchor}" if roi.sub_anchor else ""
            else:
                pos_hint = f"({roi.x_offset}, {roi.y_offset})"
                sa_hint = ""
            item = QListWidgetItem(
                f"{idx_hint}{roi.name}  {pos_hint} "
                f"{roi.width}x{roi.height}  [{roi.seg_mode}]{pattern_hint}{sa_hint}"
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
        dialog = ROIEditDialog(
            self, multi_anchor=self._multi_anchor,
            sub_anchor_names=self._sub_anchor_names,
        )
        if dialog.exec():
            roi = dialog.get_roi()
            self._rois.append(roi)
            self._refresh_list()
            self.rois_changed.emit()

    def _on_edit(self) -> None:
        idx = self.roi_list.currentRow()
        if idx < 0 or idx >= len(self._rois):
            return
        dialog = ROIEditDialog(
            self, self._rois[idx],
            multi_anchor=self._multi_anchor,
            sub_anchor_names=self._sub_anchor_names,
        )
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


# ============================================================================
# Sub-Anchor Editor
# ============================================================================

class SubAnchorEditDialog(QDialog):
    """Dialog for editing a sub-anchor's search region and template image."""

    def __init__(self, parent=None, anchor: AnchorPoint | None = None, anchors_dir: Path | None = None):
        super().__init__(parent)
        self.setWindowTitle("Edit Sub-Anchor" if anchor else "Add Sub-Anchor")
        self._anchors_dir = anchors_dir or Path("data/anchors")
        layout = QFormLayout(self)

        self.name_edit = QLineEdit(anchor.name if anchor else "")
        self.name_edit.setPlaceholderText("e.g. percent_sign")
        layout.addRow("Name:", self.name_edit)

        # Template file selection
        template_layout = QHBoxLayout()
        self.template_path_edit = QLineEdit(anchor.template_path if anchor else "")
        self.template_path_edit.setPlaceholderText("e.g. percent_sign.png (relative to data/anchors)")
        self.template_path_edit.setReadOnly(True)
        self.template_path_edit.setToolTip(
            "Path to the sub-anchor template image (relative to data/anchors/).\n"
            "Should be a PNG or JPG file."
        )
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.setFixedWidth(80)
        self.browse_btn.clicked.connect(self._on_browse_template)
        template_layout.addWidget(self.template_path_edit)
        template_layout.addWidget(self.browse_btn)
        layout.addRow("Template:", template_layout)

        self.ref_x_spin = QSpinBox()
        self.ref_x_spin.setRange(-10000, 10000)
        self.ref_x_spin.setValue(int(anchor.ref_x) if anchor else 0)
        self.ref_x_spin.setToolTip(
            "Expected X position of the sub-anchor in the reference frame."
        )
        layout.addRow("Ref X:", self.ref_x_spin)

        self.ref_y_spin = QSpinBox()
        self.ref_y_spin.setRange(-10000, 10000)
        self.ref_y_spin.setValue(int(anchor.ref_y) if anchor else 0)
        self.ref_y_spin.setToolTip(
            "Expected Y position of the sub-anchor in the reference frame."
        )
        layout.addRow("Ref Y:", self.ref_y_spin)

        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 100)
        self.threshold_spin.setValue(int((anchor.match_threshold if anchor else 0.7) * 100))
        self.threshold_spin.setSuffix("%")
        self.threshold_spin.setToolTip(
            "Template matching confidence threshold (0-100%).\n"
            "Higher = stricter matching."
        )
        layout.addRow("Match Threshold:", self.threshold_spin)

        layout.addWidget(QLabel("Search Region (optional)"))
        layout.addWidget(QLabel("Leave at 0 to use default padding"))

        self.search_x_spin = QSpinBox()
        self.search_x_spin.setRange(-10000, 10000)
        if anchor and anchor.search_region:
            self.search_x_spin.setValue(int(anchor.search_region.get("x", 0)))
        layout.addRow("Search X:", self.search_x_spin)

        self.search_y_spin = QSpinBox()
        self.search_y_spin.setRange(-10000, 10000)
        if anchor and anchor.search_region:
            self.search_y_spin.setValue(int(anchor.search_region.get("y", 0)))
        layout.addRow("Search Y:", self.search_y_spin)

        self.search_w_spin = QSpinBox()
        self.search_w_spin.setRange(0, 10000)
        if anchor and anchor.search_region:
            self.search_w_spin.setValue(int(anchor.search_region.get("width", 0)))
        layout.addRow("Search Width:", self.search_w_spin)

        self.search_h_spin = QSpinBox()
        self.search_h_spin.setRange(0, 10000)
        if anchor and anchor.search_region:
            self.search_h_spin.setValue(int(anchor.search_region.get("height", 0)))
        layout.addRow("Search Height:", self.search_h_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _on_browse_template(self) -> None:
        """Open file dialog to select a template image."""
        start_dir = str(self._anchors_dir)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Sub-Anchor Template",
            start_dir,
            "Image Files (*.png *.jpg *.jpeg);;All Files (*)",
        )
        if path:
            try:
                path_obj = Path(path)
                anchors_dir_abs = self._anchors_dir.resolve()
                rel_path = path_obj.relative_to(anchors_dir_abs)
                self.template_path_edit.setText(str(rel_path))
            except ValueError:
                # File is not under anchors_dir, show full path warning
                self.template_path_edit.setText(path)

    def get_anchor(self) -> AnchorPoint:
        """Return the edited sub-anchor."""
        search_region = None
        if (self.search_w_spin.value() > 0 or self.search_h_spin.value() > 0):
            search_region = {
                "x": self.search_x_spin.value(),
                "y": self.search_y_spin.value(),
                "width": self.search_w_spin.value(),
                "height": self.search_h_spin.value(),
            }
        
        return AnchorPoint(
            name=self.name_edit.text() or "unnamed",
            template_path=self.template_path_edit.text(),
            match_threshold=self.threshold_spin.value() / 100.0,
            ref_x=float(self.ref_x_spin.value()),
            ref_y=float(self.ref_y_spin.value()),
            search_region=search_region,
        )


class SubAnchorEditor(QGroupBox):
    """
    Sub-anchor list management widget for multi-anchor mode.

    Signals:
        sub_anchors_changed(): emitted when sub-anchors are added/removed/edited
    """
    sub_anchors_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Sub-Anchors", parent)
        layout = QVBoxLayout(self)

        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("+")
        self.add_btn.setFixedWidth(40)
        self.add_btn.setToolTip("Add sub-anchor")
        self.add_btn.clicked.connect(self._on_add)
        
        self.edit_btn = QPushButton("Edit")
        self.edit_btn.setToolTip("Edit selected sub-anchor")
        self.edit_btn.clicked.connect(self._on_edit)
        
        self.remove_btn = QPushButton("-")
        self.remove_btn.setFixedWidth(40)
        self.remove_btn.setToolTip("Remove selected sub-anchor")
        self.remove_btn.clicked.connect(self._on_remove)

        btn_row.addWidget(self.add_btn)
        btn_row.addWidget(self.edit_btn)
        btn_row.addWidget(self.remove_btn)
        layout.addLayout(btn_row)

        self._sub_anchors: list[AnchorPoint] = []
        self._anchors_dir: Path = Path("data/anchors")
        self.setVisible(False)  # Hidden by default, shown in multi-anchor mode

    @property
    def sub_anchors(self) -> list[AnchorPoint]:
        return list(self._sub_anchors)

    @property
    def anchors_dir(self) -> Path:
        return self._anchors_dir

    @anchors_dir.setter
    def anchors_dir(self, path: Path | str) -> None:
        self._anchors_dir = Path(path)

    def load_sub_anchors(self, anchors: list[AnchorPoint]) -> None:
        self._sub_anchors = list(anchors)
        self._refresh_list()

    def _refresh_list(self) -> None:
        self.list_widget.clear()
        for anchor in self._sub_anchors:
            search_hint = ""
            if anchor.search_region:
                sr = anchor.search_region
                search_hint = f"  search:({sr.get('x', 0)}, {sr.get('y', 0)}) {sr.get('width', 0)}×{sr.get('height', 0)}"
            threshold_pct = int(anchor.match_threshold * 100)
            text = (
                f"{anchor.name}  ref({anchor.ref_x:.0f}, {anchor.ref_y:.0f})  "
                f"thresh:{threshold_pct}%{search_hint}"
            )
            self.list_widget.addItem(text)

    def _on_add(self) -> None:
        dialog = SubAnchorEditDialog(self, None, self._anchors_dir)
        if dialog.exec():
            anchor = dialog.get_anchor()
            self._sub_anchors.append(anchor)
            self._refresh_list()
            self.sub_anchors_changed.emit()

    def _on_edit(self) -> None:
        idx = self.list_widget.currentRow()
        if idx < 0 or idx >= len(self._sub_anchors):
            return
        dialog = SubAnchorEditDialog(self, self._sub_anchors[idx], self._anchors_dir)
        if dialog.exec():
            new_anchor = dialog.get_anchor()
            self._sub_anchors[idx] = new_anchor
            self._refresh_list()
            self.list_widget.setCurrentRow(idx)
            self.sub_anchors_changed.emit()

    def _on_remove(self) -> None:
        idx = self.list_widget.currentRow()
        if idx < 0 or idx >= len(self._sub_anchors):
            return
        self._sub_anchors.pop(idx)
        self._refresh_list()
        self.sub_anchors_changed.emit()
