"""
Right-side controls panel for Rock Capture CNN.
Hosts anchor, ROI editor, filter, CNN model, results, and labeler widgets.
"""

from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QGroupBox, QPushButton, QLabel,
    QHBoxLayout, QSlider, QFileDialog, QLineEdit, QCheckBox,
)
from PyQt6.QtCore import Qt, pyqtSignal

from core.profile import FilterSettings
from gui.filter_widget import FilterWidget
from gui.roi_editor import ROIEditor, SubAnchorEditor
from gui.labeler_widget import LabelerWidget
from gui.word_labeler_widget import WordLabelerWidget


class ControlsPanel(QWidget):
    """
    Right-side scrollable panel hosting all control sub-widgets.

    Signals:
        anchor_browse_clicked(): user wants to browse for anchor template
        anchor_threshold_changed(float): anchor match threshold changed
        filters_changed(FilterSettings): filter settings changed for selected ROI
        roi_selected(int): ROI index selected in list
        rois_changed(): ROIs added/removed/edited
        train_requested(): user clicked Train button
        load_model_requested(): user clicked Load Model button
        labeler_toggled(bool): labeler mode toggled
    """
    anchor_browse_clicked = pyqtSignal()
    anchor_roi_clicked = pyqtSignal()
    anchor_roi_clear_clicked = pyqtSignal()
    anchor_setup_clicked = pyqtSignal()
    anchor_threshold_changed = pyqtSignal(float)
    filters_changed = pyqtSignal(object)
    roi_selected_signal = pyqtSignal(int)
    rois_changed = pyqtSignal()
    sub_anchors_changed = pyqtSignal()
    overlay_visibility_changed = pyqtSignal(dict)
    labeler_toggled = pyqtSignal(bool)
    word_labeler_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        container = QWidget()
        layout = QVBoxLayout(container)

        # 1. Anchor section
        layout.addWidget(self._create_anchor_group())

        # 1b. Visibility toggles
        layout.addWidget(self._create_visibility_group())

        # 2. Sub-anchor editor (multi-anchor mode)
        self.sub_anchor_editor = SubAnchorEditor()
        self.sub_anchor_editor.sub_anchors_changed.connect(self._on_sub_anchors_changed)
        layout.addWidget(self.sub_anchor_editor)

        # 3. ROI editor
        self.roi_editor = ROIEditor()
        self.roi_editor.roi_selected.connect(self._on_roi_selected)
        self.roi_editor.rois_changed.connect(self._on_rois_changed)
        layout.addWidget(self.roi_editor)

        # 3. Filter controls
        self.filter_widget = FilterWidget()
        self.filter_widget.filters_changed.connect(self._on_filters_changed)
        layout.addWidget(self.filter_widget)

        # 4. Results display
        layout.addWidget(self._create_results_group())

        # 6. Labeler
        self.labeler_widget = LabelerWidget()
        self.labeler_widget.enable_cb.stateChanged.connect(
            lambda state: self.labeler_toggled.emit(bool(state))
        )
        layout.addWidget(self.labeler_widget)

        # 7. Word Labeler
        self.word_labeler_widget = WordLabelerWidget()
        self.word_labeler_widget.enable_cb.stateChanged.connect(
            lambda state: self.word_labeler_toggled.emit(bool(state))
        )
        layout.addWidget(self.word_labeler_widget)

        layout.addStretch()
        scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _create_anchor_group(self) -> QGroupBox:
        group = QGroupBox("Anchor / Positioning")
        layout = QVBoxLayout(group)

        # Multi-anchor setup button
        self.anchor_setup_btn = QPushButton("Setup Anchors && ROIs…")
        self.anchor_setup_btn.setToolTip(
            "Open a visual dialog to place 2-3 main anchors,\n"
            "optional sub-anchors, and ROIs on a captured frame.\n"
            "This enables resolution-independent positioning."
        )
        self.anchor_setup_btn.clicked.connect(self.anchor_setup_clicked.emit)
        layout.addWidget(self.anchor_setup_btn)

        self.multi_anchor_status = QLabel("Multi-anchor: not configured")
        self.multi_anchor_status.setStyleSheet("color: gray; font-size: 10px;")
        self.multi_anchor_status.setWordWrap(True)
        layout.addWidget(self.multi_anchor_status)

        # --- Legacy single-anchor section ---
        self._legacy_anchor_group = QGroupBox("Legacy Single Anchor")
        legacy_layout = QVBoxLayout(self._legacy_anchor_group)

        self.anchor_browse_btn = QPushButton("Browse...")
        self.anchor_browse_btn.clicked.connect(self.anchor_browse_clicked.emit)
        legacy_layout.addWidget(self.anchor_browse_btn)

        self.anchor_path_label = QLabel("No template loaded")
        self.anchor_path_label.setStyleSheet("color: gray; font-size: 10px;")
        self.anchor_path_label.setWordWrap(True)
        legacy_layout.addWidget(self.anchor_path_label)

        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Match threshold:"))
        self.anchor_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.anchor_threshold_slider.setRange(30, 99)
        self.anchor_threshold_slider.setValue(70)
        self.anchor_threshold_slider.valueChanged.connect(self._on_anchor_thresh_changed)
        thresh_row.addWidget(self.anchor_threshold_slider)
        self.anchor_threshold_label = QLabel("0.70")
        thresh_row.addWidget(self.anchor_threshold_label)
        legacy_layout.addLayout(thresh_row)

        self.anchor_status = QLabel("Status: --")
        self.anchor_status.setStyleSheet("font-size: 10px;")
        legacy_layout.addWidget(self.anchor_status)

        # Anchor ROI (constrains where template matching searches)
        roi_row = QHBoxLayout()
        self.anchor_roi_btn = QPushButton("Set Anchor ROI")
        self.anchor_roi_btn.setToolTip(
            "Select a sub-region within the search area to limit anchor matching.\n"
            "This speeds up processing by searching a smaller area."
        )
        self.anchor_roi_btn.clicked.connect(self.anchor_roi_clicked.emit)
        roi_row.addWidget(self.anchor_roi_btn)

        self.anchor_roi_clear_btn = QPushButton("Clear")
        self.anchor_roi_clear_btn.setFixedWidth(50)
        self.anchor_roi_clear_btn.setToolTip("Search the full capture region")
        self.anchor_roi_clear_btn.clicked.connect(self.anchor_roi_clear_clicked.emit)
        roi_row.addWidget(self.anchor_roi_clear_btn)
        legacy_layout.addLayout(roi_row)

        self.anchor_roi_label = QLabel("Anchor ROI: full frame")
        self.anchor_roi_label.setStyleSheet("color: gray; font-size: 10px;")
        legacy_layout.addWidget(self.anchor_roi_label)

        layout.addWidget(self._legacy_anchor_group)

        return group

    def _create_visibility_group(self) -> QGroupBox:
        group = QGroupBox("Overlay Visibility")
        layout = QVBoxLayout(group)

        self.show_anchors_cb = QCheckBox("Show Anchors")
        self.show_anchors_cb.setChecked(True)
        self.show_anchors_cb.stateChanged.connect(lambda: self._on_visibility_changed())
        layout.addWidget(self.show_anchors_cb)

        self.show_sub_anchors_cb = QCheckBox("Show Sub-Anchors")
        self.show_sub_anchors_cb.setChecked(True)
        self.show_sub_anchors_cb.stateChanged.connect(lambda: self._on_visibility_changed())
        layout.addWidget(self.show_sub_anchors_cb)

        self.show_search_regions_cb = QCheckBox("Show Search Regions")
        self.show_search_regions_cb.setChecked(True)
        self.show_search_regions_cb.stateChanged.connect(lambda: self._on_visibility_changed())
        layout.addWidget(self.show_search_regions_cb)

        self.show_rois_cb = QCheckBox("Show ROIs")
        self.show_rois_cb.setChecked(True)
        self.show_rois_cb.stateChanged.connect(lambda: self._on_visibility_changed())
        layout.addWidget(self.show_rois_cb)

        return group

    def _on_visibility_changed(self) -> None:
        """Emit signal when visibility settings change."""
        self.overlay_visibility_changed.emit(self.get_overlay_visibility())

    def get_overlay_visibility(self) -> dict:
        """Return current overlay visibility settings."""
        return {
            "show_anchors": self.show_anchors_cb.isChecked(),
            "show_sub_anchors": self.show_sub_anchors_cb.isChecked(),
            "show_search_regions": self.show_search_regions_cb.isChecked(),
            "show_rois": self.show_rois_cb.isChecked(),
        }

    def _create_results_group(self) -> QGroupBox:
        group = QGroupBox("Recognized Values")
        self._results_layout = QVBoxLayout(group)
        self._result_labels: dict[str, QLabel] = {}
        self._result_history: dict[str, list[str]] = {}
        self._no_results_label = QLabel("No results yet")
        self._no_results_label.setStyleSheet("color: gray;")
        self._results_layout.addWidget(self._no_results_label)
        # Staged editable fields (key -> QLineEdit)
        self._staged_edits: dict[str, QLineEdit] = {}
        self._is_staged = False
        self._validation_callback = None
        return group

    def _on_anchor_thresh_changed(self) -> None:
        val = self.anchor_threshold_slider.value() / 100.0
        self.anchor_threshold_label.setText(f"{val:.2f}")
        self.anchor_threshold_changed.emit(val)

    def _on_roi_selected(self, index: int) -> None:
        roi = self.roi_editor.get_selected_roi()
        if roi:
            self.filter_widget.load_settings(roi.filters)
        self.roi_selected_signal.emit(index)

    def _on_rois_changed(self) -> None:
        self.rois_changed.emit()

    def _on_sub_anchors_changed(self) -> None:
        self.sub_anchors_changed.emit()

    def _on_filters_changed(self, settings: FilterSettings) -> None:
        self.roi_editor.update_selected_filters(settings)
        self.filters_changed.emit(settings)

    def update_results(self, roi_results: list, profile_name: str = "") -> None:
        """Update the results display with new recognized values (skipped while staged)."""
        if self._is_staged:
            return

        if self._no_results_label.isVisible() and roi_results:
            self._no_results_label.hide()

        for result in roi_results:
            key = f"{profile_name}/{result.name}" if profile_name else result.name
            current_text = result.recognized_text or "--"

            # Track last 10 values to detect stability.
            history = self._result_history.setdefault(key, [])
            history.append(current_text)
            if len(history) > 10:
                history.pop(0)
            stable = len(history) == 10 and all(v == current_text for v in history)

            if key not in self._result_labels:
                label = QLabel()
                self._result_labels[key] = label
                self._results_layout.addWidget(label)
            conf_str = f" ({result.confidence:.2f})" if result.confidence > 0 else ""
            self._result_labels[key].setText(f"{key}: {current_text}{conf_str}")
            if stable:
                self._result_labels[key].setStyleSheet(
                    "font-size: 13px; background-color: #90ee90; color: #000; border-radius: 3px; padding: 1px 3px;"
                )
            else:
                self._result_labels[key].setStyleSheet("font-size: 13px;")

    def freeze_staged(
        self,
        staged_values: dict[str, str],
        red_keys: set[str] = frozenset(),
        validation_callback=None,
    ) -> None:
        """Replace result labels with editable fields frozen to staged values.
        
        Args:
            staged_values: Dict of field keys to initial values
            red_keys: Set of keys that should be highlighted red
            validation_callback: Callable that takes current staged values and returns updated red_keys
        """
        self._is_staged = True
        self._validation_callback = validation_callback
        # Hide live labels
        for label in self._result_labels.values():
            label.hide()
        self._no_results_label.hide()
        # Create editable fields for each staged entry
        for key, value in staged_values.items():
            row = QHBoxLayout()
            lbl = QLabel(f"{key}:")
            lbl.setStyleSheet("font-size: 13px; min-width: 100px;")
            edit = QLineEdit(value)
            self._update_field_style(edit, key in red_keys)
            edit.setPlaceholderText("(empty = no value)")
            # Connect text changes to real-time validation
            edit.textChanged.connect(self._on_staged_value_changed)
            row.addWidget(lbl)
            row.addWidget(edit)
            container = QWidget()
            container.setLayout(row)
            self._results_layout.addWidget(container)
            self._staged_edits[key] = edit

    def _update_field_style(self, edit: QLineEdit, is_red: bool) -> None:
        """Update field styling based on validation status."""
        if is_red:
            edit.setStyleSheet(
                "font-size: 13px; background-color: #ffaaaa; color: #000; padding: 2px;"
            )
        else:
            edit.setStyleSheet(
                "font-size: 13px; background-color: #ffffcc; padding: 2px;"
            )

    def _on_staged_value_changed(self) -> None:
        """Called when any staged value changes; updates styling in real-time."""
        if not self._validation_callback:
            return
        current_values = self.get_staged_edits()
        new_red_keys = self._validation_callback(current_values)
        # Update styling for all fields based on new red_keys
        for key, edit in self._staged_edits.items():
            self._update_field_style(edit, key in new_red_keys)

    def get_staged_edits(self) -> dict[str, str]:
        """Return the current edited staged values."""
        return {key: edit.text() for key, edit in self._staged_edits.items()}

    def unfreeze_staged(self) -> None:
        """Remove staged edit fields and restore live result labels."""
        self._is_staged = False
        for key in list(self._staged_edits):
            edit = self._staged_edits.pop(key)
            container = edit.parentWidget()
            if container:
                self._results_layout.removeWidget(container)
                container.deleteLater()
        # Show live labels again
        for label in self._result_labels.values():
            label.show()
        if not self._result_labels:
            self._no_results_label.show()

    def remove_stale_results(self, prefix: str, active_keys: set[str]) -> None:
        """Remove result labels whose key starts with *prefix* but is not in *active_keys*."""
        for key in list(self._result_labels):
            if key.startswith(prefix) and key not in active_keys:
                label = self._result_labels.pop(key)
                self._results_layout.removeWidget(label)
                label.deleteLater()
                self._result_history.pop(key, None)

    def update_anchor_status(self, text: str, found: bool = True) -> None:
        color = "#0a0" if found else "#a00"
        self.anchor_status.setText(text)
        self.anchor_status.setStyleSheet(f"color: {color}; font-size: 10px;")

    def is_labeler_active(self) -> bool:
        return self.labeler_widget.is_active()
