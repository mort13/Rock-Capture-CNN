"""
Right-side controls panel for Rock Capture CNN.
Hosts anchor, ROI editor, filter, CNN model, results, and labeler widgets.
"""

from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QGroupBox, QPushButton, QLabel,
    QHBoxLayout, QSlider, QFileDialog,
)
from PyQt6.QtCore import Qt, pyqtSignal

from core.profile import FilterSettings
from gui.filter_widget import FilterWidget
from gui.roi_editor import ROIEditor
from gui.labeler_widget import LabelerWidget


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
    anchor_threshold_changed = pyqtSignal(float)
    filters_changed = pyqtSignal(object)
    roi_selected_signal = pyqtSignal(int)
    rois_changed = pyqtSignal()
    train_requested = pyqtSignal()
    load_model_requested = pyqtSignal()
    labeler_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        container = QWidget()
        layout = QVBoxLayout(container)

        # 1. Anchor section
        layout.addWidget(self._create_anchor_group())

        # 2. ROI editor
        self.roi_editor = ROIEditor()
        self.roi_editor.roi_selected.connect(self._on_roi_selected)
        self.roi_editor.rois_changed.connect(self._on_rois_changed)
        layout.addWidget(self.roi_editor)

        # 3. Filter controls
        self.filter_widget = FilterWidget()
        self.filter_widget.filters_changed.connect(self._on_filters_changed)
        layout.addWidget(self.filter_widget)

        # 4. CNN model
        layout.addWidget(self._create_model_group())

        # 5. Results display
        layout.addWidget(self._create_results_group())

        # 6. Labeler
        self.labeler_widget = LabelerWidget()
        self.labeler_widget.enable_cb.stateChanged.connect(
            lambda state: self.labeler_toggled.emit(bool(state))
        )
        layout.addWidget(self.labeler_widget)

        layout.addStretch()
        scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _create_anchor_group(self) -> QGroupBox:
        group = QGroupBox("Anchor Template")
        layout = QVBoxLayout(group)

        self.anchor_browse_btn = QPushButton("Browse...")
        self.anchor_browse_btn.clicked.connect(self.anchor_browse_clicked.emit)
        layout.addWidget(self.anchor_browse_btn)

        self.anchor_path_label = QLabel("No template loaded")
        self.anchor_path_label.setStyleSheet("color: gray; font-size: 10px;")
        self.anchor_path_label.setWordWrap(True)
        layout.addWidget(self.anchor_path_label)

        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Match threshold:"))
        self.anchor_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.anchor_threshold_slider.setRange(30, 99)
        self.anchor_threshold_slider.setValue(70)
        self.anchor_threshold_slider.valueChanged.connect(self._on_anchor_thresh_changed)
        thresh_row.addWidget(self.anchor_threshold_slider)
        self.anchor_threshold_label = QLabel("0.70")
        thresh_row.addWidget(self.anchor_threshold_label)
        layout.addLayout(thresh_row)

        self.anchor_status = QLabel("Status: --")
        self.anchor_status.setStyleSheet("font-size: 10px;")
        layout.addWidget(self.anchor_status)

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
        layout.addLayout(roi_row)

        self.anchor_roi_label = QLabel("Anchor ROI: full frame")
        self.anchor_roi_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.anchor_roi_label)

        return group

    def _create_model_group(self) -> QGroupBox:
        group = QGroupBox("CNN Model")
        layout = QVBoxLayout(group)

        self.model_status = QLabel("No model loaded")
        self.model_status.setStyleSheet("color: gray; font-size: 10px;")
        self.model_status.setWordWrap(True)
        layout.addWidget(self.model_status)

        self.train_btn = QPushButton("Train on Labeled Data")
        self.train_btn.clicked.connect(self.train_requested.emit)
        layout.addWidget(self.train_btn)

        self.load_model_btn = QPushButton("Load Model...")
        self.load_model_btn.clicked.connect(self.load_model_requested.emit)
        layout.addWidget(self.load_model_btn)

        return group

    def _create_results_group(self) -> QGroupBox:
        group = QGroupBox("Recognized Values")
        self._results_layout = QVBoxLayout(group)
        self._result_labels: dict[str, QLabel] = {}
        self._no_results_label = QLabel("No results yet")
        self._no_results_label.setStyleSheet("color: gray;")
        self._results_layout.addWidget(self._no_results_label)
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

    def _on_filters_changed(self, settings: FilterSettings) -> None:
        self.roi_editor.update_selected_filters(settings)
        self.filters_changed.emit(settings)

    def update_results(self, roi_results: list, profile_name: str = "") -> None:
        """Update the results display with new recognized values."""
        if self._no_results_label.isVisible() and roi_results:
            self._no_results_label.hide()

        for result in roi_results:
            key = f"{profile_name}/{result.name}" if profile_name else result.name
            if key not in self._result_labels:
                label = QLabel()
                label.setStyleSheet("font-size: 13px;")
                self._result_labels[key] = label
                self._results_layout.addWidget(label)
            self._result_labels[key].setText(
                f"{key}: {result.recognized_text or '--'}"
            )

    def remove_stale_results(self, prefix: str, active_keys: set[str]) -> None:
        """Remove result labels whose key starts with *prefix* but is not in *active_keys*."""
        for key in list(self._result_labels):
            if key.startswith(prefix) and key not in active_keys:
                label = self._result_labels.pop(key)
                self._results_layout.removeWidget(label)
                label.deleteLater()

    def update_anchor_status(self, text: str, found: bool = True) -> None:
        color = "#0a0" if found else "#a00"
        self.anchor_status.setText(text)
        self.anchor_status.setStyleSheet(f"color: {color}; font-size: 10px;")

    def is_labeler_active(self) -> bool:
        return self.labeler_widget.is_active()
