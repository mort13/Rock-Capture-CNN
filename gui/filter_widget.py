"""
Per-ROI image filter controls for Rock Capture CNN.
"""

from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QCheckBox, QComboBox,
)
from PyQt6.QtCore import Qt, pyqtSignal

from core.profile import FilterSettings


class FilterWidget(QGroupBox):
    """
    Image filter controls for the currently selected ROI.
    Emits filters_changed(FilterSettings) when any control changes.
    """
    filters_changed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__("Image Filters (selected ROI)", parent)
        self._building = False
        layout = QVBoxLayout(self)

        # Brightness
        self.brightness_slider, self.brightness_label = self._add_slider(
            layout, "Brightness:", -100, 100, 0
        )

        # Contrast
        self.contrast_slider, self.contrast_label = self._add_slider(
            layout, "Contrast:", -100, 100, 0
        )

        # Threshold
        thresh_row = QHBoxLayout()
        self.threshold_enabled = QCheckBox("Threshold:")
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(127)
        self.threshold_value_label = QLabel("127")
        thresh_row.addWidget(self.threshold_enabled)
        thresh_row.addWidget(self.threshold_slider)
        thresh_row.addWidget(self.threshold_value_label)
        layout.addLayout(thresh_row)

        # Channel
        channel_row = QHBoxLayout()
        channel_row.addWidget(QLabel("Channel:"))
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["None", "Red", "Green", "Blue"])
        channel_row.addWidget(self.channel_combo)
        layout.addLayout(channel_row)

        # Grayscale, Invert
        self.grayscale_cb = QCheckBox("Grayscale")
        self.grayscale_cb.setChecked(True)
        self.invert_cb = QCheckBox("Invert")
        layout.addWidget(self.grayscale_cb)
        layout.addWidget(self.invert_cb)

        # Connect signals
        self.brightness_slider.valueChanged.connect(self._on_changed)
        self.contrast_slider.valueChanged.connect(self._on_changed)
        self.threshold_slider.valueChanged.connect(self._on_changed)
        self.threshold_enabled.stateChanged.connect(self._on_changed)
        self.channel_combo.currentIndexChanged.connect(self._on_changed)
        self.grayscale_cb.stateChanged.connect(self._on_changed)
        self.invert_cb.stateChanged.connect(self._on_changed)

    def load_settings(self, settings: FilterSettings) -> None:
        """Populate widget controls from a FilterSettings object."""
        self._building = True
        self.brightness_slider.setValue(settings.brightness)
        self.contrast_slider.setValue(settings.contrast)
        self.threshold_slider.setValue(settings.threshold)
        self.threshold_enabled.setChecked(settings.threshold_enabled)
        self.grayscale_cb.setChecked(settings.grayscale)
        self.invert_cb.setChecked(settings.invert)
        channel_map = {"none": 0, "red": 1, "green": 2, "blue": 3}
        self.channel_combo.setCurrentIndex(channel_map.get(settings.channel, 0))
        self._building = False

    def get_settings(self) -> FilterSettings:
        """Read current control values into a FilterSettings."""
        channels = ["none", "red", "green", "blue"]
        return FilterSettings(
            brightness=self.brightness_slider.value(),
            contrast=self.contrast_slider.value(),
            threshold=self.threshold_slider.value(),
            threshold_enabled=self.threshold_enabled.isChecked(),
            grayscale=self.grayscale_cb.isChecked(),
            invert=self.invert_cb.isChecked(),
            channel=channels[self.channel_combo.currentIndex()],
        )

    def _on_changed(self) -> None:
        if self._building:
            return
        self.brightness_label.setText(str(self.brightness_slider.value()))
        self.contrast_label.setText(str(self.contrast_slider.value()))
        self.threshold_value_label.setText(str(self.threshold_slider.value()))
        self.filters_changed.emit(self.get_settings())

    def _add_slider(
        self, layout: QVBoxLayout, label_text: str, min_v: int, max_v: int, default: int
    ) -> tuple[QSlider, QLabel]:
        row = QHBoxLayout()
        row.addWidget(QLabel(label_text))
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setValue(default)
        row.addWidget(slider)
        value_label = QLabel(str(default))
        value_label.setMinimumWidth(30)
        row.addWidget(value_label)
        layout.addLayout(row)
        return slider, value_label
