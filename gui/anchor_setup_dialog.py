"""
Multi-anchor and ROI setup dialog for Rock Capture CNN.

Shows a captured reference frame and lets the user:
- Draw 2-3 rectangles for main anchor templates
- Draw rectangles for sub-anchor templates
- Draw rectangles for ROI definitions
All items are listed in a sidebar and can be removed.
"""

from __future__ import annotations

import uuid
import cv2
import numpy as np
from enum import Enum, auto
from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QDialogButtonBox, QRadioButton,
    QButtonGroup, QGroupBox, QMessageBox, QLineEdit, QFormLayout,
    QSplitter, QWidget, QComboBox,
)
from PyQt6.QtCore import Qt, QPoint, QRect
from PyQt6.QtGui import QPainter, QPen, QColor, QPixmap, QMouseEvent, QFont

from utils.image_utils import cv_to_qpixmap
from core.profile import AnchorPoint, ROIDefinition


class PlacementMode(Enum):
    ANCHOR = auto()
    SUB_ANCHOR = auto()
    ROI = auto()
    SEARCH_REGION = auto()


# ── Colours for each item type ─────────────────────────────────────

_COLORS = {
    PlacementMode.ANCHOR: QColor(0, 255, 0),         # green
    PlacementMode.SUB_ANCHOR: QColor(255, 0, 255),    # magenta
    PlacementMode.ROI: QColor(255, 255, 0),            # cyan-yellow
    PlacementMode.SEARCH_REGION: QColor(0, 255, 255), # cyan
}


class _PlacedItem:
    """Internal representation of an item drawn on the frame."""
    __slots__ = ("mode", "name", "rect", "sub_anchor_name", "search_region")

    def __init__(self, mode: PlacementMode, name: str, rect: QRect,
                 sub_anchor_name: str = "", search_region: dict | None = None):
        self.mode = mode
        self.name = name
        self.rect = rect  # in *frame* coordinates
        self.sub_anchor_name = sub_anchor_name  # only for ROI mode
        self.search_region = search_region or {}  # search region dict for SUB_ANCHOR items


class FrameCanvas(QLabel):
    """Displays a frame and lets the user draw rectangles on it."""

    def __init__(self, frame: np.ndarray, parent=None):
        super().__init__(parent)
        self._frame = frame
        self._display_scale = 1.0
        self._start = QPoint()
        self._end = QPoint()
        self._selecting = False
        self.pending_rect: QRect | None = None  # last drawn rect (frame coords)
        self.items: list[_PlacedItem] = []
        self.current_mode: PlacementMode = PlacementMode.ANCHOR

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._update_display()

    def _update_display(self) -> None:
        pixmap = cv_to_qpixmap(self._frame)
        max_w, max_h = 960, 720
        if pixmap.width() > max_w or pixmap.height() > max_h:
            scaled = pixmap.scaled(
                max_w, max_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._display_scale = pixmap.width() / scaled.width()
            self.setPixmap(scaled)
        else:
            self._display_scale = 1.0
            self.setPixmap(pixmap)
        self.setFixedSize(self.pixmap().size())

    # -- mouse interaction ------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._start = event.pos()
            self._end = event.pos()
            self._selecting = True
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._selecting:
            self._end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._selecting:
            self._selecting = False
            self._end = event.pos()
            s = self._display_scale
            x1 = int(min(self._start.x(), self._end.x()) * s)
            y1 = int(min(self._start.y(), self._end.y()) * s)
            x2 = int(max(self._start.x(), self._end.x()) * s)
            y2 = int(max(self._start.y(), self._end.y()) * s)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(self._frame.shape[1], x2)
            y2 = min(self._frame.shape[0], y2)
            w, h = x2 - x1, y2 - y1
            if w > 3 and h > 3:
                self.pending_rect = QRect(x1, y1, w, h)
            self.update()

    # -- painting ---------------------------------------------------------

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        s = self._display_scale

        # Draw existing items
        for item in self.items:
            pen = QPen(_COLORS[item.mode], 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            r = item.rect
            painter.drawRect(
                int(r.x() / s), int(r.y() / s),
                int(r.width() / s), int(r.height() / s),
            )
            painter.setFont(QFont("sans-serif", 8))
            painter.drawText(int(r.x() / s), int(r.y() / s) - 3, item.name)

        # Draw active drag rectangle
        if self._selecting:
            pen = QPen(_COLORS[self.current_mode], 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            x = min(self._start.x(), self._end.x())
            y = min(self._start.y(), self._end.y())
            w = abs(self._end.x() - self._start.x())
            h = abs(self._end.y() - self._start.y())
            painter.drawRect(x, y, w, h)

        # Draw pending (just drawn) rect
        elif self.pending_rect:
            pen = QPen(_COLORS[self.current_mode], 2, Qt.PenStyle.DashDotLine)
            painter.setPen(pen)
            r = self.pending_rect
            painter.drawRect(
                int(r.x() / s), int(r.y() / s),
                int(r.width() / s), int(r.height() / s),
            )

        painter.end()


class AnchorSetupDialog(QDialog):
    """
    Interactive dialog for placing anchors, sub-anchors, and ROIs on a
    captured reference frame.

    The dialog allows:
    - Drawing 2-3 main anchor rectangles (templates are auto-cropped & saved)
    - Drawing sub-anchor rectangles
    - Drawing ROI rectangles (optionally linked to a sub-anchor)

    Returns the placed anchors, sub-anchors, and ROI definitions.
    """

    def __init__(
        self,
        frame: np.ndarray,
        anchors_dir: Path,
        existing_anchors: list[AnchorPoint] | None = None,
        existing_sub_anchors: list[AnchorPoint] | None = None,
        existing_rois: list[ROIDefinition] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Anchor & ROI Setup")
        self.setModal(True)

        self._frame = frame
        self._anchors_dir = anchors_dir
        self._anchors_dir.mkdir(parents=True, exist_ok=True)

        main_layout = QHBoxLayout(self)

        # Left: canvas
        left = QVBoxLayout()
        info = QLabel(
            "Draw rectangles on the frame to define anchors and ROIs.\n"
            "Select a mode below, draw the rectangle, enter a name, click Add."
        )
        info.setStyleSheet("color: #aaa;")
        left.addWidget(info)

        self.canvas = FrameCanvas(frame)
        left.addWidget(self.canvas)

        main_layout.addLayout(left, stretch=3)

        # Right: controls
        right = QVBoxLayout()

        # Mode selector
        mode_group = QGroupBox("Placement Mode")
        mode_layout = QVBoxLayout(mode_group)
        self._mode_btn_group = QButtonGroup(self)
        self._anchor_radio = QRadioButton("Main Anchor (2-3 required)")
        self._sub_anchor_radio = QRadioButton("Sub-Anchor (optional)")
        self._search_region_radio = QRadioButton("Search Region (for sub-anchor)")
        self._roi_radio = QRadioButton("ROI")
        self._anchor_radio.setChecked(True)
        self._mode_btn_group.addButton(self._anchor_radio, PlacementMode.ANCHOR.value)
        self._mode_btn_group.addButton(self._sub_anchor_radio, PlacementMode.SUB_ANCHOR.value)
        self._mode_btn_group.addButton(self._search_region_radio, PlacementMode.SEARCH_REGION.value)
        self._mode_btn_group.addButton(self._roi_radio, PlacementMode.ROI.value)
        self._mode_btn_group.idToggled.connect(self._on_mode_changed)
        mode_layout.addWidget(self._anchor_radio)
        mode_layout.addWidget(self._sub_anchor_radio)
        mode_layout.addWidget(self._search_region_radio)
        mode_layout.addWidget(self._roi_radio)
        right.addWidget(mode_group)

        # Name + add/remove
        entry_group = QGroupBox("Item Properties")
        entry_layout = QFormLayout(entry_group)
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. anchor_top_left")
        entry_layout.addRow("Name:", self._name_edit)

        self._sub_anchor_combo = QComboBox()
        self._sub_anchor_combo.addItem("(none – use main transform)")
        self._sub_anchor_combo_label = QLabel("Relative to:")
        entry_layout.addRow(self._sub_anchor_combo_label, self._sub_anchor_combo)
        self._sub_anchor_combo.setVisible(False)
        self._sub_anchor_combo_label.setVisible(False)

        self._search_region_target_combo = QComboBox()
        self._search_region_target_combo.addItem("(select sub-anchor)")
        self._search_region_target_label = QLabel("For sub-anchor:")
        entry_layout.addRow(self._search_region_target_label, self._search_region_target_combo)
        self._search_region_target_combo.setVisible(False)
        self._search_region_target_label.setVisible(False)

        btn_row = QHBoxLayout()
        self._add_btn = QPushButton("Add Item")
        self._add_btn.clicked.connect(self._on_add_item)
        btn_row.addWidget(self._add_btn)

        self._remove_btn = QPushButton("Remove Selected")
        self._remove_btn.clicked.connect(self._on_remove_item)
        btn_row.addWidget(self._remove_btn)
        entry_layout.addRow(btn_row)
        right.addWidget(entry_group)

        # Items list
        self._items_list = QListWidget()
        right.addWidget(self._items_list)

        # Status
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #aaa; font-size: 10px;")
        right.addWidget(self._status_label)

        right.addStretch()

        # OK / Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        right.addWidget(buttons)

        main_layout.addLayout(right, stretch=1)

        # Load existing items
        self._load_existing(existing_anchors, existing_sub_anchors, existing_rois)
        self._update_status()

    # -- mode switching ---------------------------------------------------

    def _on_mode_changed(self, btn_id: int, checked: bool) -> None:
        if not checked:
            return
        mode = PlacementMode(btn_id)
        self.canvas.current_mode = mode
        is_roi = mode == PlacementMode.ROI
        is_search = mode == PlacementMode.SEARCH_REGION
        self._sub_anchor_combo.setVisible(is_roi)
        self._sub_anchor_combo_label.setVisible(is_roi)
        self._search_region_target_combo.setVisible(is_search)
        self._search_region_target_label.setVisible(is_search)

    def _current_mode(self) -> PlacementMode:
        return PlacementMode(self._mode_btn_group.checkedId())

    # -- add / remove items -----------------------------------------------

    def _on_add_item(self) -> None:
        rect = self.canvas.pending_rect
        if rect is None or rect.width() < 3 or rect.height() < 3:
            QMessageBox.warning(self, "No Rectangle", "Draw a rectangle on the frame first.")
            return

        mode = self._current_mode()

        # Handle search region mode separately
        if mode == PlacementMode.SEARCH_REGION:
            target_name = self._search_region_target_combo.currentText()
            if target_name == "(select sub-anchor)":
                QMessageBox.warning(
                    self, "No Target",
                    "Select a sub-anchor to attach the search region to."
                )
                return
            # Find the sub-anchor item and update its search_region
            found = False
            for item in self.canvas.items:
                if item.mode == PlacementMode.SUB_ANCHOR and item.name == target_name:
                    item.search_region = {
                        "x": float(rect.x()),
                        "y": float(rect.y()),
                        "width": float(rect.width()),
                        "height": float(rect.height()),
                    }
                    found = True
                    break
            if not found:
                QMessageBox.warning(
                    self, "Sub-anchor Not Found",
                    f"Could not find sub-anchor '{target_name}'."
                )
                return
            self.canvas.pending_rect = None
            self.canvas.update()
            self._refresh_list()
            self._update_status()
            return

        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "No Name", "Enter a name for this item.")
            return

        # Check uniqueness
        for item in self.canvas.items:
            if item.name == name and item.mode == mode:
                QMessageBox.warning(self, "Duplicate", f"An item named '{name}' already exists in this mode.")
                return

        # Check anchor count limit
        if mode == PlacementMode.ANCHOR:
            anchor_count = sum(1 for i in self.canvas.items if i.mode == PlacementMode.ANCHOR)
            if anchor_count >= 3:
                QMessageBox.warning(self, "Limit Reached", "Maximum 3 main anchors allowed.")
                return

        sub_anchor_name = ""
        if mode == PlacementMode.ROI and self._sub_anchor_combo.currentIndex() > 0:
            sub_anchor_name = self._sub_anchor_combo.currentText()

        item = _PlacedItem(mode, name, QRect(rect), sub_anchor_name)
        self.canvas.items.append(item)
        self.canvas.pending_rect = None
        self.canvas.update()

        # Save template for anchors/sub-anchors
        if mode in (PlacementMode.ANCHOR, PlacementMode.SUB_ANCHOR):
            self._save_template(name, rect)
            # Update sub-anchor combos
            self._refresh_sub_anchor_combo()
            self._refresh_search_region_combo()

        self._refresh_list()
        self._update_status()
        self._name_edit.clear()

    def _on_remove_item(self) -> None:
        row = self._items_list.currentRow()
        if row < 0 or row >= len(self.canvas.items):
            return
        removed = self.canvas.items.pop(row)
        self.canvas.update()

        if removed.mode in (PlacementMode.ANCHOR, PlacementMode.SUB_ANCHOR):
            self._refresh_sub_anchor_combo()
            self._refresh_search_region_combo()

        self._refresh_list()
        self._update_status()

    def _refresh_list(self) -> None:
        self._items_list.clear()
        for item in self.canvas.items:
            r = item.rect
            mode_label = item.mode.name.lower().replace("_", "-")
            sa_hint = f" → {item.sub_anchor_name}" if item.sub_anchor_name else ""
            search_hint = f" [search:{int(item.search_region.get('width', 0))}×{int(item.search_region.get('height', 0))}]" if item.search_region else ""
            text = (
                f"[{mode_label}] {item.name}  "
                f"({r.x()}, {r.y()}) {r.width()}×{r.height()}{sa_hint}{search_hint}"
            )
            self._items_list.addItem(text)

    def _refresh_sub_anchor_combo(self) -> None:
        current = self._sub_anchor_combo.currentText()
        self._sub_anchor_combo.clear()
        self._sub_anchor_combo.addItem("(none – use main transform)")
        for item in self.canvas.items:
            if item.mode == PlacementMode.SUB_ANCHOR:
                self._sub_anchor_combo.addItem(item.name)
        idx = self._sub_anchor_combo.findText(current)
        if idx >= 0:
            self._sub_anchor_combo.setCurrentIndex(idx)

    def _refresh_search_region_combo(self) -> None:
        current = self._search_region_target_combo.currentText()
        self._search_region_target_combo.clear()
        self._search_region_target_combo.addItem("(select sub-anchor)")
        for item in self.canvas.items:
            if item.mode == PlacementMode.SUB_ANCHOR:
                self._search_region_target_combo.addItem(item.name)
        idx = self._search_region_target_combo.findText(current)
        if idx >= 0:
            self._search_region_target_combo.setCurrentIndex(idx)

    def _update_status(self) -> None:
        n_anchors = sum(1 for i in self.canvas.items if i.mode == PlacementMode.ANCHOR)
        n_sub = sum(1 for i in self.canvas.items if i.mode == PlacementMode.SUB_ANCHOR)
        n_rois = sum(1 for i in self.canvas.items if i.mode == PlacementMode.ROI)
        parts = [f"Anchors: {n_anchors}/2-3"]
        if n_sub:
            parts.append(f"Sub-anchors: {n_sub}")
        parts.append(f"ROIs: {n_rois}")
        if n_anchors < 2:
            parts.append("⚠ Need at least 2 main anchors")
        self._status_label.setText("  |  ".join(parts))

    # -- template management ----------------------------------------------

    def _save_template(self, name: str, rect: QRect) -> None:
        """Crop the template from the frame and save to anchors_dir."""
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        crop = self._frame[y:y+h, x:x+w]
        if crop.size == 0:
            return
        path = self._anchors_dir / f"{name}.png"
        cv2.imwrite(str(path), crop)

    # -- load existing items ----------------------------------------------

    def _load_existing(
        self,
        anchors: list[AnchorPoint] | None,
        sub_anchors: list[AnchorPoint] | None,
        rois: list[ROIDefinition] | None,
    ) -> None:
        """Populate the canvas with existing profile items."""
        if anchors:
            for ap in anchors:
                # We don't know the original rect size, so use a small marker
                # Load the template to get the size
                tpl_path = self._anchors_dir / ap.template_path
                if tpl_path.exists():
                    tpl = cv2.imread(str(tpl_path))
                    if tpl is not None:
                        h, w = tpl.shape[:2]
                        rect = QRect(int(ap.ref_x), int(ap.ref_y), w, h)
                        item = _PlacedItem(PlacementMode.ANCHOR, ap.name, rect)
                        self.canvas.items.append(item)

        if sub_anchors:
            for ap in sub_anchors:
                tpl_path = self._anchors_dir / ap.template_path
                if tpl_path.exists():
                    tpl = cv2.imread(str(tpl_path))
                    if tpl is not None:
                        h, w = tpl.shape[:2]
                        rect = QRect(int(ap.ref_x), int(ap.ref_y), w, h)
                        item = _PlacedItem(
                            PlacementMode.SUB_ANCHOR, ap.name, rect,
                            search_region=ap.search_region
                        )
                        self.canvas.items.append(item)

        if rois:
            for roi in rois:
                rect = QRect(int(roi.ref_x), int(roi.ref_y), roi.width, roi.height)
                item = _PlacedItem(PlacementMode.ROI, roi.name, rect, roi.sub_anchor)
                self.canvas.items.append(item)

        self._refresh_sub_anchor_combo()
        self._refresh_search_region_combo()
        self._refresh_list()
        self.canvas.update()

    # -- accept / result --------------------------------------------------

    def _on_accept(self) -> None:
        n_anchors = sum(1 for i in self.canvas.items if i.mode == PlacementMode.ANCHOR)
        if n_anchors < 2:
            QMessageBox.warning(
                self, "Not Enough Anchors",
                "Place at least 2 main anchor rectangles before confirming.",
            )
            return
        self.accept()

    def get_anchors(self) -> list[AnchorPoint]:
        """Return the configured main anchors."""
        result = []
        for item in self.canvas.items:
            if item.mode != PlacementMode.ANCHOR:
                continue
            result.append(AnchorPoint(
                name=item.name,
                template_path=f"{item.name}.png",
                match_threshold=0.7,
                ref_x=float(item.rect.x()),
                ref_y=float(item.rect.y()),
            ))
        return result

    def get_sub_anchors(self) -> list[AnchorPoint]:
        """Return the configured sub-anchors."""
        result = []
        for item in self.canvas.items:
            if item.mode != PlacementMode.SUB_ANCHOR:
                continue
            ap = AnchorPoint(
                name=item.name,
                template_path=f"{item.name}.png",
                match_threshold=0.7,
                ref_x=float(item.rect.x()),
                ref_y=float(item.rect.y()),
                search_region=item.search_region if item.search_region else None,
            )
            result.append(ap)
        return result

    def get_rois(self) -> list[ROIDefinition]:
        """Return ROI definitions placed in this dialog.

        Only ref_x, ref_y, width, height, name, and sub_anchor are set.
        Other ROI properties (filters, recognition mode, etc.) keep defaults
        and should be configured via the ROI editor afterwards.
        """
        result = []
        for item in self.canvas.items:
            if item.mode != PlacementMode.ROI:
                continue
            result.append(ROIDefinition(
                name=item.name,
                ref_x=float(item.rect.x()),
                ref_y=float(item.rect.y()),
                width=item.rect.width(),
                height=item.rect.height(),
                sub_anchor=item.sub_anchor_name,
            ))
        return result
