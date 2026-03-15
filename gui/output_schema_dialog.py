"""
Dialog for editing the output_schema of a HUD profile.

The schema is an ordered list of SchemaNode trees.  Each top-level SchemaNode
maps to one key in the committed capture JSON.  Nodes can nest arbitrarily:
- "object" nodes collect their children into a dict
- "array"  nodes collect their children into a JSON array

Leaf nodes (ROIRef) reference a single ROI from a named profile.
"""

import copy

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QFont

from core.profile import ROIRef, SchemaNode

_ROLE = Qt.ItemDataRole.UserRole

_BOLD_FONT = QFont()
_BOLD_FONT.setBold(True)

_BRUSH_NODE = QBrush(QColor("#4a9eff"))   # blue — SchemaNode
_BRUSH_REF = QBrush(QColor("#4caf50"))    # green — ROIRef


class OutputSchemaDialog(QDialog):
    """Modal tree editor for a list[SchemaNode] output schema."""

    def __init__(
        self,
        schema: list[SchemaNode],
        profile_rois: dict[str, list[str]],   # profile_name -> ordered list of ROI names
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Edit Output Schema")
        self.setMinimumSize(960, 600)
        self.setModal(True)

        # Deep-copy so Cancel leaves the original untouched
        self._schema: list[SchemaNode] = [self._copy_node(n) for n in schema]
        self._profile_rois = profile_rois

        self._build_ui()
        self._populate_tree()

    # ── deep copy helpers ─────────────────────────────────────────

    @staticmethod
    def _copy_node(node) -> SchemaNode | ROIRef:
        if isinstance(node, ROIRef):
            return ROIRef(profile=node.profile, roi=node.roi, key=node.key)
        return SchemaNode(
            key=node.key,
            type=node.type,
            children=[OutputSchemaDialog._copy_node(c) for c in node.children],
        )

    # ── UI construction ───────────────────────────────────────────

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)

        body = QHBoxLayout()
        root_layout.addLayout(body)

        # ── Left: tree + toolbar ──────────────────────────────────
        left = QVBoxLayout()

        toolbar = QHBoxLayout()
        self._add_node_btn = QPushButton("Add Group")
        self._add_node_btn.setToolTip("Add a new child SchemaNode under the selected node (or at root)")
        self._add_node_btn.clicked.connect(self._on_add_node)
        toolbar.addWidget(self._add_node_btn)

        self._add_ref_btn = QPushButton("Add ROI Ref")
        self._add_ref_btn.setToolTip("Add a ROIRef leaf under the selected SchemaNode")
        self._add_ref_btn.clicked.connect(self._on_add_ref)
        toolbar.addWidget(self._add_ref_btn)

        self._del_btn = QPushButton("Delete")
        self._del_btn.setToolTip("Delete the selected item and all its children")
        self._del_btn.clicked.connect(self._on_delete)
        toolbar.addWidget(self._del_btn)

        self._up_btn = QPushButton("↑")
        self._up_btn.setFixedWidth(30)
        self._up_btn.clicked.connect(self._on_move_up)
        toolbar.addWidget(self._up_btn)

        self._dn_btn = QPushButton("↓")
        self._dn_btn.setFixedWidth(30)
        self._dn_btn.clicked.connect(self._on_move_down)
        toolbar.addWidget(self._dn_btn)

        toolbar.addStretch()
        left.addLayout(toolbar)

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setAlternatingRowColors(True)
        self._tree.currentItemChanged.connect(self._on_selection_changed)
        self._tree.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left.addWidget(self._tree)
        body.addLayout(left, 3)

        # ── Right: detail panel (stacked) ─────────────────────────
        self._stack = QStackedWidget()
        self._stack.setFixedWidth(300)
        body.addWidget(self._stack)

        # Page 0 — nothing selected
        self._stack.addWidget(QLabel("  Select an item in the tree."))

        # Page 1 — SchemaNode editor
        node_page = QWidget()
        nv = QVBoxLayout(node_page)
        nv.setContentsMargins(8, 8, 8, 8)
        nv.addWidget(QLabel("<b>Group / Node</b>"))

        nv.addWidget(QLabel("Output key:"))
        self._node_key_edit = QLineEdit()
        self._node_key_edit.setPlaceholderText("e.g.  scan  or  materials")
        self._node_key_edit.editingFinished.connect(self._on_node_key_changed)
        nv.addWidget(self._node_key_edit)

        nv.addWidget(QLabel("Type:"))
        self._node_type_combo = QComboBox()
        self._node_type_combo.addItems(["object", "array"])
        self._node_type_combo.setToolTip(
            "object — children become key/value pairs in a dict\n"
            "array  — children become elements of a JSON array"
        )
        self._node_type_combo.currentTextChanged.connect(self._on_node_type_changed)
        nv.addWidget(self._node_type_combo)
        nv.addStretch()
        self._stack.addWidget(node_page)  # index 1

        # Page 2 — ROIRef editor
        ref_page = QWidget()
        rv = QVBoxLayout(ref_page)
        rv.setContentsMargins(8, 8, 8, 8)
        rv.addWidget(QLabel("<b>ROI Reference</b>"))

        rv.addWidget(QLabel("Output key (blank = use ROI name):"))
        self._ref_key_edit = QLineEdit()
        self._ref_key_edit.setPlaceholderText("leave blank to use ROI name")
        self._ref_key_edit.editingFinished.connect(self._on_ref_key_changed)
        rv.addWidget(self._ref_key_edit)

        rv.addWidget(QLabel("Profile:"))
        self._ref_profile_combo = QComboBox()
        profiles = sorted(self._profile_rois.keys())
        self._ref_profile_combo.addItems(profiles)
        self._ref_profile_combo.currentTextChanged.connect(self._on_ref_profile_changed)
        rv.addWidget(self._ref_profile_combo)

        rv.addWidget(QLabel("ROI:"))
        self._ref_roi_combo = QComboBox()
        self._ref_roi_combo.currentTextChanged.connect(self._on_ref_roi_changed)
        rv.addWidget(self._ref_roi_combo)

        rv.addStretch()
        self._stack.addWidget(ref_page)  # index 2

        # ── Dialog buttons ────────────────────────────────────────
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        root_layout.addWidget(btn_box)

    # ── Tree population ───────────────────────────────────────────

    def _populate_tree(self) -> None:
        self._tree.blockSignals(True)
        self._tree.clear()
        for node in self._schema:
            item = self._make_tree_item(node)
            self._tree.addTopLevelItem(item)
        self._tree.expandAll()
        self._tree.blockSignals(False)
        self._stack.setCurrentIndex(0)

    def _make_tree_item(self, obj: SchemaNode | ROIRef) -> QTreeWidgetItem:
        item = QTreeWidgetItem()
        item.setData(0, _ROLE, obj)
        if isinstance(obj, SchemaNode):
            item.setFont(0, _BOLD_FONT)
            item.setForeground(0, _BRUSH_NODE)
            for child in obj.children:
                item.addChild(self._make_tree_item(child))
        else:
            item.setForeground(0, _BRUSH_REF)
        self._refresh_item_text(item)
        return item

    def _refresh_item_text(self, item: QTreeWidgetItem) -> None:
        obj = item.data(0, _ROLE)
        if isinstance(obj, SchemaNode):
            item.setText(0, f"[{obj.type}]  {obj.key}")
        else:
            out_key = obj.key or obj.roi
            item.setText(0, f"{out_key}  ←  {obj.profile}.{obj.roi}")

    # ── Parent context helpers ─────────────────────────────────────

    def _get_parent_list_and_index(self, item: QTreeWidgetItem):
        """Return (children_list, index) for the data object of item."""
        obj = item.data(0, _ROLE)
        parent_item = item.parent()
        if parent_item is None:
            lst = self._schema
        else:
            lst = parent_item.data(0, _ROLE).children
        idx = next(i for i, c in enumerate(lst) if c is obj)
        return lst, idx

    def _target_for_add(self):
        """Return (SchemaNode, QTreeWidgetItem | None) where a new child will be added.

        If a SchemaNode is selected: use it.
        If a ROIRef is selected: use its parent SchemaNode.
        If nothing: use root level (None acts as top-level sentinel).
        """
        item = self._tree.currentItem()
        if item is None:
            return None, None
        obj = item.data(0, _ROLE)
        if isinstance(obj, ROIRef):
            parent_item = item.parent()
            if parent_item is None:
                return None, None
            return parent_item.data(0, _ROLE), parent_item
        return obj, item

    # ── Tree slot helpers ─────────────────────────────────────────

    def _on_add_node(self) -> None:
        new_node = SchemaNode(key="new_group", type="object")
        target_node, target_item = self._target_for_add()
        new_item = self._make_tree_item(new_node)
        if target_node is None:
            self._schema.append(new_node)
            self._tree.addTopLevelItem(new_item)
        else:
            target_node.children.append(new_node)
            target_item.addChild(new_item)
            target_item.setExpanded(True)
        self._tree.setCurrentItem(new_item)

    def _on_add_ref(self) -> None:
        target_node, target_item = self._target_for_add()
        profiles = sorted(self._profile_rois.keys())
        if not profiles:
            return
        first_profile = profiles[0]
        rois = self._profile_rois.get(first_profile, [])
        first_roi = rois[0] if rois else ""
        new_ref = ROIRef(profile=first_profile, roi=first_roi, key="")
        new_item = self._make_tree_item(new_ref)
        if target_node is None:
            # ROIRefs at root level: still add them, just at top
            self._schema.append(new_ref)
            self._tree.addTopLevelItem(new_item)
        else:
            target_node.children.append(new_ref)
            target_item.addChild(new_item)
            target_item.setExpanded(True)
        self._tree.setCurrentItem(new_item)

    def _on_delete(self) -> None:
        item = self._tree.currentItem()
        if item is None:
            return
        lst, idx = self._get_parent_list_and_index(item)
        lst.pop(idx)
        parent_item = item.parent()
        if parent_item is None:
            self._tree.takeTopLevelItem(self._tree.indexOfTopLevelItem(item))
        else:
            parent_item.removeChild(item)

    def _on_move_up(self) -> None:
        item = self._tree.currentItem()
        if item is None:
            return
        lst, idx = self._get_parent_list_and_index(item)
        if idx <= 0:
            return
        lst[idx], lst[idx - 1] = lst[idx - 1], lst[idx]
        parent_item = item.parent()
        if parent_item is None:
            self._tree.takeTopLevelItem(idx)
            self._tree.insertTopLevelItem(idx - 1, item)
        else:
            parent_item.takeChild(idx)
            parent_item.insertChild(idx - 1, item)
        self._tree.setCurrentItem(item)

    def _on_move_down(self) -> None:
        item = self._tree.currentItem()
        if item is None:
            return
        lst, idx = self._get_parent_list_and_index(item)
        if idx >= len(lst) - 1:
            return
        lst[idx], lst[idx + 1] = lst[idx + 1], lst[idx]
        parent_item = item.parent()
        if parent_item is None:
            self._tree.takeTopLevelItem(idx)
            self._tree.insertTopLevelItem(idx + 1, item)
        else:
            parent_item.takeChild(idx)
            parent_item.insertChild(idx + 1, item)
        self._tree.setCurrentItem(item)

    # ── Selection change ──────────────────────────────────────────

    def _on_selection_changed(self, current: QTreeWidgetItem, _previous) -> None:
        if current is None:
            self._stack.setCurrentIndex(0)
            return
        obj = current.data(0, _ROLE)
        if isinstance(obj, SchemaNode):
            self._stack.setCurrentIndex(1)
            self._node_key_edit.blockSignals(True)
            self._node_key_edit.setText(obj.key)
            self._node_key_edit.blockSignals(False)
            self._node_type_combo.blockSignals(True)
            self._node_type_combo.setCurrentText(obj.type)
            self._node_type_combo.blockSignals(False)
        else:
            self._stack.setCurrentIndex(2)
            self._ref_key_edit.blockSignals(True)
            self._ref_key_edit.setText(obj.key)
            self._ref_key_edit.blockSignals(False)
            # Populate profile combo
            self._ref_profile_combo.blockSignals(True)
            self._ref_profile_combo.clear()
            self._ref_profile_combo.addItems(sorted(self._profile_rois.keys()))
            self._ref_profile_combo.setCurrentText(obj.profile)
            self._ref_profile_combo.blockSignals(False)
            # Populate ROI combo
            self._ref_roi_combo.blockSignals(True)
            self._ref_roi_combo.clear()
            rois = self._profile_rois.get(obj.profile, [])
            self._ref_roi_combo.addItems(rois)
            self._ref_roi_combo.setCurrentText(obj.roi)
            self._ref_roi_combo.blockSignals(False)

    # ── SchemaNode edit slots ─────────────────────────────────────

    def _on_node_key_changed(self) -> None:
        item = self._tree.currentItem()
        if item is None:
            return
        obj = item.data(0, _ROLE)
        if not isinstance(obj, SchemaNode):
            return
        new_key = self._node_key_edit.text().strip()
        if new_key:
            obj.key = new_key
            self._refresh_item_text(item)

    def _on_node_type_changed(self, text: str) -> None:
        item = self._tree.currentItem()
        if item is None:
            return
        obj = item.data(0, _ROLE)
        if not isinstance(obj, SchemaNode):
            return
        obj.type = text
        self._refresh_item_text(item)

    # ── ROIRef edit slots ─────────────────────────────────────────

    def _on_ref_key_changed(self) -> None:
        item = self._tree.currentItem()
        if item is None:
            return
        obj = item.data(0, _ROLE)
        if not isinstance(obj, ROIRef):
            return
        obj.key = self._ref_key_edit.text().strip()
        self._refresh_item_text(item)

    def _on_ref_profile_changed(self, profile: str) -> None:
        item = self._tree.currentItem()
        if item is None:
            return
        obj = item.data(0, _ROLE)
        if not isinstance(obj, ROIRef):
            return
        obj.profile = profile
        rois = self._profile_rois.get(profile, [])
        self._ref_roi_combo.blockSignals(True)
        self._ref_roi_combo.clear()
        self._ref_roi_combo.addItems(rois)
        self._ref_roi_combo.blockSignals(False)
        # Set ROI to first available
        first_roi = rois[0] if rois else ""
        obj.roi = first_roi
        self._ref_roi_combo.setCurrentText(first_roi)
        self._refresh_item_text(item)

    def _on_ref_roi_changed(self, roi: str) -> None:
        item = self._tree.currentItem()
        if item is None:
            return
        obj = item.data(0, _ROLE)
        if not isinstance(obj, ROIRef):
            return
        obj.roi = roi
        self._refresh_item_text(item)

    # ── Public result ─────────────────────────────────────────────

    def get_schema(self) -> list[SchemaNode]:
        """Return the edited schema. Only meaningful after exec() == Accepted."""
        return self._schema
