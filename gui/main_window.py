"""
Main window for Rock Capture CNN.
Top-level QMainWindow with toolbar, preview, and controls panel.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from pynput import keyboard

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QToolBar, QLabel, QComboBox,
    QPushButton, QApplication, QFileDialog, QMessageBox, QSpinBox,
    QLineEdit,
)
from PyQt6.QtCore import Qt, QTimer

from core.config import AppConfig
from core.profile import Profile, HUDProfile, SchemaNode, ROIRef
from core.pipeline import RecognitionPipeline, FrameResult
from cnn.predictor import Predictor
from cnn.trainer import TrainerThread
from gui.preview_widget import PreviewWidget
from gui.controls_panel import ControlsPanel
from gui.region_selector import ScreenCaptureOverlay
from gui.training_dialog import TrainingDialog
from gui.anchor_roi_dialog import AnchorROIDialog
from gui.output_schema_dialog import OutputSchemaDialog


class MainWindow(QMainWindow):
    """Top-level application window for Rock Capture CNN."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rock Capture CNN")
        self.setGeometry(100, 100, 1400, 900)

        self._base_dir = Path(__file__).resolve().parent.parent
        self._hud_profiles_dir = self._base_dir / "data" / "hud_profiles"
        self._predictor = Predictor()
        self._pipelines: dict[str, RecognitionPipeline] = {}
        self._profiles: dict[str, Profile] = {}
        self._editing_profile_name: str | None = None
        self._running = False
        self._fps = 10
        self._last_frame_results: dict[str, FrameResult] = {}
        self._staged_data: dict | None = None
        self._captures_dir = self._base_dir.parent / "Rock Capture Database" / "captures"
        self._config_path = self._base_dir / "data" / "config.json"
        self._config = AppConfig.load(self._config_path)
        self._cluster_id: int = 1
        self._session_id = uuid.uuid4().hex[:8]
        self._session_started_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self._session_captures: list[dict] = []
        self._session_file: Path | None = None
        self._model_path: str = ""
        self._active_output_schema: list[SchemaNode] = []
        self._active_hud_name: str = ""

        self._init_capture_toolbar()
        self.addToolBarBreak(Qt.ToolBarArea.TopToolBarArea)
        self._init_settings_toolbar()
        self.addToolBarBreak(Qt.ToolBarArea.TopToolBarArea)
        self._init_hud_toolbar()
        self._init_central()
        self._init_statusbar()
        self._connect_signals()
        self._setup_hotkeys()

        self._load_all_profiles()

    # ── Row 1: capture toolbar ────────────────────────────────────

    def _init_capture_toolbar(self) -> None:
        toolbar = QToolBar("Capture")
        toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        toolbar.addWidget(QLabel(" Profile: "))
        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumWidth(120)
        self.profile_combo.currentTextChanged.connect(self._on_profile_selected)
        toolbar.addWidget(self.profile_combo)

        self.new_profile_btn = QPushButton("New")
        self.new_profile_btn.clicked.connect(self._on_new_profile)
        toolbar.addWidget(self.new_profile_btn)

        self.duplicate_profile_btn = QPushButton("Duplicate")
        self.duplicate_profile_btn.clicked.connect(self._on_duplicate_profile)
        toolbar.addWidget(self.duplicate_profile_btn)

        self.save_profile_btn = QPushButton("Save")
        self.save_profile_btn.clicked.connect(self._on_save_profile)
        toolbar.addWidget(self.save_profile_btn)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel(" Monitor: "))
        self.monitor_combo = QComboBox()
        self._populate_monitors()
        toolbar.addWidget(self.monitor_combo)

        self.set_region_btn = QPushButton("Set Search Region")
        self.set_region_btn.clicked.connect(self._on_set_search_region)
        toolbar.addWidget(self.set_region_btn)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel(" FPS: "))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 30)
        self.fps_spin.setValue(10)
        self.fps_spin.valueChanged.connect(self._on_fps_changed)
        toolbar.addWidget(self.fps_spin)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel(" Cluster: "))
        self.cluster_id_spin = QSpinBox()
        self.cluster_id_spin.setRange(1, 99999)
        self.cluster_id_spin.setValue(1)
        self.cluster_id_spin.setToolTip("Groups related captures. Press F11 to advance.")
        self.cluster_id_spin.valueChanged.connect(self._on_cluster_id_changed)
        toolbar.addWidget(self.cluster_id_spin)

        self.cluster_advance_btn = QPushButton("+ (F11)")
        self.cluster_advance_btn.setToolTip("Advance cluster ID by 1")
        self.cluster_advance_btn.clicked.connect(self._on_advance_cluster_id)
        toolbar.addWidget(self.cluster_advance_btn)

        self.start_stop_btn = QPushButton("Start All")
        self.start_stop_btn.clicked.connect(self._on_start_stop)
        toolbar.addWidget(self.start_stop_btn)

    # ── Row 2: settings toolbar ───────────────────────────────────

    def _init_settings_toolbar(self) -> None:
        toolbar = QToolBar("Settings")
        toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        toolbar.addWidget(QLabel(" Model: "))
        self.model_status_label = QLabel("No model loaded")
        self.model_status_label.setStyleSheet("color: gray;")
        self.model_status_label.setMinimumWidth(160)
        toolbar.addWidget(self.model_status_label)

        self.load_model_btn = QPushButton("Load Model...")
        self.load_model_btn.clicked.connect(self._on_load_model)
        toolbar.addWidget(self.load_model_btn)

        self.train_model_btn = QPushButton("Train")
        self.train_model_btn.clicked.connect(self._on_train)
        toolbar.addWidget(self.train_model_btn)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel(" User: "))
        self.user_edit = QLineEdit(self._config.user)
        self.user_edit.setMaximumWidth(80)
        self.user_edit.editingFinished.connect(self._on_config_changed)
        toolbar.addWidget(self.user_edit)

        toolbar.addWidget(QLabel(" Org: "))
        self.org_edit = QLineEdit(self._config.org)
        self.org_edit.setMaximumWidth(80)
        self.org_edit.editingFinished.connect(self._on_config_changed)
        toolbar.addWidget(self.org_edit)

        toolbar.addWidget(QLabel(" Version: "))
        self.version_edit = QLineEdit(self._config.tool_version)
        self.version_edit.setMaximumWidth(100)
        self.version_edit.editingFinished.connect(self._on_config_changed)
        toolbar.addWidget(self.version_edit)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel(" System: "))
        self.system_edit = QLineEdit(self._config.system)
        self.system_edit.setMaximumWidth(100)
        self.system_edit.setToolTip("Star system where scanning takes place (e.g. Pyro)")
        self.system_edit.editingFinished.connect(self._on_config_changed)
        toolbar.addWidget(self.system_edit)

        toolbar.addWidget(QLabel(" Gravity Well: "))
        self.gravity_well_edit = QLineEdit(self._config.gravity_well)
        self.gravity_well_edit.setMaximumWidth(140)
        self.gravity_well_edit.setToolTip(
            "Gravity well / location within the system.\n"
            "Use / to indicate sub-levels, e.g. Pyro4, Pyro4/Rings, Pyro4/City."
        )
        self.gravity_well_edit.editingFinished.connect(self._on_config_changed)
        toolbar.addWidget(self.gravity_well_edit)

    # ── Row 3: HUD profile toolbar ────────────────────────────────

    def _init_hud_toolbar(self) -> None:
        hud_toolbar = QToolBar("HUD Profiles")
        hud_toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, hud_toolbar)

        hud_toolbar.addWidget(QLabel(" HUD Profile: "))
        self.hud_profile_combo = QComboBox()
        self.hud_profile_combo.setMinimumWidth(140)
        hud_toolbar.addWidget(self.hud_profile_combo)

        self.save_hud_btn = QPushButton("Save HUD")
        self.save_hud_btn.setToolTip("Save current state of all profiles as a HUD profile")
        self.save_hud_btn.clicked.connect(self._on_save_hud_profile)
        hud_toolbar.addWidget(self.save_hud_btn)

        self.load_hud_btn = QPushButton("Load HUD")
        self.load_hud_btn.setToolTip("Overwrite all small profiles with the selected HUD profile")
        self.load_hud_btn.clicked.connect(self._on_load_hud_profile)
        hud_toolbar.addWidget(self.load_hud_btn)

        self.delete_hud_btn = QPushButton("Delete HUD")
        self.delete_hud_btn.setToolTip("Delete the selected HUD profile file")
        self.delete_hud_btn.clicked.connect(self._on_delete_hud_profile)
        hud_toolbar.addWidget(self.delete_hud_btn)

        hud_toolbar.addSeparator()

        self.edit_schema_btn = QPushButton("Edit Schema…")
        self.edit_schema_btn.setToolTip(
            "Edit the output_schema for the active HUD profile\n"
            "(controls how profiles are grouped in the session JSON)"
        )
        self.edit_schema_btn.clicked.connect(self._on_edit_schema)
        hud_toolbar.addWidget(self.edit_schema_btn)

    def _refresh_hud_profile_combo(self) -> None:
        names = HUDProfile.list_profiles(self._hud_profiles_dir)
        self.hud_profile_combo.blockSignals(True)
        current = self.hud_profile_combo.currentText()
        self.hud_profile_combo.clear()
        self.hud_profile_combo.addItem("(none)")
        for name in names:
            self.hud_profile_combo.addItem(name)
        # Restore previous selection if still present
        idx = self.hud_profile_combo.findText(current)
        self.hud_profile_combo.setCurrentIndex(max(0, idx))
        self.hud_profile_combo.blockSignals(False)

    def _on_save_hud_profile(self) -> None:
        from PyQt6.QtWidgets import QInputDialog
        # Sync current editing profile before snapshotting
        self._sync_profile_from_ui()

        # Suggest the currently selected HUD name as default
        default_name = self.hud_profile_combo.currentText()
        if default_name == "(none)":
            default_name = ""

        name, ok = QInputDialog.getText(
            self, "Save HUD Profile",
            "HUD profile name (will overwrite if it already exists):",
            text=default_name,
        )
        if not ok or not name.strip():
            return
        name = name.strip()

        # Snapshot every profile that is currently loaded
        hud = HUDProfile(name=name, output_schema=self._active_output_schema)
        for pname, profile in self._profiles.items():
            hud.profiles[pname] = profile.to_dict()

        hud.save(self._hud_profiles_dir)
        self._refresh_hud_profile_combo()
        self.hud_profile_combo.setCurrentText(name)
        self.statusBar().showMessage(
            f"HUD profile '{name}' saved ({len(hud.profiles)} profiles)"
        )

    def _on_load_hud_profile(self) -> None:
        name = self.hud_profile_combo.currentText()
        if name == "(none)" or not name:
            QMessageBox.warning(self, "No HUD Profile", "Select a HUD profile first.")
            return

        path = self._hud_profiles_dir / f"{name}.json"
        if not path.exists():
            QMessageBox.warning(self, "Not Found", f"HUD profile file not found:\n{path}")
            return

        reply = QMessageBox.question(
            self, "Load HUD Profile",
            f"Load HUD profile '{name}'?\n\n"
            f"This will overwrite all matching small profile files on disk and reload them.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        hud = HUDProfile.load(path)
        self._active_output_schema = hud.output_schema
        self._active_hud_name = name
        was_running = self._running
        if was_running:
            for p in self._pipelines.values():
                p.stop()
            self._running = False
            self.start_stop_btn.setText("Start All")
            self.start_stop_btn.setStyleSheet("")
            self.set_region_btn.setEnabled(True)

        profiles_dir = self._base_dir / "data" / "profiles"
        profiles_dir.mkdir(parents=True, exist_ok=True)

        for pname, pdata in hud.profiles.items():
            # Write each small profile back to disk
            out_path = profiles_dir / f"{pname}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(pdata, f, indent=2)

        # Reload everything from disk
        self._load_all_profiles()

        # Restore selection in HUD combo (load_all_profiles resets it)
        self.hud_profile_combo.setCurrentText(name)

        self.statusBar().showMessage(
            f"Loaded HUD profile '{name}' — {len(hud.profiles)} profiles applied"
        )

        if was_running:
            started = self._start_all_pipelines()
            if started:
                self._running = True
                self.start_stop_btn.setText("Stop All")
                self.start_stop_btn.setStyleSheet("background-color: #ff6666;")
                self.set_region_btn.setEnabled(False)

    def _on_delete_hud_profile(self) -> None:
        name = self.hud_profile_combo.currentText()
        if name == "(none)" or not name:
            QMessageBox.warning(self, "No HUD Profile", "Select a HUD profile to delete.")
            return

        reply = QMessageBox.question(
            self, "Delete HUD Profile",
            f"Delete HUD profile '{name}'?\nThis only removes the HUD file; small profiles are not affected.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        path = self._hud_profiles_dir / f"{name}.json"
        if path.exists():
            path.unlink()
        self._refresh_hud_profile_combo()
        self.statusBar().showMessage(f"HUD profile '{name}' deleted")

    def _on_edit_schema(self) -> None:
        """Open the output-schema editor for the active HUD profile."""
        profile_rois = {
            name: [roi.name for roi in profile.rois]
            for name, profile in self._profiles.items()
        }
        dlg = OutputSchemaDialog(self._active_output_schema, profile_rois, parent=self)
        if dlg.exec() != OutputSchemaDialog.DialogCode.Accepted:
            return

        new_schema = dlg.get_schema()
        self._active_output_schema = new_schema

        # Persist immediately if there is an active HUD profile on disk
        if self._active_hud_name:
            path = self._hud_profiles_dir / f"{self._active_hud_name}.json"
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    data["output_schema"] = [n.to_dict() for n in new_schema]
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    self.statusBar().showMessage(
                        f"Output schema saved to HUD profile '{self._active_hud_name}'"
                    )
                except Exception as e:
                    self.statusBar().showMessage(f"Schema save error: {e}")
                return

        self.statusBar().showMessage(
            "Output schema updated (no HUD profile loaded — changes apply only to this session)"
        )

    # ── Central Layout ───────────────────────────────────────────

    def _init_central(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.preview_widget = PreviewWidget()
        splitter.addWidget(self.preview_widget)

        self.controls_panel = ControlsPanel()
        splitter.addWidget(self.controls_panel)

        splitter.setSizes([980, 420])
        self.setCentralWidget(splitter)

    # ── Helper properties ────────────────────────────────────────

    @property
    def _editing_profile(self) -> Profile | None:
        if self._editing_profile_name:
            return self._profiles.get(self._editing_profile_name)
        return None

    @property
    def _editing_pipeline(self) -> RecognitionPipeline | None:
        if self._editing_profile_name:
            return self._pipelines.get(self._editing_profile_name)
        return None

    # ── Status Bar ───────────────────────────────────────────────

    def _init_statusbar(self) -> None:
        self.statusBar().showMessage("Ready - Set search region and load a profile to begin")
        self.anchor_status_label = QLabel("Anchor: --")
        self.fps_label = QLabel("FPS: --")
        self.statusBar().addPermanentWidget(self.anchor_status_label)
        self.statusBar().addPermanentWidget(self.fps_label)

        self._frame_count = 0
        self._fps_timer = QTimer(self)
        self._fps_timer.timeout.connect(self._update_fps)
        self._fps_timer.start(1000)

    def _update_fps(self) -> None:
        if self._running:
            self.fps_label.setText(f"FPS: {self._frame_count}")
        self._frame_count = 0

    # ── Signal Connections ───────────────────────────────────────

    def _connect_signals(self) -> None:
        # Controls -> MainWindow
        self.controls_panel.anchor_browse_clicked.connect(self._on_anchor_browse)
        self.controls_panel.anchor_roi_clicked.connect(self._on_set_anchor_roi)
        self.controls_panel.anchor_roi_clear_clicked.connect(self._on_clear_anchor_roi)
        self.controls_panel.anchor_threshold_changed.connect(self._on_anchor_threshold)
        self.controls_panel.filters_changed.connect(self._on_filters_changed)
        self.controls_panel.rois_changed.connect(self._on_rois_changed)
        self.controls_panel.labeler_toggled.connect(self._on_labeler_toggle)

    # ── Frame Processing ─────────────────────────────────────────

    def _on_frame_result(self, result: FrameResult) -> None:
        self._frame_count += 1
        self._last_frame_results[result.profile_name] = result

        # Show the edited profile's frame in the main preview
        if result.profile_name == self._editing_profile_name:
            self.preview_widget.update_main_preview(result.annotated_frame)
            self.anchor_status_label.setText(
                f"Anchor [{result.profile_name}]: ({result.anchor.x}, {result.anchor.y}) "
                f"conf={result.anchor.confidence:.2f}"
            )
            self.controls_panel.update_anchor_status(
                f"Found at ({result.anchor.x}, {result.anchor.y}) "
                f"conf={result.anchor.confidence:.2f}",
                found=True,
            )

        # ROI cards and results — qualified by profile name
        self.preview_widget.update_roi_previews(
            result.roi_results, profile_name=result.profile_name
        )
        self.controls_panel.update_results(
            result.roi_results, profile_name=result.profile_name
        )

        if self.controls_panel.is_labeler_active():
            for roi_result in result.roi_results:
                self.controls_panel.labeler_widget.queue_characters(
                    roi_result.characters, roi_result.name
                )

    def _on_anchor_lost(self, profile_name: str) -> None:
        self._last_frame_results.pop(profile_name, None)
        if profile_name == self._editing_profile_name:
            self.anchor_status_label.setText(f"Anchor [{profile_name}]: NOT FOUND")
            self.controls_panel.update_anchor_status("Not found", found=False)

    # ── Profile Management ───────────────────────────────────────

    def _load_all_profiles(self) -> None:
        """Load all profiles from disk and create a pipeline for each."""
        for p in self._pipelines.values():
            p.stop()
        self._pipelines.clear()
        self._profiles.clear()

        profiles_dir = self._base_dir / "data" / "profiles"
        names = Profile.list_profiles(profiles_dir)

        for name in names:
            path = profiles_dir / f"{name}.json"
            profile = Profile.load(path)
            self._profiles[name] = profile
            self._create_pipeline_for(name, profile)

        # Auto-load the first available model from data/models/
        if not self._predictor.is_loaded:
            models_dir = self._base_dir / "data" / "models"
            for pth in sorted(models_dir.glob("*.pth")):
                if self._predictor.load_model(str(pth)):
                    self._model_path = pth.name
                    self._update_model_status()
                    break

        self._refresh_profile_combo(names)
        self._refresh_hud_profile_combo()

    def _create_pipeline_for(self, name: str, profile: Profile) -> None:
        """Create and wire a RecognitionPipeline for a single profile."""
        pipeline = RecognitionPipeline(
            parent=self, predictor=self._predictor, profile_name=name,
        )
        pipeline.load_profile(profile, self._base_dir)
        pipeline.frame_processed.connect(self._on_frame_result)
        pipeline.anchor_lost.connect(lambda _n=name: self._on_anchor_lost(_n))
        self._pipelines[name] = pipeline

    def _refresh_profile_combo(self, names: list[str] | None = None) -> None:
        if names is None:
            names = sorted(self._profiles.keys())
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItem("(none)")
        for name in names:
            self.profile_combo.addItem(name)
        self.profile_combo.blockSignals(False)

    def _on_profile_selected(self, name: str) -> None:
        if name == "(none)" or not name:
            self._editing_profile_name = None
            return
        self._editing_profile_name = name
        profile = self._profiles.get(name)
        if profile:
            self._apply_profile_to_ui(profile)

    def _apply_profile_to_ui(self, profile: Profile) -> None:
        """Sync right-panel UI widgets to reflect the given profile's settings."""
        self.controls_panel.roi_editor.load_rois(profile.rois)

        if profile.anchor_template_path:
            self.controls_panel.anchor_path_label.setText(profile.anchor_template_path)
            self.controls_panel.anchor_path_label.setStyleSheet(
                "color: green; font-size: 10px;"
            )
        else:
            self.controls_panel.anchor_path_label.setText("No template loaded")
            self.controls_panel.anchor_path_label.setStyleSheet(
                "color: gray; font-size: 10px;"
            )

        self.controls_panel.anchor_threshold_slider.setValue(
            int(profile.anchor_match_threshold * 100)
        )

        self.monitor_combo.setCurrentIndex(
            min(profile.monitor_index, self.monitor_combo.count() - 1)
        )
        self._update_anchor_roi_label(profile.anchor_roi)
        self.statusBar().showMessage(f"Editing profile '{profile.name}'")

    def _on_new_profile(self) -> None:
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "New Profile", "Profile name:")
        if ok and name:
            profile = Profile(name=name)
            self._profiles[name] = profile
            self._create_pipeline_for(name, profile)
            self._refresh_profile_combo()
            self.profile_combo.setCurrentText(name)

    def _on_duplicate_profile(self) -> None:
        from PyQt6.QtWidgets import QInputDialog
        source = self._editing_profile
        if not source:
            QMessageBox.warning(self, "No Profile", "Select a profile to duplicate first.")
            return
        self._sync_profile_from_ui()  # make sure edits are captured
        name, ok = QInputDialog.getText(
            self, "Duplicate Profile",
            f"New name (copying '{source.name}'):",
            text=f"{source.name}_copy",
        )
        if not ok or not name:
            return
        if name in self._profiles:
            QMessageBox.warning(self, "Name Taken", f"A profile named '{name}' already exists.")
            return
        # Deep-copy via dict round-trip, then rename
        copy_data = source.to_dict()
        copy_data["name"] = name
        new_profile = Profile.from_dict(copy_data)
        self._profiles[name] = new_profile
        self._create_pipeline_for(name, new_profile)
        self._refresh_profile_combo()
        self.profile_combo.setCurrentText(name)

    def _on_save_profile(self) -> None:
        profile = self._editing_profile
        if not profile:
            QMessageBox.warning(self, "No Profile", "Create or load a profile first.")
            return
        self._sync_profile_from_ui()
        profiles_dir = self._base_dir / "data" / "profiles"
        profile.save(profiles_dir)
        self.statusBar().showMessage(f"Profile '{profile.name}' saved")

    def _sync_profile_from_ui(self) -> None:
        """Sync current UI state back into the editing profile object."""
        profile = self._editing_profile
        if not profile:
            return
        profile.rois = self.controls_panel.roi_editor.rois
        profile.monitor_index = self.monitor_combo.currentIndex()
        profile.anchor_match_threshold = (
            self.controls_panel.anchor_threshold_slider.value() / 100.0
        )

    # ── Anchor ───────────────────────────────────────────────────

    def _on_anchor_browse(self) -> None:
        anchors_dir = self._base_dir / "data" / "anchors"
        anchors_dir.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Anchor Template", str(anchors_dir),
            "Images (*.png *.jpg *.bmp)"
        )
        if path:
            rel = Path(path).name
            # Copy to anchors dir if not already there
            dest = anchors_dir / rel
            if str(Path(path).parent.resolve()) != str(anchors_dir.resolve()):
                import shutil
                shutil.copy2(path, dest)

            profile = self._editing_profile
            pipeline = self._editing_pipeline
            if profile and pipeline:
                profile.anchor_template_path = rel
                pipeline.anchor_matcher.load_template(str(dest))
                self.controls_panel.anchor_path_label.setText(rel)
                self.controls_panel.anchor_path_label.setStyleSheet(
                    "color: green; font-size: 10px;"
                )

    def _on_anchor_threshold(self, value: float) -> None:
        pipeline = self._editing_pipeline
        if pipeline:
            pipeline.anchor_matcher.threshold = value
        profile = self._editing_profile
        if profile:
            profile.anchor_match_threshold = value

    def _on_set_anchor_roi(self) -> None:
        """Open dialog to draw an anchor search ROI on the last captured frame."""
        pipeline = self._editing_pipeline
        if not pipeline:
            QMessageBox.warning(self, "No Profile", "Select a profile first.")
            return

        frame = pipeline.capture_engine.grab_single_frame()
        if frame is None:
            sr = self._editing_profile.search_region if self._editing_profile else {}
            if sr.get("w", 0) > 0:
                QMessageBox.information(
                    self, "No Frame",
                    "Start capture first (or click Start All) so a frame is available.",
                )
            else:
                QMessageBox.warning(self, "No Region", "Set a search region first.")
            return

        current_roi = self._editing_profile.anchor_roi if self._editing_profile else {}
        dialog = AnchorROIDialog(frame, current_roi, self)
        if dialog.exec():
            roi = dialog.get_roi_dict()
            if roi and self._editing_profile:
                self._editing_profile.anchor_roi = roi
                pipeline._profile = self._editing_profile
                self._update_anchor_roi_label(roi)

    def _on_clear_anchor_roi(self) -> None:
        """Remove the anchor ROI constraint - search the full captured frame."""
        profile = self._editing_profile
        pipeline = self._editing_pipeline
        if profile:
            profile.anchor_roi = {}
        if pipeline and profile:
            pipeline._profile = profile
        self._update_anchor_roi_label({})

    def _update_anchor_roi_label(self, roi: dict) -> None:
        if roi and all(k in roi for k in ("x", "y", "w", "h")):
            text = f"Anchor ROI: {roi['w']}x{roi['h']} at ({roi['x']}, {roi['y']})"
            self.controls_panel.anchor_roi_label.setText(text)
            self.controls_panel.anchor_roi_label.setStyleSheet(
                "color: #b060ff; font-size: 10px;"
            )
        else:
            self.controls_panel.anchor_roi_label.setText("Anchor ROI: full frame")
            self.controls_panel.anchor_roi_label.setStyleSheet(
                "color: gray; font-size: 10px;"
            )

    # ── Search Region ────────────────────────────────────────────

    def _on_set_search_region(self) -> None:
        pipeline = self._editing_pipeline
        profile = self._editing_profile
        if not pipeline or not profile:
            QMessageBox.warning(self, "No Profile", "Select a profile first.")
            return

        was_running = self._running
        if was_running:
            for p in self._pipelines.values():
                p.stop()
            self._running = False

        screen_index = self.monitor_combo.currentIndex()
        screens = QApplication.screens()
        if screen_index >= len(screens):
            return
        screen = screens[screen_index]

        overlay = ScreenCaptureOverlay(screen.geometry())
        overlay.showFullScreen()
        while overlay.isVisible():
            QApplication.processEvents()

        rect = overlay.get_selected_rect()
        if rect:
            pipeline.capture_engine.set_search_region(rect, screen_index)
            profile.search_region = {
                "x": rect.x(), "y": rect.y(),
                "w": rect.width(), "h": rect.height(),
            }
            profile.monitor_index = screen_index
            self.statusBar().showMessage(
                f"Search region for '{profile.name}': "
                f"{rect.width()}x{rect.height()} at ({rect.x()}, {rect.y()})"
            )
            if was_running:
                self._start_all_pipelines()

    # ── Start/Stop ───────────────────────────────────────────────

    def _on_start_stop(self) -> None:
        if self._running:
            for p in self._pipelines.values():
                p.stop()
                p.reload_templates()
            self._running = False
            self.start_stop_btn.setText("Start All")
            self.start_stop_btn.setStyleSheet("")
            self.set_region_btn.setEnabled(True)
            self.statusBar().showMessage("Capture stopped")
        else:
            started = self._start_all_pipelines()
            if started == 0:
                QMessageBox.warning(
                    self, "Cannot Start",
                    "No profiles with a valid anchor and search region found.\n"
                    "Load at least one profile with an anchor template and search region.",
                )
                return
            self._running = True
            self.start_stop_btn.setText("Stop All")
            self.start_stop_btn.setStyleSheet("background-color: #ff6666;")
            self.set_region_btn.setEnabled(False)
            self.statusBar().showMessage(
                f"Capturing ({started} profile{'s' if started != 1 else ''})..."
            )

    def _start_all_pipelines(self) -> int:
        """Start every pipeline that has a valid anchor and search region. Returns count started."""
        started = 0
        for name, pipeline in self._pipelines.items():
            profile = self._profiles[name]
            sr = profile.search_region
            has_region = sr.get("w", 0) > 0 and sr.get("h", 0) > 0
            if pipeline.anchor_matcher.is_loaded and has_region:
                pipeline.reload_templates()
                pipeline.start(self._fps)
                started += 1
        if started > 0:
            self._running = True
        return started

    def _on_fps_changed(self, value: int) -> None:
        self._fps = value
        if self._running:
            for p in self._pipelines.values():
                if p.is_running:
                    p.stop()
                    p.start(self._fps)

    # ── Filters & ROIs ───────────────────────────────────────────

    def _on_filters_changed(self, settings) -> None:
        profile = self._editing_profile
        if profile:
            profile.rois = self.controls_panel.roi_editor.rois

    def _on_rois_changed(self) -> None:
        profile = self._editing_profile
        pipeline = self._editing_pipeline
        if profile:
            profile.rois = self.controls_panel.roi_editor.rois
            if pipeline:
                pipeline._profile = profile
            # Remove stale ROI cards for this profile
            prefix = f"{self._editing_profile_name}/"
            active_keys = {f"{self._editing_profile_name}/{roi.name}" for roi in profile.rois}
            for key in list(self.preview_widget._roi_cards):
                if key.startswith(prefix) and key not in active_keys:
                    self.preview_widget.remove_card(key)
            # Also clean stale result labels
            self.controls_panel.remove_stale_results(prefix, active_keys)

    # ── CNN Model ────────────────────────────────────────────────

    def _update_model_status(self) -> None:
        if self._model_path:
            self.model_status_label.setText(self._model_path)
            self.model_status_label.setStyleSheet("color: green;")
        else:
            self.model_status_label.setText("No model loaded")
            self.model_status_label.setStyleSheet("color: gray;")

    def _on_train(self) -> None:
        data_dir = self._base_dir / "data" / "training_data"
        if not data_dir.exists() or not any(data_dir.iterdir()):
            QMessageBox.warning(
                self, "No Data",
                "No training data found. Use the labeler to collect samples first."
            )
            return

        models_dir = self._base_dir / "data" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_name = "model.pth"
        model_path = models_dir / model_name

        char_classes = self._predictor.char_classes if self._predictor.is_loaded else "0123456789.-%"
        trainer = TrainerThread(
            data_dir=str(data_dir),
            output_model_path=str(model_path),
            char_classes=char_classes,
            parent=self,
        )

        dialog = TrainingDialog(trainer, self)
        if dialog.exec():
            if self._predictor.load_model(str(model_path)):
                self._model_path = model_name
                self._update_model_status()

    def _on_load_model(self) -> None:
        models_dir = self._base_dir / "data" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(
            self, "Load CNN Model", str(models_dir), "PyTorch Models (*.pth)"
        )
        if path:
            if self._predictor.load_model(path):
                self._model_path = Path(path).name
                self._update_model_status()
            else:
                QMessageBox.warning(self, "Load Failed", "Failed to load model.")

    # ── Labeler ──────────────────────────────────────────────────

    def _on_labeler_toggle(self, active: bool) -> None:
        for pipeline in self._pipelines.values():
            pipeline.labeler_mode = active
        data_dir = self._base_dir / "data" / "training_data"
        self.controls_panel.labeler_widget.set_data_dir(data_dir)
        if active:
            self.controls_panel.labeler_widget.refresh_counts()

    # ── Monitor ──────────────────────────────────────────────────

    def _populate_monitors(self) -> None:
        screens = QApplication.screens()
        for i, screen in enumerate(screens):
            g = screen.geometry()
            self.monitor_combo.addItem(
                f"Monitor {i + 1} ({g.width()}x{g.height()})", screen
            )

    # ── Hotkeys ──────────────────────────────────────────────────

    def _setup_hotkeys(self) -> None:
        self._hotkey_listener = keyboard.Listener(
            on_press=self._on_global_key
        )
        self._hotkey_listener.daemon = True
        self._hotkey_listener.start()

    def _on_global_key(self, key) -> None:
        try:
            if key == keyboard.Key.f11:
                QTimer.singleShot(0, self._on_advance_cluster_id)
            elif key == keyboard.Key.f9:
                QTimer.singleShot(0, self._on_stage_pressed)
            elif key == keyboard.Key.f10:
                QTimer.singleShot(0, self._on_commit_to_json)
        except Exception:
            pass

    def _on_stage_pressed(self) -> None:
        if not self._last_frame_results:
            self.statusBar().showMessage("Nothing to stage — no frame captured yet")
            return

        # Build per-profile ROI dicts, sorted by csv_index within each profile
        raw: dict[str, dict] = {}  # profile_name -> {roi_name: {value, confidence}}
        for prof_name, frame_result in self._last_frame_results.items():
            profile = self._profiles.get(prof_name)
            csv_idx_map: dict[str, int] = {}
            if profile:
                csv_idx_map = {roi.name: roi.csv_index for roi in profile.rois}

            roi_entries: list[tuple[int, str, dict]] = []
            for r in frame_result.roi_results:
                idx = csv_idx_map.get(r.name, 0)
                roi_entries.append((idx, r.name, {
                    "value": r.recognized_text,
                    "confidence": round(r.confidence, 4),
                }))
            roi_entries.sort(key=lambda t: (t[0] == 0, t[0]))
            raw[prof_name] = {name: val for _, name, val in roi_entries}

        def _eval_node(node: SchemaNode):
            if node.type == "array":
                result = []
                for child in node.children:
                    if isinstance(child, ROIRef):
                        result.append(
                            raw.get(child.profile, {}).get(
                                child.roi, {"value": "", "confidence": 0.0}
                            )
                        )
                    else:
                        result.append(_eval_node(child))
                return result
            else:  # object
                result = {}
                for child in node.children:
                    if isinstance(child, ROIRef):
                        out_key = child.key or child.roi
                        result[out_key] = raw.get(child.profile, {}).get(
                            child.roi, {"value": "", "confidence": 0.0}
                        )
                    else:
                        result[child.key] = _eval_node(child)
                return result

        # Apply output schema if one is active, otherwise fall back to flat "values" dict
        if self._active_output_schema:
            structured = {node.key: _eval_node(node) for node in self._active_output_schema}
            self._staged_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "_structured": True,
                **structured,
            }
        else:
            # Legacy flat format
            profile_groups: list[tuple[int, str, dict]] = []
            for prof_name, rois_dict in raw.items():
                min_idx = min(
                    (csv_idx_map.get(r, 0) for r in rois_dict),
                    default=0,
                )
                profile_groups.append((min_idx, prof_name, rois_dict))
            profile_groups.sort(key=lambda t: (t[0] == 0, t[0]))
            values = {name: rois for _, name, rois in profile_groups}
            self._staged_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "_structured": False,
                "values": values,
            }

        # Status bar preview
        parts: list[str] = []
        for key, val in self._staged_data.items():
            if key in ("timestamp", "_structured"):
                continue
            if isinstance(val, list):
                for item in val:
                    for roi_name, v in item.items():
                        parts.append(f"{roi_name}={v['value']}({v['confidence']:.2f})")
            elif isinstance(val, dict):
                for sub_k, sub_v in val.items():
                    if isinstance(sub_v, dict) and "value" in sub_v:
                        parts.append(f"{sub_k}={sub_v['value']}({sub_v['confidence']:.2f})")
                    elif isinstance(sub_v, dict):
                        for roi_name, v in sub_v.items():
                            parts.append(f"{roi_name}={v['value']}({v['confidence']:.2f})")
        self.statusBar().showMessage(f"Staged (F10 to commit): {', '.join(parts)}")

    def _on_commit_to_json(self) -> None:
        if self._staged_data is None:
            self.statusBar().showMessage("Nothing staged — press F9 first")
            return

        capture_id = uuid.uuid4().hex[:8]
        is_structured = self._staged_data.get("_structured", False)
        if is_structured:
            payload = {k: v for k, v in self._staged_data.items()
                       if k not in ("timestamp", "_structured")}
        else:
            payload = {"values": self._staged_data["values"]}
        location: dict = {}
        if self._config.system:
            location["system"] = self._config.system
        if self._config.gravity_well:
            location["gravity_well"] = self._config.gravity_well
            parts = self._config.gravity_well.split("/", 1)
            location["gravity_well_root"] = parts[0].strip()
        capture = {
            "timestamp": self._staged_data["timestamp"],
            "capture_id": capture_id,
            "cluster_id": self._cluster_id,
            **({"location": location} if location else {}),
            **payload,
        }
        self._session_captures.append(capture)

        session_data = {
            "session_id": self._session_id,
            "started_at": self._session_started_at,
            "tool_version": self._config.tool_version,
            "source": {
                "user": self._config.user,
                "org": self._config.org,
            },
            "captures": self._session_captures,
        }

        if self._session_file is None:
            safe_ts = self._session_started_at.replace(":", "").replace("-", "").replace("T", "_")
            filename = f"session_{safe_ts}_{self._session_id}.json"
            self._captures_dir.mkdir(parents=True, exist_ok=True)
            self._session_file = self._captures_dir / filename

        try:
            with open(self._session_file, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2)
            self._staged_data = None
            count = len(self._session_captures)
            self.statusBar().showMessage(
                f"Committed capture #{count} (cluster {self._cluster_id}) → {self._session_file.name}"
            )
        except Exception as e:
            self._session_captures.pop()
            self.statusBar().showMessage(f"JSON write error: {e}")

    def _on_cluster_id_changed(self, value: int) -> None:
        self._cluster_id = value

    def _on_advance_cluster_id(self) -> None:
        self._cluster_id += 1
        self.cluster_id_spin.setValue(self._cluster_id)
        self.statusBar().showMessage(f"Cluster ID → {self._cluster_id}")

    def _on_config_changed(self) -> None:
        self._config.user = self.user_edit.text().strip()
        self._config.org = self.org_edit.text().strip()
        self._config.tool_version = self.version_edit.text().strip()
        self._config.system = self.system_edit.text().strip()
        self._config.gravity_well = self.gravity_well_edit.text().strip()
        self._config.save(self._config_path)

    # ── Cleanup ──────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        for p in self._pipelines.values():
            p.stop()
        if hasattr(self, "_hotkey_listener"):
            self._hotkey_listener.stop()
        event.accept()
