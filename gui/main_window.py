"""
Main window for Rock Capture CNN.
Top-level QMainWindow with toolbar, preview, and controls panel.
"""

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from pynput import keyboard

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QToolBar, QLabel, QComboBox,
    QPushButton, QApplication, QFileDialog, QMessageBox,
    QLineEdit, QSpinBox,
)
from PyQt6.QtCore import Qt, QTimer

from core.config import AppConfig
from core.profile import Profile, HUDProfile, SchemaNode, ROIRef
from core.pipeline import RecognitionPipeline, FrameResult
from cnn.predictor import Predictor
from cnn.trainer import TrainerThread
from word_cnn.predictor import WordPredictor
from word_cnn.trainer import WordTrainerThread
from word_cnn.seed import seed_from_templates
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
        self._word_predictor = WordPredictor()
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
        self._cluster_id: str = str(uuid.uuid4())
        self._cluster_history: list[str] = [self._cluster_id]
        self._cluster_history_index: int = 0
        self._session_id = str(uuid.uuid4())
        self._session_started_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self._session_captures: list[dict] = []
        self._session_file: Path | None = None
        self._model_path: str = ""
        self._word_model_path: str = ""
        self._active_output_schema: list[SchemaNode] = []
        self._active_hud_name: str = ""
        self._tolerance_percentage: float = 0.1

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
        self.cluster_prev_btn = QPushButton("< Prev")
        self.cluster_prev_btn.setToolTip("Go back to previous cluster ID")
        self.cluster_prev_btn.clicked.connect(self._on_prev_cluster_id)
        toolbar.addWidget(self.cluster_prev_btn)

        self.cluster_id_label = QLabel(self._cluster_id[:8] + "…")
        self.cluster_id_label.setToolTip(self._cluster_id)
        self.cluster_id_label.setMinimumWidth(80)
        self.cluster_id_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        toolbar.addWidget(self.cluster_id_label)

        self.cluster_advance_btn = QPushButton("Next > (F11)")
        self.cluster_advance_btn.setToolTip("Advance to a new cluster ID")
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

        toolbar.addWidget(QLabel(" Word Model: "))
        self.word_model_status_label = QLabel("No word model")
        self.word_model_status_label.setStyleSheet("color: gray;")
        self.word_model_status_label.setMinimumWidth(120)
        toolbar.addWidget(self.word_model_status_label)

        self.load_word_model_btn = QPushButton("Load Word Model...")
        self.load_word_model_btn.clicked.connect(self._on_load_word_model)
        toolbar.addWidget(self.load_word_model_btn)

        self.seed_templates_btn = QPushButton("Seed Templates")
        self.seed_templates_btn.clicked.connect(self._on_seed_templates)
        toolbar.addWidget(self.seed_templates_btn)

        self.train_word_model_btn = QPushButton("Train Words")
        self.train_word_model_btn.clicked.connect(self._on_train_words)
        toolbar.addWidget(self.train_word_model_btn)

        self.debug_word_btn = QPushButton("Debug ROI: OFF")
        self.debug_word_btn.setCheckable(True)
        self.debug_word_btn.clicked.connect(self._on_debug_word_toggle)
        toolbar.addWidget(self.debug_word_btn)

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

        toolbar.addWidget(QLabel(" Location: "))
        location_str = self._config.system
        if self._config.gravity_well:
            location_str += "/" + self._config.gravity_well if location_str else self._config.gravity_well
        self.location_edit = QLineEdit(location_str)
        self.location_edit.setMaximumWidth(250)
        self.location_edit.setToolTip(
            "Location path: System/GravityWell/Region/Place\n"
            "e.g. Pyro/Bloom, Stanton/Hurston/Aberdeen, Pyro/Bloom/Outpost Alpha"
        )
        self.location_edit.editingFinished.connect(self._on_config_changed)
        toolbar.addWidget(self.location_edit)

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
        self.controls_panel.word_labeler_toggled.connect(self._on_word_labeler_toggle)

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

        if self.controls_panel.word_labeler_widget.is_active():
            for roi_result in result.roi_results:
                # Only queue ROIs that use template or word_cnn mode
                if not roi_result.characters and roi_result.raw_image is not None:
                    self.controls_panel.word_labeler_widget.queue_image(
                        roi_result.raw_image, roi_result.name
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

        # Auto-load word model
        if not self._word_predictor.is_loaded:
            word_model_path = self._base_dir / "data" / "models" / "word_model.pth"
            if word_model_path.exists():
                if self._word_predictor.load_model(str(word_model_path)):
                    self._word_model_path = word_model_path.name
                    self._update_word_model_status()

        self._refresh_profile_combo(names)
        self._refresh_hud_profile_combo()

    def _create_pipeline_for(self, name: str, profile: Profile) -> None:
        """Create and wire a RecognitionPipeline for a single profile."""
        pipeline = RecognitionPipeline(
            parent=self, predictor=self._predictor,
            word_predictor=self._word_predictor, profile_name=name,
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
        default_dir = self._base_dir / "data" / "training_data"
        chosen = QFileDialog.getExistingDirectory(
            self, "Select Training Data Directory", str(default_dir)
        )
        if not chosen:
            return
        data_dir = Path(chosen)
        if not data_dir.exists() or not any(data_dir.iterdir()):
            QMessageBox.warning(
                self, "No Data",
                f"No training data found in:\n{data_dir}\n\nUse the labeler to collect samples first."
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

    # ── Word CNN ─────────────────────────────────────────────────

    def _update_word_model_status(self) -> None:
        if self._word_predictor.is_loaded:
            classes = self._word_predictor.word_classes
            self.word_model_status_label.setText(
                f"{self._word_model_path} ({len(classes)} classes)"
            )
            self.word_model_status_label.setStyleSheet("color: green;")
        else:
            self.word_model_status_label.setText("No word model")
            self.word_model_status_label.setStyleSheet("color: gray;")

    def _on_debug_word_toggle(self, checked: bool) -> None:
        debug_dir = self._base_dir / "debug_word_rois" if checked else None
        self._word_predictor.debug_dir = debug_dir
        self.debug_word_btn.setText(f"Debug ROI: {'ON' if checked else 'OFF'}")
        if checked:
            self._word_predictor._debug_counter = 0
            QMessageBox.information(
                self, "Debug ON",
                f"Live ROI images will be saved to:\n{debug_dir}\n\n"
                "Each frame saves the preprocessed canvas so you can compare\n"
                "with debug_preprocessed/ (training inputs).\n\n"
                "Click again to stop.",
            )

    def _on_seed_templates(self) -> None:
        """Copy template images into word_training_data class folders."""
        template_dirs = [
            self._base_dir / "data" / "templates" / "resources",
            self._base_dir / "data" / "templates" / "deposits",
        ]
        output_dir = self._base_dir / "data" / "word_training_data"
        counts = seed_from_templates(template_dirs, output_dir)
        if counts:
            summary = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
            QMessageBox.information(
                self, "Seed Complete",
                f"Copied templates into word_training_data/\n\n"
                f"{len(counts)} classes: {summary}",
            )
            # Refresh word labeler counts if visible
            if self.controls_panel.word_labeler_widget.isVisible():
                self.controls_panel.word_labeler_widget.refresh_counts()
        else:
            QMessageBox.warning(
                self, "No Templates",
                "No template images found in data/templates/.",
            )

    def _on_train_words(self) -> None:
        data_dir = self._base_dir / "data" / "word_training_data"
        if not data_dir.exists() or not any(
            d.is_dir() and (
                any(d.glob("*.png")) or any(d.glob("*.jpg"))
            )
            for d in data_dir.iterdir()
        ):
            QMessageBox.warning(
                self, "No Data",
                "No word training data found.\n\n"
                "Click 'Seed Templates' to populate from template images,\n"
                "or use the Word Labeler to collect samples manually.\n"
                "Data is stored in data/word_training_data/<label>/",
            )
            return

        models_dir = self._base_dir / "data" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_name = "word_model.pth"
        model_path = models_dir / model_name

        existing_classes = (
            self._word_predictor.word_classes
            if self._word_predictor.is_loaded else None
        )
        trainer = WordTrainerThread(
            data_dir=str(data_dir),
            output_model_path=str(model_path),
            word_classes=existing_classes,
            parent=self,
        )

        dialog = TrainingDialog(trainer, self)
        if dialog.exec():
            if self._word_predictor.load_model(str(model_path)):
                self._word_model_path = model_name
                self._update_word_model_status()

    def _on_load_word_model(self) -> None:
        models_dir = self._base_dir / "data" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Word CNN Model", str(models_dir), "PyTorch Models (*.pth)"
        )
        if path:
            if self._word_predictor.load_model(path):
                self._word_model_path = Path(path).name
                self._update_word_model_status()
            else:
                QMessageBox.warning(self, "Load Failed", "Failed to load word model.")

    # ── Labeler ──────────────────────────────────────────────────

    def _on_labeler_toggle(self, active: bool) -> None:
        for pipeline in self._pipelines.values():
            pipeline.labeler_mode = active
        data_dir = self._base_dir / "data" / "training_data"
        self.controls_panel.labeler_widget.set_data_dir(data_dir)
        if active:
            self.controls_panel.labeler_widget.refresh_counts()

    def _on_word_labeler_toggle(self, active: bool) -> None:
        data_dir = self._base_dir / "data" / "word_training_data"
        self.controls_panel.word_labeler_widget.set_data_dir(data_dir)
        if active:
            # Pre-populate combo with existing template names
            templates_dir = self._base_dir / "data" / "templates" / "resources"
            if templates_dir.exists():
                for img in sorted(templates_dir.glob("*")):
                    if img.is_file():
                        name = img.stem
                        combo = self.controls_panel.word_labeler_widget.label_combo
                        if combo.findText(name) < 0:
                            combo.addItem(name)
            self.controls_panel.word_labeler_widget.refresh_counts()

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

    def _compute_amount_red_keys(self, flat_values: dict[str, str]) -> set[str]:
        """Return flat keys for amount fields when named-material percentages don't sum to ~100."""
        named_keys: list[tuple[str, str]] = []
        total = 0.0

        if any(k.startswith("scan/composition") for k in flat_values):
            # Structured output-schema format: scan/composition[N]/name, amount_int, amount_dec
            for i in range(6):
                name = flat_values.get(f"scan/composition[{i}]/name", "").strip()
                if not name:
                    continue
                int_key = f"scan/composition[{i}]/amount_int"
                dec_key = f"scan/composition[{i}]/amount_dec"
                int_str = flat_values.get(int_key, "") or "0"
                dec_str = flat_values.get(dec_key, "") or "0"
                try:
                    total += float(f"{int_str}.{dec_str}")
                    named_keys.append((int_key, dec_key))
                except ValueError:
                    pass
        else:
            # Legacy flat format: values/volume/matNname, values/materialN/int, etc.
            for n in range(1, 7):
                name = flat_values.get(f"values/volume/mat{n}name", "").strip()
                if not name:
                    continue
                int_key = f"values/material{n}/int"
                dec_key = f"values/material{n}/decimal"
                int_str = flat_values.get(int_key, "") or "0"
                dec_str = flat_values.get(dec_key, "") or "0"
                try:
                    total += float(f"{int_str}.{dec_str}")
                    named_keys.append((int_key, dec_key))
                except ValueError:
                    pass

        if named_keys:
            # Flag red if total is outside 100% ± tolerance (both above and below)
            if total < (100.0 - self._tolerance_percentage) or total > (100.0 + self._tolerance_percentage):
                return {k for pair in named_keys for k in pair}
        return set()

    def _compute_deposit_red_keys(self, flat_values: dict[str, str]) -> set[str]:
        """Return the deposit-name flat key when deposit changed but cluster ID is the same."""
        if not self._session_captures:
            return set()

        # Current deposit name from flat staged values
        if "scan/deposit_name" in flat_values:
            current_deposit = flat_values.get("scan/deposit_name", "").strip()
            deposit_key = "scan/deposit_name"
        else:
            current_deposit = flat_values.get("values/deposit_name/name", "").strip()
            deposit_key = "values/deposit_name/name"

        if not current_deposit:
            return set()

        # Last committed capture's deposit name
        last = self._session_captures[-1]
        if last.get("cluster_id") != self._cluster_id:
            return set()

        # Extract deposit name from the nested capture structure
        scan = last.get("scan") or last.get("values", {}).get("deposit_name", {})
        if "deposit_name" in last.get("scan", {}):
            last_deposit = (last["scan"]["deposit_name"] or {}).get("value", "") or ""
        elif "deposit_name" in last.get("values", {}):
            last_deposit = (last["values"]["deposit_name"].get("name") or {}).get("value", "") or ""
        else:
            return set()

        if last_deposit.strip() and last_deposit.strip() != current_deposit:
            return {deposit_key}
        return set()

    def _compute_volume_mass_red_keys(self, flat_values: dict[str, str]) -> set[str]:
        """Return flat keys for volume and mass when volume*100 > mass."""
        red_keys = set()
        
        # Try structured format first
        if "scan/volume_int" in flat_values or "scan/mass" in flat_values:
            mass_str = flat_values.get("scan/mass", "").strip() or "0"
            vol_int_str = flat_values.get("scan/volume_int", "").strip() or "0"
            vol_dec_str = flat_values.get("scan/volume_dec", "").strip() or "0"
            
            try:
                mass = float(mass_str)
                volume = float(f"{vol_int_str}.{vol_dec_str}")
                
                if volume * 100 > mass:
                    red_keys.add("scan/mass")
                    red_keys.add("scan/volume_int")
                    red_keys.add("scan/volume_dec")
            except ValueError:
                pass
        else:
            # Try legacy format
            mass_str = flat_values.get("values/mass/value", "").strip() or "0"
            vol_int_str = flat_values.get("values/volume/int", "").strip() or "0"
            vol_dec_str = flat_values.get("values/volume/decimal", "").strip() or "0"
            
            try:
                mass = float(mass_str)
                volume = float(f"{vol_int_str}.{vol_dec_str}")
                
                if volume * 100 > mass:
                    red_keys.add("values/mass/value")
                    red_keys.add("values/volume/int")
                    red_keys.add("values/volume/decimal")
            except ValueError:
                pass
        
        return red_keys

    def _validate_staged_values(self, flat_values: dict[str, str]) -> set[str]:
        """Validation callback for real-time updates while editing staged values."""
        return (
            self._compute_amount_red_keys(flat_values)
            | self._compute_deposit_red_keys(flat_values)
            | self._compute_volume_mass_red_keys(flat_values)
        )

    def _collect_flat_staged_values(self) -> dict[str, str]:
        """Collect a flat key->value dict from staged data for display in edit fields."""
        flat: dict[str, str] = {}
        if not self._staged_data:
            return flat

        def _flatten(obj, prefix=""):
            if isinstance(obj, list):
                for i, item in enumerate(obj):
                    _flatten(item, f"{prefix}[{i}]")
            elif isinstance(obj, dict):
                if "value" in obj and "confidence" in obj:
                    flat[prefix] = obj["value"] or ""
                else:
                    for k, v in obj.items():
                        _flatten(v, f"{prefix}/{k}" if prefix else k)

        for key, val in self._staged_data.items():
            if key in ("timestamp", "_structured"):
                continue
            _flatten(val, key)
        return flat

    def _apply_edits_to_staged(self, edits: dict[str, str]) -> None:
        """Write edited values back into self._staged_data.

        Values that differ from the original staged value are treated as manually
        entered; their confidence is set to None so they can be identified as such
        in the raw JSON while being exported as confidence=1.0.
        """
        if not self._staged_data:
            return

        original = self._collect_flat_staged_values()

        def _set_value(obj, path_parts: list[str], value: str, manual: bool):
            if len(path_parts) == 0:
                return
            key = path_parts[0]
            # Handle array index like [0]
            if key.startswith("[") and key.endswith("]"):
                idx = int(key[1:-1])
                if isinstance(obj, list) and idx < len(obj):
                    if len(path_parts) == 1:
                        if isinstance(obj[idx], dict) and "value" in obj[idx]:
                            obj[idx]["value"] = value if value else None
                            if manual:
                                obj[idx]["confidence"] = None
                    else:
                        _set_value(obj[idx], path_parts[1:], value, manual)
            elif isinstance(obj, dict):
                if key in obj:
                    if len(path_parts) == 1:
                        if isinstance(obj[key], dict) and "value" in obj[key]:
                            obj[key]["value"] = value if value else None
                            if manual:
                                obj[key]["confidence"] = None
                    else:
                        _set_value(obj[key], path_parts[1:], value, manual)

        for flat_key, value in edits.items():
            manual = value != original.get(flat_key, value)
            # Split on "/" but treat "[" as starting a new segment so that
            # "scan/composition[0]/name" -> ["scan", "composition", "[0]", "name"]
            tokens = [t for t in flat_key.replace("[", "/[").split("/") if t]
            if not tokens:
                continue
            top_key = tokens[0]
            if top_key in self._staged_data:
                if len(tokens) == 1:
                    if isinstance(self._staged_data[top_key], dict) and "value" in self._staged_data[top_key]:
                        self._staged_data[top_key]["value"] = value if value else None
                        if manual:
                            self._staged_data[top_key]["confidence"] = None
                else:
                    _set_value(self._staged_data[top_key], tokens[1:], value, manual)

    def _on_stage_pressed(self) -> None:
        # F9 again while staged → cancel and return to live preview
        if self._staged_data is not None:
            self._staged_data = None
            self.controls_panel.unfreeze_staged()
            self.statusBar().showMessage("Unstaged — back to live preview")
            return

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

        # Freeze the controls panel to show editable staged values
        flat_values = self._collect_flat_staged_values()
        red_keys = (
            self._compute_amount_red_keys(flat_values)
            | self._compute_deposit_red_keys(flat_values)
        )
        self.controls_panel.freeze_staged(
            flat_values, red_keys=red_keys, validation_callback=self._validate_staged_values
        )

        warnings = []
        if self._compute_amount_red_keys(flat_values):
            warnings.append("amounts ≠ 100%")
        if self._compute_deposit_red_keys(flat_values):
            warnings.append("deposit name changed in same cluster")
        status = f"Staged {len(flat_values)} values — edit if needed, then F10 to commit"
        if warnings:
            status += "  ⚠ " + "; ".join(warnings)
        self.statusBar().showMessage(status)

    def _parse_location(self) -> dict:
        """Parse the location string into the structured JSON format."""
        loc_text = self.location_edit.text().strip()
        if not loc_text:
            return {}
        parts = [p.strip() for p in loc_text.split("/")]
        # Pad to 4 parts: system, gravity_well, region, place
        while len(parts) < 4:
            parts.append(None)
        return {
            "system": parts[0],
            "gravity_well": parts[1],
            "region": parts[2],
            "place": parts[3],
        }

    def _on_commit_to_json(self) -> None:
        if self._staged_data is None:
            self.statusBar().showMessage("Nothing staged — press F9 first")
            return

        # Apply any edits the user made in the frozen staged fields
        edits = self.controls_panel.get_staged_edits()
        self._apply_edits_to_staged(edits)

        capture_id = str(uuid.uuid4())
        is_structured = self._staged_data.get("_structured", False)
        if is_structured:
            payload = {k: v for k, v in self._staged_data.items()
                       if k not in ("timestamp", "_structured")}
        else:
            payload = {"values": self._staged_data["values"]}
        location = self._parse_location()
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
            # Unfreeze to restore live preview
            self.controls_panel.unfreeze_staged()
            count = len(self._session_captures)
            self.statusBar().showMessage(
                f"Committed capture #{count} (cluster {self._cluster_id[:8]}…) → {self._session_file.name}"
            )
        except Exception as e:
            self._session_captures.pop()
            self.statusBar().showMessage(f"JSON write error: {e}")

    def _on_prev_cluster_id(self) -> None:
        if self._cluster_history_index > 0:
            self._cluster_history_index -= 1
            self._cluster_id = self._cluster_history[self._cluster_history_index]
            self._update_cluster_label()
            self.statusBar().showMessage(f"Cluster ID ← {self._cluster_id[:8]}…")
        else:
            self.statusBar().showMessage("Already at first cluster ID")

    def _on_advance_cluster_id(self) -> None:
        # If we're not at the end of history, move forward
        if self._cluster_history_index < len(self._cluster_history) - 1:
            self._cluster_history_index += 1
            self._cluster_id = self._cluster_history[self._cluster_history_index]
        else:
            # Generate a new UUID and append to history
            self._cluster_id = str(uuid.uuid4())
            self._cluster_history.append(self._cluster_id)
            self._cluster_history_index = len(self._cluster_history) - 1
        self._update_cluster_label()
        self.statusBar().showMessage(f"Cluster ID → {self._cluster_id[:8]}…")

    def _update_cluster_label(self) -> None:
        self.cluster_id_label.setText(self._cluster_id[:8] + "…")
        self.cluster_id_label.setToolTip(self._cluster_id)

    def _on_config_changed(self) -> None:
        self._config.user = self.user_edit.text().strip()
        self._config.org = self.org_edit.text().strip()
        self._config.tool_version = self.version_edit.text().strip()
        # Parse location string into system and gravity_well
        loc_text = self.location_edit.text().strip()
        parts = [p.strip() for p in loc_text.split("/", 1)] if loc_text else ["", ""]
        self._config.system = parts[0] if len(parts) > 0 else ""
        self._config.gravity_well = parts[1] if len(parts) > 1 else ""
        self._config.save(self._config_path)

    # ── Cleanup ──────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        for p in self._pipelines.values():
            p.stop()
        if hasattr(self, "_hotkey_listener"):
            self._hotkey_listener.stop()
        event.accept()
