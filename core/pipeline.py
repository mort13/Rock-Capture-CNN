"""
Recognition pipeline orchestrator for Rock Capture CNN.
Ties capture -> anchor match -> ROI extract -> filter -> segment -> predict.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from PyQt6.QtCore import QObject, QRect, pyqtSignal

from core.capture import CaptureEngine
from core.anchor import AnchorMatcher, AnchorResult
from core.filters import ImageFilterPipeline
from core.segmenter import CharacterSegmenter, SegmentedChar
from core.profile import Profile, ROIDefinition
from cnn.predictor import Predictor


@dataclass
class ROIResult:
    """Result for a single ROI."""
    name: str
    raw_image: np.ndarray
    filtered_image: np.ndarray
    characters: list[SegmentedChar] = field(default_factory=list)
    recognized_text: str = ""


@dataclass
class FrameResult:
    """Full result of processing one frame."""
    frame: np.ndarray
    annotated_frame: np.ndarray
    anchor: AnchorResult
    roi_results: list[ROIResult] = field(default_factory=list)
    profile_name: str = ""


class RecognitionPipeline(QObject):
    """
    Orchestrates the full processing pipeline.

    Signals:
        frame_processed(FrameResult): emitted after each frame is processed
        anchor_lost(): emitted when anchor not found in frame
    """
    frame_processed = pyqtSignal(object)
    anchor_lost = pyqtSignal()

    def __init__(self, parent=None, predictor: Predictor | None = None, profile_name: str = ""):
        super().__init__(parent)
        self.capture_engine = CaptureEngine(self)
        self.anchor_matcher = AnchorMatcher()
        self.filter_pipeline = ImageFilterPipeline()
        self.segmenter = CharacterSegmenter()
        if predictor is not None:
            self.predictor = predictor
            self._owns_predictor = False
        else:
            self.predictor = Predictor()
            self._owns_predictor = True

        self._profile: Profile | None = None
        self._profile_name: str = profile_name
        self._labeler_mode: bool = False
        self._base_dir: Path = Path(".")

        self.capture_engine.frame_captured.connect(self._on_frame)

    @property
    def labeler_mode(self) -> bool:
        return self._labeler_mode

    @labeler_mode.setter
    def labeler_mode(self, value: bool) -> None:
        self._labeler_mode = value

    def load_profile(self, profile: Profile, base_dir: Path) -> None:
        """Load a profile: set anchor template, ROI definitions, model."""
        self._profile = profile
        self._base_dir = base_dir

        anchor_path = base_dir / "data" / "anchors" / profile.anchor_template_path
        if profile.anchor_template_path and anchor_path.exists():
            self.anchor_matcher.load_template(str(anchor_path))
        self.anchor_matcher.threshold = profile.anchor_match_threshold

        if self._owns_predictor:
            model_path = base_dir / "data" / "models" / profile.model_path
            if profile.model_path and model_path.exists():
                self.predictor.load_model(
                    str(model_path),
                    char_classes=profile.char_classes,
                )

        sr = profile.search_region
        self.capture_engine.set_search_region(
            QRect(sr["x"], sr["y"], sr["w"], sr["h"]),
            profile.monitor_index,
        )

    def start(self, fps: int = 10) -> None:
        self.capture_engine.start(fps)

    def stop(self) -> None:
        self.capture_engine.stop()

    @property
    def is_running(self) -> bool:
        return self.capture_engine.is_running

    def _on_frame(self, frame: np.ndarray) -> None:
        """Process a single captured frame through the full pipeline."""
        if self._profile is None:
            return

        anchor = self.anchor_matcher.find_anchor(
            frame,
            anchor_roi=self._profile.anchor_roi if self._profile.anchor_roi else None,
        )
        if not anchor.found:
            self.anchor_lost.emit()
            return

        roi_results = []
        for roi_def in self._profile.rois:
            if not roi_def.enabled:
                continue

            roi_x = anchor.x + roi_def.x_offset
            roi_y = anchor.y + roi_def.y_offset

            roi_x = max(0, roi_x)
            roi_y = max(0, roi_y)
            roi_x2 = min(frame.shape[1], roi_x + roi_def.width)
            roi_y2 = min(frame.shape[0], roi_y + roi_def.height)

            raw_roi = frame[roi_y:roi_y2, roi_x:roi_x2]
            if raw_roi.size == 0:
                continue

            filtered_roi = self.filter_pipeline.apply(raw_roi, roi_def.filters)

            pattern = roi_def.format_pattern

            if pattern and "{" in pattern:
                # Advanced pattern: variable groups + dot-width aware segmentation
                seg_results = self.segmenter.segment_formatted(
                    filtered_roi,
                    pattern,
                    char_width=roi_def.char_width,
                    dot_width=roi_def.dot_width,
                )
                characters = [c for c, lit in seg_results if c is not None]

                text = ""
                if not self._labeler_mode and self.predictor.is_loaded:
                    raw_preds = list(self.predictor.predict_sequence(
                        characters, allowed_chars=roi_def.allowed_chars,
                    ))
                    pred_idx = 0
                    parts = []
                    for c, lit in seg_results:
                        if lit is not None:
                            parts.append(lit)
                        else:
                            parts.append(raw_preds[pred_idx] if pred_idx < len(raw_preds) else "?")
                            pred_idx += 1
                    text = "".join(parts)

            else:
                # Simple / legacy pattern: x-count overrides char_count
                x_count = pattern.count("x") if pattern else 0
                effective_char_count = x_count if x_count > 0 else roi_def.char_count

                characters = self.segmenter.segment(
                    filtered_roi,
                    seg_mode=roi_def.seg_mode,
                    char_width=roi_def.char_width,
                    char_count=effective_char_count,
                )

                text = ""
                if not self._labeler_mode and self.predictor.is_loaded:
                    raw_text = self.predictor.predict_sequence(
                        characters,
                        allowed_chars=roi_def.allowed_chars,
                    )
                    text = (
                        self._apply_format_pattern(pattern, raw_text)
                        if pattern else raw_text
                    )

            roi_results.append(
                ROIResult(
                    name=roi_def.name,
                    raw_image=raw_roi,
                    filtered_image=filtered_roi,
                    characters=characters,
                    recognized_text=text,
                )
            )

        annotated = self._draw_overlays(
            frame, anchor, self._profile.rois, self._profile.anchor_roi
        )

        result = FrameResult(
            frame=frame,
            annotated_frame=annotated,
            anchor=anchor,
            roi_results=roi_results,
            profile_name=self._profile_name,
        )
        self.frame_processed.emit(result)

    @staticmethod
    def _apply_format_pattern(pattern: str, predicted: str) -> str:
        """
        Merge CNN predictions into a format pattern.

        'x' positions are filled from predicted (left to right).
        All other characters are inserted literally without CNN involvement.

        Examples:
            pattern="xx%",   predicted="75"  -> "75%"
            pattern="xxx.xx", predicted="12345" -> "123.45"
        """
        result = []
        pred_idx = 0
        for ch in pattern:
            if ch == "x":
                result.append(predicted[pred_idx] if pred_idx < len(predicted) else "?")
                pred_idx += 1
            else:
                result.append(ch)
        return "".join(result)

    def _draw_overlays(
        self,
        frame: np.ndarray,
        anchor: AnchorResult,
        rois: list[ROIDefinition],
        anchor_roi: dict | None = None,
    ) -> np.ndarray:
        """Draw anchor ROI (magenta), anchor match (green), and data ROI rects (cyan)."""
        vis = frame.copy()

        # Anchor ROI search area (magenta dashed)
        if anchor_roi and all(k in anchor_roi for k in ("x", "y", "w", "h")):
            arx, ary = anchor_roi["x"], anchor_roi["y"]
            arw, arh = anchor_roi["w"], anchor_roi["h"]
            cv2.rectangle(
                vis, (arx, ary), (arx + arw, ary + arh),
                (255, 0, 255), 1,
            )
            cv2.putText(
                vis, "anchor ROI", (arx, ary - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1,
            )

        # Anchor match (green)
        cv2.rectangle(
            vis,
            (anchor.x, anchor.y),
            (anchor.x + anchor.anchor_w, anchor.y + anchor.anchor_h),
            (0, 255, 0),
            2,
        )

        fh, fw = vis.shape[:2]
        for roi in rois:
            rx = max(0, anchor.x + roi.x_offset)
            ry = max(0, anchor.y + roi.y_offset)
            rx2 = min(fw, rx + roi.width)
            ry2 = min(fh, ry + roi.height)
            cv2.rectangle(
                vis,
                (rx, ry),
                (rx2, ry2),
                (255, 255, 0),
                1,
            )
            cv2.putText(
                vis,
                roi.name,
                (rx, max(0, ry - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 0),
                1,
            )

        return vis
