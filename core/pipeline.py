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
from core.anchor import AnchorMatcher, AnchorResult, MultiAnchorResult, transform_point
from core.filters import ImageFilterPipeline
from core.segmenter import CharacterSegmenter, SegmentedChar
from core.profile import Profile, ROIDefinition
from core.template_matcher import WordTemplateMatcher
from cnn.predictor import Predictor
from word_cnn.predictor import WordPredictor


@dataclass
class ROIResult:
    """Result for a single ROI."""
    name: str
    raw_image: np.ndarray
    filtered_image: np.ndarray
    characters: list[SegmentedChar] = field(default_factory=list)
    recognized_text: str = ""
    confidence: float = 0.0
    word_scores: list[tuple[str, float]] = field(default_factory=list)
    recognition_mode: str = ""


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

    def __init__(self, parent=None, predictor: Predictor | None = None,
                 word_predictor: WordPredictor | None = None,
                 profile_name: str = ""):
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

        self.word_predictor = word_predictor or WordPredictor()

        self._profile: Profile | None = None
        self._profile_name: str = profile_name
        self._labeler_mode: bool = False
        self._base_dir: Path = Path(".")
        self._template_matchers: dict[str, WordTemplateMatcher] = {}
        self._overlay_visibility: dict = {
            "show_anchors": True,
            "show_sub_anchors": True,
            "show_search_regions": True,
            "show_rois": True,
        }

        self.capture_engine.frame_captured.connect(self._on_frame)

    @property
    def labeler_mode(self) -> bool:
        return self._labeler_mode

    @labeler_mode.setter
    def labeler_mode(self, value: bool) -> None:
        self._labeler_mode = value

    @property
    def overlay_visibility(self) -> dict:
        return self._overlay_visibility

    @overlay_visibility.setter
    def overlay_visibility(self, value: dict) -> None:
        self._overlay_visibility = value

    def load_profile(self, profile: Profile, base_dir: Path) -> None:
        """Load a profile: set anchor template(s), ROI definitions, model."""
        self._profile = profile
        self._base_dir = base_dir
        self._template_matchers.clear()  # Invalidate cached matchers on profile reload

        if profile.uses_multi_anchor:
            # Multi-anchor mode: load all anchor + sub-anchor templates
            self.anchor_matcher.clear_anchor_templates()
            anchors_dir = base_dir / "data" / "anchors"
            for ap in profile.anchors + profile.sub_anchors:
                if ap.template_path:
                    p = anchors_dir / ap.template_path
                    if p.exists():
                        self.anchor_matcher.load_anchor_template(ap.name, str(p))
        else:
            # Legacy single-anchor mode
            anchor_path = base_dir / "data" / "anchors" / profile.anchor_template_path
            if profile.anchor_template_path and anchor_path.exists():
                self.anchor_matcher.load_template(str(anchor_path))
            self.anchor_matcher.threshold = profile.anchor_match_threshold

        sr = profile.search_region
        self.capture_engine.set_search_region(
            QRect(sr["x"], sr["y"], sr["w"], sr["h"]),
            profile.monitor_index,
        )

    def reload_templates(self) -> None:
        """Clear cached template matchers so they are reloaded from disk on next use."""
        self._template_matchers.clear()

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

        if self._profile.uses_multi_anchor:
            self._on_frame_multi_anchor(frame)
        else:
            self._on_frame_legacy(frame)

    # -- Legacy single-anchor frame processing ----------------------------

    def _on_frame_legacy(self, frame: np.ndarray) -> None:
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

            # Crop to visible area without clamping the origin, so the ROI can
            # scroll freely beyond the frame edges (partially off-screen = cropped).
            res = self._extract_and_recognize(frame, roi_def, roi_x, roi_y, roi_def.width, roi_def.height)
            if res is not None:
                roi_results.append(res)

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

    # -- Multi-anchor frame processing ------------------------------------

    def _on_frame_multi_anchor(self, frame: np.ndarray) -> None:
        profile = self._profile
        ma_result = self.anchor_matcher.find_anchors(frame, profile.anchors)

        if not ma_result.found or ma_result.transform is None:
            self.anchor_lost.emit()
            return

        M = ma_result.transform
        scale = ma_result.scale

        # Resolve sub-anchors
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sub_anchor_positions: dict[str, tuple[float, float]] = {}
        for sa in profile.sub_anchors:
            ex, ey = transform_point(M, sa.ref_x, sa.ref_y)
            
            # Compute transformed search region if defined
            search_region = None
            if sa.search_region:
                sr = sa.search_region
                # Transform the search region bounds
                tl_x, tl_y = transform_point(M, sr.get("x", 0), sr.get("y", 0))
                br_x, br_y = transform_point(
                    M,
                    sr.get("x", 0) + sr.get("width", 100),
                    sr.get("y", 0) + sr.get("height", 100),
                )
                search_region = {
                    "x": min(tl_x, br_x),
                    "y": min(tl_y, br_y),
                    "width": abs(br_x - tl_x),
                    "height": abs(br_y - tl_y),
                }
            
            ar = self.anchor_matcher.find_anchor_near(
                gray, sa.name, ex, ey,
                threshold=sa.match_threshold,
                padding=max(40, int(40 * scale)),
                search_region=search_region,
            )
            if ar.found:
                sub_anchor_positions[sa.name] = (float(ar.x), float(ar.y))
            else:
                # Fall back to transformed position
                sub_anchor_positions[sa.name] = (ex, ey)

        # Build a synthetic AnchorResult for FrameResult (use first matched anchor)
        first_ar = next(
            (ar for ar in ma_result.anchor_results.values() if ar.found),
            AnchorResult(found=True),
        )

        roi_results = []
        for roi_def in profile.rois:
            if not roi_def.enabled:
                continue

            # Compute effective width / height scaled by transform
            eff_w = max(1, int(round(roi_def.width * scale)))
            eff_h = max(1, int(round(roi_def.height * scale)))

            if roi_def.sub_anchor and roi_def.sub_anchor in sub_anchor_positions:
                # Position relative to sub-anchor
                sa = profile.get_sub_anchor(roi_def.sub_anchor)
                sa_pos = sub_anchor_positions[roi_def.sub_anchor]
                dx = roi_def.ref_x - sa.ref_x
                dy = roi_def.ref_y - sa.ref_y
                roi_x = int(round(sa_pos[0] + dx * scale))
                roi_y = int(round(sa_pos[1] + dy * scale))
            else:
                # Direct transform from reference frame
                tx, ty = transform_point(M, roi_def.ref_x, roi_def.ref_y)
                roi_x = int(round(tx))
                roi_y = int(round(ty))

            res = self._extract_and_recognize(frame, roi_def, roi_x, roi_y, eff_w, eff_h)
            if res is not None:
                roi_results.append(res)

        annotated = self._draw_overlays_multi(
            frame, ma_result, profile, sub_anchor_positions, scale,
            overlay_visibility=self._overlay_visibility,
        )

        result = FrameResult(
            frame=frame,
            annotated_frame=annotated,
            anchor=first_ar,
            roi_results=roi_results,
            profile_name=self._profile_name,
        )
        self.frame_processed.emit(result)

    # -- Shared ROI extraction + recognition ------------------------------

    def _extract_and_recognize(
        self,
        frame: np.ndarray,
        roi_def,
        roi_x: int,
        roi_y: int,
        roi_w: int,
        roi_h: int,
    ) -> ROIResult | None:
        """Crop a ROI from *frame* and run recognition. Returns None if out-of-bounds."""
        slice_x1 = max(0, roi_x)
        slice_y1 = max(0, roi_y)
        slice_x2 = min(frame.shape[1], roi_x + roi_w)
        slice_y2 = min(frame.shape[0], roi_y + roi_h)

        if slice_x1 >= slice_x2 or slice_y1 >= slice_y2:
            return None

        raw_roi = frame[slice_y1:slice_y2, slice_x1:slice_x2]
        if raw_roi.size == 0:
            return None

        filtered_roi = self.filter_pipeline.apply(raw_roi, roi_def.filters)

        # ── Template matching mode ─────────────────────────────
        if roi_def.recognition_mode == "template":
            matcher = self._get_template_matcher(roi_def.template_dir)
            if matcher and matcher.is_loaded:
                result = matcher.match(raw_roi)
                text = result.label
                conf = result.confidence
            else:
                if not hasattr(self, '_tmpl_warn_logged'):
                    self._tmpl_warn_logged = set()
                key = roi_def.name
                if key not in self._tmpl_warn_logged:
                    print(f"[Pipeline] Template matcher not loaded for roi='{roi_def.name}' template_dir='{roi_def.template_dir}'")
                    self._tmpl_warn_logged.add(key)
                text = "?"
                conf = 0.0
            return ROIResult(
                name=roi_def.name,
                raw_image=raw_roi,
                filtered_image=filtered_roi,
                characters=[],
                recognized_text=text,
                confidence=conf,
                recognition_mode="template",
            )

        # ── Word CNN mode ──────────────────────────────────────
        if roi_def.recognition_mode == "word_cnn":
            if self.word_predictor.is_loaded:
                word_scores = self.word_predictor.predict_all(raw_roi)
                text, conf = word_scores[0] if word_scores else ("?", 0.0)
            else:
                word_scores = []
                text = "?"
                conf = 0.0
            return ROIResult(
                name=roi_def.name,
                raw_image=raw_roi,
                filtered_image=filtered_roi,
                characters=[],
                recognized_text=text,
                confidence=conf,
                word_scores=word_scores,
                recognition_mode="word_cnn",
            )

        # ── CNN recognition mode ───────────────────────────────
        pattern = roi_def.format_pattern

        if pattern and "{" in pattern:
            seg_results = self.segmenter.segment_formatted(
                filtered_roi,
                pattern,
                char_width=roi_def.char_width,
                dot_width=roi_def.dot_width,
            )
            characters = [c for c, lit in seg_results if c is not None]

            text = ""
            conf = 0.0
            if not self._labeler_mode and self.predictor.is_loaded:
                preds_with_conf = self.predictor.predict_sequence(
                    characters, allowed_chars=roi_def.allowed_chars,
                )
                pred_idx = 0
                parts = []
                confs = []
                for c, lit in seg_results:
                    if lit is not None:
                        parts.append(lit)
                    else:
                        if pred_idx < len(preds_with_conf):
                            ch, ch_conf = preds_with_conf[pred_idx]
                            parts.append(ch)
                            confs.append(ch_conf)
                        else:
                            parts.append("?")
                        pred_idx += 1
                text = "".join(parts)
                conf = sum(confs) / len(confs) if confs else 0.0
        else:
            x_count = pattern.count("x") if pattern else 0
            effective_char_count = x_count if x_count > 0 else roi_def.char_count

            characters = self.segmenter.segment(
                filtered_roi,
                seg_mode=roi_def.seg_mode,
                char_width=roi_def.char_width,
                char_count=effective_char_count,
            )

            text = ""
            conf = 0.0
            if not self._labeler_mode and self.predictor.is_loaded:
                preds_with_conf = self.predictor.predict_sequence(
                    characters,
                    allowed_chars=roi_def.allowed_chars,
                )
                raw_text = "".join(ch for ch, _ in preds_with_conf)
                confs = [c for _, c in preds_with_conf]
                text = (
                    self._apply_format_pattern(pattern, raw_text)
                    if pattern else raw_text
                )
                conf = sum(confs) / len(confs) if confs else 0.0

        return ROIResult(
            name=roi_def.name,
            raw_image=raw_roi,
            filtered_image=filtered_roi,
            characters=characters,
            recognized_text=text,
            confidence=conf,
        )

    def _get_template_matcher(self, template_dir: str) -> WordTemplateMatcher | None:
        """Return a cached WordTemplateMatcher for the given directory."""
        if not template_dir:
            return None
        if template_dir not in self._template_matchers:
            matcher = WordTemplateMatcher()
            full_path = self._base_dir / "data" / template_dir
            matcher.load_templates(full_path)
            self._template_matchers[template_dir] = matcher
        return self._template_matchers[template_dir]

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

        for roi in rois:
            rx = anchor.x + roi.x_offset
            ry = anchor.y + roi.y_offset
            rx2 = rx + roi.width
            ry2 = ry + roi.height
            # OpenCV clips drawing operations to image bounds, so out-of-frame
            # coordinates are safe and correctly show a partial rectangle.
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
                (max(0, rx), max(0, ry - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 0),
                1,
            )

        return vis

    def _draw_overlays_multi(
        self,
        frame: np.ndarray,
        ma_result: MultiAnchorResult,
        profile: Profile,
        sub_anchor_positions: dict[str, tuple[float, float]],
        scale: float,
        overlay_visibility: dict | None = None,
    ) -> np.ndarray:
        """Draw multi-anchor overlays: anchors (green), sub-anchors (magenta), ROIs (cyan), search regions (orange).
        
        Args:
            overlay_visibility: dict with keys show_anchors, show_sub_anchors, show_search_regions, show_rois
        """
        if overlay_visibility is None:
            overlay_visibility = {
                "show_anchors": True,
                "show_sub_anchors": True,
                "show_search_regions": True,
                "show_rois": True,
            }
        
        vis = frame.copy()
        M = ma_result.transform

        # Draw main anchors (green)
        if overlay_visibility.get("show_anchors", True):
            for name, ar in ma_result.anchor_results.items():
                if ar.found:
                    cv2.rectangle(
                        vis, (ar.x, ar.y),
                        (ar.x + ar.anchor_w, ar.y + ar.anchor_h),
                        (0, 255, 0), 2,
                    )
                    cv2.putText(
                        vis, name, (ar.x, ar.y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
                    )

        # Draw sub-anchors (magenta)
        if overlay_visibility.get("show_sub_anchors", True):
            for sa_name, (sx, sy) in sub_anchor_positions.items():
                ix, iy = int(sx), int(sy)
                cv2.drawMarker(
                    vis, (ix, iy), (255, 0, 255),
                    cv2.MARKER_CROSS, 12, 1,
                )
                cv2.putText(
                    vis, sa_name, (ix + 6, iy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1,
                )

        # Draw search regions (orange dashed)
        if overlay_visibility.get("show_search_regions", True):
            for sa in profile.sub_anchors:
                if sa.search_region:
                    sr = sa.search_region
                    # Transform search region bounds
                    tl_x, tl_y = transform_point(M, sr.get("x", 0), sr.get("y", 0))
                    br_x, br_y = transform_point(
                        M,
                        sr.get("x", 0) + sr.get("width", 100),
                        sr.get("y", 0) + sr.get("height", 100),
                    )
                    x1, y1 = int(min(tl_x, br_x)), int(min(tl_y, br_y))
                    x2, y2 = int(max(tl_x, br_x)), int(max(tl_y, br_y))
                    
                    # Draw dashed rectangle (using line segments)
                    pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
                    for i in range(len(pts) - 1):
                        p1, p2 = pts[i], pts[i + 1]
                        # Dashed line effect: draw small segments with gaps
                        dx = p2[0] - p1[0]
                        dy = p2[1] - p1[1]
                        dist = int(np.sqrt(dx*dx + dy*dy))
                        steps = max(1, dist // 8)
                        for j in range(0, steps, 2):
                            start = (
                                p1[0] + dx * j // steps,
                                p1[1] + dy * j // steps,
                            )
                            end = (
                                p1[0] + dx * min(j + 1, steps) // steps,
                                p1[1] + dy * min(j + 1, steps) // steps,
                            )
                            cv2.line(vis, start, end, (0, 165, 255), 1)  # Orange
                    
                    cv2.putText(
                        vis, f"search:{sa.name}", (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1,
                    )

        # Draw ROIs (cyan)
        if overlay_visibility.get("show_rois", True):
            for roi in profile.rois:
                if not roi.enabled:
                    continue
                eff_w = max(1, int(round(roi.width * scale)))
                eff_h = max(1, int(round(roi.height * scale)))

                if roi.sub_anchor and roi.sub_anchor in sub_anchor_positions:
                    sa = profile.get_sub_anchor(roi.sub_anchor)
                    sa_pos = sub_anchor_positions[roi.sub_anchor]
                    dx = roi.ref_x - sa.ref_x
                    dy = roi.ref_y - sa.ref_y
                    rx = int(round(sa_pos[0] + dx * scale))
                    ry = int(round(sa_pos[1] + dy * scale))
                elif M is not None:
                    tx, ty = transform_point(M, roi.ref_x, roi.ref_y)
                    rx = int(round(tx))
                    ry = int(round(ty))
                else:
                    continue

                cv2.rectangle(vis, (rx, ry), (rx + eff_w, ry + eff_h), (255, 255, 0), 1)
                cv2.putText(
                    vis, roi.name, (max(0, rx), max(0, ry - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1,
                )

        return vis
