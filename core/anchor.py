"""
Anchor template matching for Rock Capture CNN.

Single-anchor (legacy) and multi-anchor (resolution-independent) modes.
Multi-anchor computes a similarity / affine transform from 2-3 reference
points so that ROI positions scale and translate automatically.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AnchorResult:
    """Result of a single anchor template match."""
    found: bool
    x: int = 0
    y: int = 0
    confidence: float = 0.0
    anchor_w: int = 0
    anchor_h: int = 0


@dataclass
class MultiAnchorResult:
    """Result of matching 2-3 anchors and computing a transform."""
    found: bool
    # Per-anchor results keyed by AnchorPoint.name
    anchor_results: dict[str, AnchorResult] = field(default_factory=dict)
    # 2×3 affine matrix  (None when not enough anchors matched)
    transform: np.ndarray | None = None
    # Uniform scale factor extracted from the transform
    scale: float = 1.0


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

def compute_transform(
    ref_points: list[tuple[float, float]],
    det_points: list[tuple[float, float]],
) -> tuple[np.ndarray, float]:
    """Compute a similarity or affine transform from *ref_points* → *det_points*.

    With 2 points a partial-affine (similarity: translate + rotate + uniform
    scale) is estimated.  With 3+ points a full affine is computed.

    Returns ``(M, scale)`` where *M* is a 2×3 matrix and *scale* is the
    uniform scale factor.
    """
    src = np.array(ref_points, dtype=np.float32)
    dst = np.array(det_points, dtype=np.float32)

    if len(ref_points) == 2:
        M, _ = cv2.estimateAffinePartial2D(
            src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2),
        )
        if M is None:
            M = np.eye(2, 3, dtype=np.float64)
    else:
        M = cv2.getAffineTransform(src[:3], dst[:3])

    scale = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
    return M, scale


def transform_point(M: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Apply 2×3 affine matrix *M* to point *(x, y)*."""
    pt = np.array([x, y, 1.0], dtype=np.float64)
    out = M @ pt
    return float(out[0]), float(out[1])


# ---------------------------------------------------------------------------
# AnchorMatcher – handles both legacy single-anchor and multi-anchor
# ---------------------------------------------------------------------------

class AnchorMatcher:
    """
    Finds template images (anchors) within a captured frame
    using ``cv2.matchTemplate`` with ``TM_CCOEFF_NORMED``.
    """

    # Default scale search range: covers 0.5× – 2.0× of the reference resolution.
    # 17 steps → ~8.8 % increment between probes, fast enough for real-time use.
    SCALE_MIN: float = 0.5
    SCALE_MAX: float = 2.0
    SCALE_STEPS: int = 17

    def __init__(self):
        # Legacy single-template state
        self._template: np.ndarray | None = None
        self._template_gray: np.ndarray | None = None
        self._threshold: float = 0.7

        # Multi-anchor template cache: name → (gray_img, color_img)
        self._templates: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        # Multi-scale matching config
        self.multi_scale: bool = True
        self.scale_range: tuple[float, float] = (self.SCALE_MIN, self.SCALE_MAX)
        self.scale_steps: int = self.SCALE_STEPS

        # Last-found scale cache: id(template_gray) → scale float
        # Templates are kept alive in self._templates so the id is stable.
        self._scale_cache: dict[int, float] = {}

    # -- Legacy single-anchor properties ----------------------------------

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = max(0.0, min(1.0, value))

    @property
    def is_loaded(self) -> bool:
        return self._template is not None

    @property
    def template_size(self) -> tuple[int, int]:
        """(width, height) of loaded legacy template."""
        if self._template is None:
            return (0, 0)
        h, w = self._template.shape[:2]
        return (w, h)

    def load_template(self, path: str | Path) -> bool:
        """Load the legacy single anchor template. Returns True on success."""
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return False
        # Drop cached scale for the old template before replacing it.
        if self._template_gray is not None:
            self._scale_cache.pop(id(self._template_gray), None)
        self._template = img
        self._template_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return True

    # -- Multi-anchor template management ----------------------------------

    def load_anchor_template(self, name: str, path: str | Path) -> bool:
        """Load a named anchor template for multi-anchor matching."""
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._templates[name] = (gray, img)
        return True

    def clear_anchor_templates(self) -> None:
        self._templates.clear()
        self._scale_cache.clear()

    # -- Legacy single-anchor matching -------------------------------------

    def find_anchor(
        self, frame: np.ndarray, anchor_roi: dict | None = None,
    ) -> AnchorResult:
        """Search for the legacy single anchor template in *frame*."""
        if self._template_gray is None:
            return AnchorResult(found=False)

        search_rect: tuple[int, int, int, int] | None = None
        if anchor_roi and all(k in anchor_roi for k in ("x", "y", "w", "h")):
            search_rect = (
                anchor_roi["x"], anchor_roi["y"],
                anchor_roi["w"], anchor_roi["h"],
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self._match_template_multiscale(
            gray, self._template_gray, self._threshold, search_rect
        )

    # -- Multi-anchor matching ---------------------------------------------

    def _match_template_in(
        self,
        gray_frame: np.ndarray,
        template_gray: np.ndarray,
        threshold: float,
        search_rect: tuple[int, int, int, int] | None = None,
    ) -> AnchorResult:
        """Low-level: match *template_gray* inside *gray_frame*.

        *search_rect* ``(x, y, w, h)`` optionally constrains the search area.
        Returned coordinates are in the full *gray_frame* coordinate space.
        """
        ox, oy = 0, 0
        region = gray_frame
        if search_rect:
            sx, sy, sw, sh = search_rect
            sx = max(0, sx)
            sy = max(0, sy)
            sx2 = min(gray_frame.shape[1], sx + sw)
            sy2 = min(gray_frame.shape[0], sy + sh)
            region = gray_frame[sy:sy2, sx:sx2]
            ox, oy = sx, sy

        if region.size == 0:
            return AnchorResult(found=False)

        th, tw = template_gray.shape[:2]
        if region.shape[0] < th or region.shape[1] < tw:
            return AnchorResult(found=False)

        res = cv2.matchTemplate(region, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val >= threshold:
            return AnchorResult(
                found=True,
                x=max_loc[0] + ox,
                y=max_loc[1] + oy,
                confidence=max_val,
                anchor_w=tw,
                anchor_h=th,
            )
        return AnchorResult(found=False, confidence=max_val)

    def _match_template_multiscale(
        self,
        gray_frame: np.ndarray,
        template_gray: np.ndarray,
        threshold: float,
        search_rect: tuple[int, int, int, int] | None = None,
    ) -> AnchorResult:
        """Multi-scale wrapper around ``_match_template_in``.

        Resizes *template_gray* over ``self.scale_range`` (``self.scale_steps``
        probes) and returns the best-confidence match.  Falls back to a single
        unit-scale attempt when ``self.multi_scale`` is False.
        """
        if not self.multi_scale:
            return self._match_template_in(
                gray_frame, template_gray, threshold, search_rect
            )

        th, tw = template_gray.shape[:2]
        cache_key = id(template_gray)

        # Try last-known scale first — avoids the full sweep every frame.
        cached_scale = self._scale_cache.get(cache_key)
        if cached_scale is not None:
            new_w = max(1, int(round(tw * cached_scale)))
            new_h = max(1, int(round(th * cached_scale)))
            scaled_tpl = cv2.resize(
                template_gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )
            ar = self._match_template_in(gray_frame, scaled_tpl, threshold, search_rect)
            if ar.found:
                return ar

        # Full sweep — runs when cache is cold or cached scale no longer matches.
        best = AnchorResult(found=False, confidence=0.0)
        best_scale: float = 1.0
        scales = np.linspace(self.scale_range[0], self.scale_range[1], self.scale_steps)

        for s in scales:
            new_w = max(1, int(round(tw * s)))
            new_h = max(1, int(round(th * s)))
            # Skip if scaled template would be larger than the search area
            if search_rect is not None:
                _, _, sw, sh = search_rect
                if new_w > sw or new_h > sh:
                    continue
            scaled_tpl = cv2.resize(
                template_gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )
            ar = self._match_template_in(gray_frame, scaled_tpl, threshold, search_rect)
            if ar.confidence > best.confidence:
                best = ar
                best_scale = s

        if best.found:
            self._scale_cache[cache_key] = best_scale

        return best

    def find_anchors(
        self,
        frame: np.ndarray,
        anchor_points: list,
    ) -> MultiAnchorResult:
        """Find all *anchor_points* in *frame* and compute an affine transform.

        ``anchor_points`` is a list of
        :class:`~core.profile.AnchorPoint` instances.

        Returns a :class:`MultiAnchorResult` whose *transform* maps reference-
        frame coordinates to current-frame coordinates.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results: dict[str, AnchorResult] = {}
        ref_pts: list[tuple[float, float]] = []
        det_pts: list[tuple[float, float]] = []

        for ap in anchor_points:
            tpl = self._templates.get(ap.name)
            if tpl is None:
                continue
            tpl_gray, _ = tpl
            ar = self._match_template_multiscale(gray, tpl_gray, ap.match_threshold)
            results[ap.name] = ar
            if ar.found:
                ref_pts.append((ap.ref_x, ap.ref_y))
                det_pts.append((float(ar.x), float(ar.y)))

        if len(det_pts) < 2:
            return MultiAnchorResult(found=False, anchor_results=results)

        M, scale = compute_transform(ref_pts, det_pts)
        return MultiAnchorResult(
            found=True,
            anchor_results=results,
            transform=M,
            scale=scale,
        )

    def find_anchor_near(
        self,
        frame_gray: np.ndarray,
        anchor_name: str,
        expected_x: float,
        expected_y: float,
        threshold: float = 0.7,
        padding: int = 40,
        search_region: dict | None = None,
    ) -> AnchorResult:
        """Search for the named anchor template near an expected position.

        Used for sub-anchor refinement after the main transform has been
        applied.

        Args:
            frame_gray: Grayscale frame
            anchor_name: Name of sub-anchor template
            expected_x, expected_y: Expected position (from transform)
            threshold: Template match threshold
            padding: Default padding (used if search_region not provided)
            search_region: Optional dict with keys x, y, width, height.
                           If provided, search within this region (in frame coords).
                           Otherwise, uses padding around expected position.
        """
        tpl = self._templates.get(anchor_name)
        if tpl is None:
            return AnchorResult(found=False)
        tpl_gray, _ = tpl

        if search_region is not None:
            # Use explicit search region
            sx = int(search_region.get("x", 0))
            sy = int(search_region.get("y", 0))
            sw = int(search_region.get("width", frame_gray.shape[1] - sx))
            sh = int(search_region.get("height", frame_gray.shape[0] - sy))
        else:
            # Fallback to padding around expected position
            th, tw = tpl_gray.shape[:2]
            sx = int(expected_x) - padding
            sy = int(expected_y) - padding
            sw = tw + 2 * padding
            sh = th + 2 * padding

        return self._match_template_multiscale(
            frame_gray, tpl_gray, threshold, (sx, sy, sw, sh),
        )
