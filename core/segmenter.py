"""
Character segmentation for Rock Capture CNN.

Three modes:
  projection  - vertical histogram valleys (default, handles close digits)
  contour     - OpenCV contour bounding boxes (needs clear gaps between chars)
  fixed_width - equal-width column slices (handles touching digits)
"""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class SegmentedChar:
    """A single segmented character with its position and normalized image."""
    image: np.ndarray            # 28x28 grayscale, float32, [0,1]
    bbox: tuple[int, int, int, int]  # (x, y, w, h) in the ROI image
    original: np.ndarray         # original crop before normalization


class CharacterSegmenter:
    """Segments a filtered ROI image into individual 28x28 character crops."""

    TARGET_SIZE = 28

    def __init__(self):
        self.min_char_width: int = 3
        self.min_char_height: int = 5
        self.padding: int = 2

    # ── Public API ────────────────────────────────────────────────

    def segment(
        self,
        roi_image: np.ndarray,
        seg_mode: str = "projection",
        char_width: int = 0,
        char_count: int = 0,
    ) -> list[SegmentedChar]:
        """
        Segment an ROI image into individual characters.

        Args:
            roi_image:   BGR or grayscale numpy array.
            seg_mode:    "projection" | "contour" | "fixed_width"
            char_count:  fixed_width only — exact number of characters expected.
                         When > 0 this takes priority over char_width and gives
                         perfectly even slices regardless of pixel rounding.
            char_width:  fixed_width fallback — pixel width per character.
                         Used only when char_count == 0.
                         0 = estimate from ROI height (approx square chars).
        """
        if roi_image is None or roi_image.size == 0:
            return []

        gray = self._to_gray(roi_image)

        if seg_mode == "contour":
            boxes = self._boxes_contour(gray)
        elif seg_mode == "fixed_width":
            boxes = self._boxes_fixed_width(gray, char_width, char_count)
        else:
            boxes = self._boxes_projection(gray)

        return [self._make_char(gray, box) for box in boxes]

    def segment_formatted(
        self,
        roi_image: np.ndarray,
        pattern: str,
        char_width: int = 0,
        dot_width: int = 0,
    ) -> list[tuple["SegmentedChar | None", "str | None"]]:
        """
        Segment using an advanced format pattern with variable-width groups.

        Returns a list of (SegmentedChar | None, literal | None) tuples:
            (SegmentedChar, None)  → position predicted by the CNN
            (None, 'char')        → literal character, no CNN involvement

        Pattern token syntax (may be mixed freely):
            x           → single predicted position (char_width pixels)
            {n}         → exactly n predicted positions (n × char_width pixels)
            {n1,n2,...} → variable: projection finds actual count in [min,max]
                          pixel budget = (total - fixed) / n_variable_groups
            .           → decimal-point glyph IS in the image; dot_width pixels
                          consumed but the literal '.' is inserted in the output
            other char  → pure literal (e.g. '%'), 0 pixels consumed from image

        Examples:
            "{1,3}.xx"  → 1-3 digits, dot glyph (dot_width px), 2 digits
            "{1,3}.{2}" → same, explicit fixed-2 after dot
            "xx%"       → 2 digits, literal % (not in image)
        """
        if roi_image is None or roi_image.size == 0:
            return []

        gray = self._to_gray(roi_image)
        h, w = gray.shape

        tokens = self._parse_format_tokens(pattern)
        px_widths = self._allocate_pixels(tokens, char_width, dot_width, w, h)

        results: list[tuple[SegmentedChar | None, str | None]] = []
        x_cursor = 0

        for token, px_w in zip(tokens, px_widths):
            x1 = x_cursor
            x2 = min(x1 + px_w, w)
            x_cursor += px_w

            if token[0] == "literal":
                results.append((None, token[1]))
                continue

            if x2 <= x1:
                continue

            region = gray[:, x1:x2]
            if region.size == 0:
                continue

            if token[0] == "fixed":
                n = token[1]
                boxes = self._boxes_fixed_width(region, char_width, char_count=n)
            else:  # variable
                _, n_min, n_max = token
                cw = char_width if char_width > 0 else max(1, int(gray.shape[0] * 0.6))
                n = self._estimate_digit_count(region, cw, n_min, n_max)
                # Crop to exactly the predicted digit area before slicing
                cropped = region[:, : n * cw]
                boxes = self._boxes_fixed_width(cropped, cw, char_count=n)

            # Offset box x-coordinates into full-ROI space
            for bx, by, bw, bh in boxes:
                results.append((self._make_char(gray, (x1 + bx, by, bw, bh)), None))

        return results

    # ── Format pattern helpers ────────────────────────────────────

    @staticmethod
    def _parse_format_tokens(
        pattern: str,
    ) -> list[tuple]:
        """
        Parse a format pattern string into a list of tokens:
            ('fixed', n)              - exactly n chars
            ('variable', n_min, n_max)- projection-determined count in [min,max]
            ('literal', char)         - insert char literally; '.' also eats pixels
        """
        tokens = []
        i = 0
        while i < len(pattern):
            ch = pattern[i]
            if ch == "{":
                end = pattern.index("}", i)
                nums = sorted(int(n.strip()) for n in pattern[i + 1 : end].split(","))
                n_min, n_max = nums[0], nums[-1]
                tokens.append(("fixed", n_min) if n_min == n_max else ("variable", n_min, n_max))
                i = end + 1
            elif ch == "x":
                tokens.append(("fixed", 1))
                i += 1
            else:
                tokens.append(("literal", ch))
                i += 1
        return tokens

    @staticmethod
    def _allocate_pixels(
        tokens: list[tuple],
        char_width: int,
        dot_width: int,
        total_w: int,
        roi_h: int,
    ) -> list[int]:
        """
        Calculate the pixel width each token consumes from the ROI.

        Fixed/variable tokens get char_width px per character slot.
        Literal '.' consumes dot_width (or char_width // 4 if 0).
        Other literals consume 0 px (they are not present as glyphs).
        Variable groups split the remaining budget equally.
        """
        cw = char_width if char_width > 0 else max(1, int(roi_h * 0.6))
        dw = dot_width if dot_width > 0 else max(1, cw // 4)

        fixed_total = 0
        n_variable = 0
        for t in tokens:
            if t[0] == "fixed":
                fixed_total += t[1] * cw
            elif t[0] == "variable":
                n_variable += 1
            elif t[0] == "literal":
                fixed_total += dw if t[1] == "." else 0

        variable_budget = (
            max(0, total_w - fixed_total) // n_variable if n_variable > 0 else 0
        )

        widths = []
        for t in tokens:
            if t[0] == "fixed":
                widths.append(t[1] * cw)
            elif t[0] == "variable":
                _, n_min, n_max = t
                widths.append(min(variable_budget, n_max * cw))
            else:
                widths.append(dw if t[1] == "." else 0)
        return widths

    def _estimate_digit_count(self, region: np.ndarray, char_width: int, n_min: int, n_max: int) -> int:
        """
        Estimate how many digits are present in *region* by measuring the
        rightmost column that contains above-threshold pixels, then dividing
        by char_width.  Clamped to [n_min, n_max].
        """
        binary = self._binarize(region)
        col_sums = np.sum(binary > 0, axis=0)
        nonzero_cols = np.where(col_sums > 0)[0]
        if len(nonzero_cols) == 0:
            return n_min
        active_width = int(nonzero_cols[-1]) + 1
        count = round(active_width / char_width)
        return max(n_min, min(n_max, count))

    # ── Segmentation strategies ───────────────────────────────────

    def _boxes_projection(self, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Vertical projection histogram.

        1. Invert + Otsu-threshold so characters are white on black.
        2. Sum white pixels in each column → projection array.
        3. Runs of non-zero columns = character spans.
        4. For each span, tighten the vertical bounds by row projection.
        """
        binary = self._binarize(gray)
        proj = np.sum(binary > 0, axis=0)  # column sums

        # Find contiguous non-zero spans
        spans = []
        in_char = False
        start = 0
        for x, val in enumerate(proj):
            if val > 0 and not in_char:
                in_char = True
                start = x
            elif val == 0 and in_char:
                in_char = False
                if x - start >= self.min_char_width:
                    spans.append((start, x))
        if in_char and len(proj) - start >= self.min_char_width:
            spans.append((start, len(proj)))

        boxes = []
        for x1, x2 in spans:
            col_slice = binary[:, x1:x2]
            row_proj = np.sum(col_slice > 0, axis=1)
            nonzero_rows = np.where(row_proj > 0)[0]
            if len(nonzero_rows) == 0:
                continue
            y1 = int(nonzero_rows[0])
            y2 = int(nonzero_rows[-1]) + 1
            if y2 - y1 < self.min_char_height:
                continue
            boxes.append((x1, y1, x2 - x1, y2 - y1))

        return boxes

    def _boxes_contour(self, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        """OpenCV external-contour bounding boxes, sorted left-to-right."""
        binary = self._binarize(gray)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w >= self.min_char_width and h >= self.min_char_height:
                boxes.append((x, y, w, h))
        boxes.sort(key=lambda b: b[0])
        return boxes

    def _boxes_fixed_width(
        self, gray: np.ndarray, char_width: int, char_count: int = 0
    ) -> list[tuple[int, int, int, int]]:
        """
        Slice the ROI into equal-width columns.

        Priority:
          char_count > 0  →  divide roi_width / char_count (exact, no rounding gaps)
          char_count == 0 →  use char_width px; 0 = estimate as height * 0.6
        """
        h, w = gray.shape
        if char_count > 0:
            n_chars = char_count
        else:
            cw = char_width if char_width > 0 else max(1, int(h * 0.6))
            n_chars = max(1, round(w / cw))

        actual_cw = w / n_chars  # float — no integer rounding errors

        binary = self._binarize(gray)
        boxes = []
        for i in range(n_chars):
            x1 = int(i * actual_cw)
            x2 = int((i + 1) * actual_cw)
            x2 = min(x2, w)
            if x2 - x1 < self.min_char_width:
                continue

            col_slice = binary[:, x1:x2]
            row_proj = np.sum(col_slice > 0, axis=1)
            nonzero_rows = np.where(row_proj > 0)[0]
            if len(nonzero_rows) == 0:
                continue
            y1 = int(nonzero_rows[0])
            y2 = int(nonzero_rows[-1]) + 1
            if y2 - y1 < self.min_char_height:
                continue
            boxes.append((x1, y1, x2 - x1, y2 - y1))

        return boxes

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()

    @staticmethod
    def _binarize(gray: np.ndarray) -> np.ndarray:
        """
        Otsu threshold producing WHITE characters on BLACK background.

        Assumes characters are brighter than the background, which is the
        typical case for game HUD text (light digits on dark bg).
        If your text is darker than the background, enable the Invert filter
        in the ROI filter settings to flip it before segmentation.
        """
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary

    def _make_char(
        self, gray: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> SegmentedChar:
        """Crop, pad, resize to TARGET_SIZE x TARGET_SIZE, normalise."""
        x, y, w, h = bbox
        crop = gray[y : y + h, x : x + w]

        padded = cv2.copyMakeBorder(
            crop,
            self.padding, self.padding, self.padding, self.padding,
            cv2.BORDER_CONSTANT,
            value=255,
        )
        resized = cv2.resize(
            padded,
            (self.TARGET_SIZE, self.TARGET_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        normalized = resized.astype(np.float32) / 255.0

        return SegmentedChar(image=normalized, bbox=bbox, original=crop)
