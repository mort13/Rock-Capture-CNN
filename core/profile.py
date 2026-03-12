"""
Core data models for Rock Capture CNN.
Profile, ROIDefinition, and FilterSettings with JSON serialization.
"""

from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class FilterSettings:
    """Per-ROI image filter configuration."""
    brightness: int = 0
    contrast: int = 0
    threshold: int = 127
    threshold_enabled: bool = False
    grayscale: bool = True
    invert: bool = False
    channel: str = "none"  # "none", "red", "green", "blue"

    def to_dict(self) -> dict:
        return {
            "brightness": self.brightness,
            "contrast": self.contrast,
            "threshold": self.threshold,
            "threshold_enabled": self.threshold_enabled,
            "grayscale": self.grayscale,
            "invert": self.invert,
            "channel": self.channel,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FilterSettings":
        return cls(
            brightness=d.get("brightness", 0),
            contrast=d.get("contrast", 0),
            threshold=d.get("threshold", 127),
            threshold_enabled=d.get("threshold_enabled", False),
            grayscale=d.get("grayscale", True),
            invert=d.get("invert", False),
            channel=d.get("channel", "none"),
        )


@dataclass
class ROIDefinition:
    """A rectangular region defined relative to the anchor's top-left corner."""
    name: str
    x_offset: int = 0
    y_offset: int = 0
    width: int = 80
    height: int = 24
    filters: FilterSettings = field(default_factory=FilterSettings)
    # Segmentation settings
    seg_mode: str = "projection"  # "projection", "contour", "fixed_width"
    char_width: int = 0           # fixed_width fallback: px per char (0 = guess from height)
    char_count: int = 0           # fixed_width primary: exact number of chars (0 = use char_width)
    # Prediction filter: only these chars are considered valid predictions for this ROI.
    # Empty string = allow all classes the model knows.
    allowed_chars: str = ""
    # Format pattern: 'x' = predicted char, any other char = literal inserted without CNN.
    # e.g. "xx%" -> 2 predicted digits + literal %, "xxx.xx" -> 3 digits + literal . + 2 digits
    # Advanced syntax: {n} = exactly n chars, {1,3} = variable 1-3 chars (projection used).
    # When set, the x-count (or {n} totals) override char_count for segmentation.
    format_pattern: str = ""
    # Pixel width of the decimal-point glyph in the image.
    # Used when the format pattern contains '.' so the segmenter can skip over it correctly.
    # 0 = estimate as char_width // 4.
    dot_width: int = 0
    # Whether this ROI is active. Disabled ROIs are skipped in the pipeline and labeler.
    enabled: bool = True
    # Recognition mode: "cnn" = digit/char segmentation + CNN, "template" = template matching.
    recognition_mode: str = "cnn"
    # Directory (relative to data/) containing template images for template matching mode.
    # Each .png/.jpg in the folder represents one word; the filename (minus extension) is the label.
    template_dir: str = ""
    # Column order in the exported CSV. 0 = append after all explicitly-ordered columns.
    csv_index: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "x_offset": self.x_offset,
            "y_offset": self.y_offset,
            "width": self.width,
            "height": self.height,
            "filters": self.filters.to_dict(),
            "seg_mode": self.seg_mode,
            "char_width": self.char_width,
            "char_count": self.char_count,
            "allowed_chars": self.allowed_chars,
            "format_pattern": self.format_pattern,
            "dot_width": self.dot_width,
            "enabled": self.enabled,
            "recognition_mode": self.recognition_mode,
            "template_dir": self.template_dir,
            "csv_index": self.csv_index,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ROIDefinition":
        return cls(
            name=d["name"],
            x_offset=d.get("x_offset", 0),
            y_offset=d.get("y_offset", 0),
            width=d.get("width", 80),
            height=d.get("height", 24),
            filters=FilterSettings.from_dict(d.get("filters", {})),
            seg_mode=d.get("seg_mode", "projection"),
            char_width=d.get("char_width", 0),
            char_count=d.get("char_count", 0),
            allowed_chars=d.get("allowed_chars", ""),
            format_pattern=d.get("format_pattern", ""),
            dot_width=d.get("dot_width", 0),
            enabled=d.get("enabled", True),
            recognition_mode=d.get("recognition_mode", "cnn"),
            template_dir=d.get("template_dir", ""),
            csv_index=d.get("csv_index", 0),
        )


@dataclass
class Profile:
    """A named configuration: anchor template + ROIs + filter settings."""
    name: str
    anchor_template_path: str = ""
    anchor_match_threshold: float = 0.7
    rois: list[ROIDefinition] = field(default_factory=list)
    search_region: dict = field(
        default_factory=lambda: {"x": 0, "y": 0, "w": 800, "h": 600}
    )
    anchor_roi: dict = field(
        default_factory=lambda: {}  # empty = search full frame; {"x":, "y":, "w":, "h":} relative to search_region
    )
    monitor_index: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "anchor_template_path": self.anchor_template_path,
            "anchor_match_threshold": self.anchor_match_threshold,
            "rois": [r.to_dict() for r in self.rois],
            "search_region": self.search_region,
            "anchor_roi": self.anchor_roi,
            "monitor_index": self.monitor_index,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Profile":
        return cls(
            name=d["name"],
            anchor_template_path=d.get("anchor_template_path", ""),
            anchor_match_threshold=d.get("anchor_match_threshold", 0.7),
            rois=[ROIDefinition.from_dict(r) for r in d.get("rois", [])],
            search_region=d.get("search_region", {"x": 0, "y": 0, "w": 800, "h": 600}),
            anchor_roi=d.get("anchor_roi", {}),
            monitor_index=d.get("monitor_index", 0),
            # model_path and char_classes intentionally dropped — model is global, not per-profile
        )

    def save(self, profiles_dir: Path) -> None:
        """Save profile as JSON to profiles_dir/{self.name}.json."""
        profiles_dir.mkdir(parents=True, exist_ok=True)
        path = profiles_dir / f"{self.name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Profile":
        """Load profile from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @staticmethod
    def list_profiles(profiles_dir: Path) -> list[str]:
        """List available profile names from the profiles directory."""
        if not profiles_dir.exists():
            return []
        return [p.stem for p in sorted(profiles_dir.glob("*.json"))]
