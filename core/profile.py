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
class AnchorPoint:
    """A template-matched reference point for resolution-independent positioning."""
    name: str
    template_path: str = ""           # relative to data/anchors/
    match_threshold: float = 0.7
    ref_x: float = 0                   # expected x in the reference frame
    ref_y: float = 0                   # expected y in the reference frame
    search_region: dict | None = None  # {"x": x, "y": y, "width": w, "height": h} in ref frame

    def to_dict(self) -> dict:
        result = {
            "name": self.name,
            "template_path": self.template_path,
            "match_threshold": self.match_threshold,
            "ref_x": self.ref_x,
            "ref_y": self.ref_y,
        }
        if self.search_region:
            result["search_region"] = self.search_region
        return result

    @classmethod
    def from_dict(cls, d: dict) -> AnchorPoint:
        return cls(
            name=d["name"],
            template_path=d.get("template_path", ""),
            match_threshold=d.get("match_threshold", 0.7),
            ref_x=d.get("ref_x", 0),
            ref_y=d.get("ref_y", 0),
            search_region=d.get("search_region", None),
        )


@dataclass
class ROIDefinition:
    """A rectangular region of interest.

    Legacy mode (single anchor): positioned via *x_offset / y_offset* from anchor.
    Multi-anchor mode: positioned via *ref_x / ref_y* in the reference frame,
    optionally refined by a named *sub_anchor*.
    """
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
    # --- Multi-anchor positioning fields ---
    # Absolute position in the reference frame (used when Profile.uses_multi_anchor).
    ref_x: float = 0.0
    ref_y: float = 0.0
    # Name of a sub-anchor for local refinement (empty = main transform only).
    sub_anchor: str = ""

    def to_dict(self) -> dict:
        d = {
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
        # Only write multi-anchor fields when they carry information
        if self.ref_x or self.ref_y:
            d["ref_x"] = self.ref_x
            d["ref_y"] = self.ref_y
        if self.sub_anchor:
            d["sub_anchor"] = self.sub_anchor
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ROIDefinition:
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
            ref_x=d.get("ref_x", 0.0),
            ref_y=d.get("ref_y", 0.0),
            sub_anchor=d.get("sub_anchor", ""),
        )


@dataclass
class Profile:
    """A named configuration: anchor(s) + ROIs + filter settings.

    Supports two modes:
    * **Legacy (single anchor):** one *anchor_template_path* + ROIs with
      *x_offset / y_offset* pixel offsets.
    * **Multi-anchor:** 2-3 *anchors* define a reference coordinate system.
      An affine / similarity transform maps reference positions to the
      current frame so that ROI placement is resolution-independent.
      Optional *sub_anchors* provide local refinement.
    """
    name: str
    # --- Legacy single-anchor fields (kept for backward compat) ---
    anchor_template_path: str = ""
    anchor_match_threshold: float = 0.7
    anchor_roi: dict = field(
        default_factory=lambda: {}
    )
    # --- Common fields ---
    rois: list[ROIDefinition] = field(default_factory=list)
    search_region: dict = field(
        default_factory=lambda: {"x": 0, "y": 0, "w": 800, "h": 600}
    )
    monitor_index: int = 0
    # --- Multi-anchor fields ---
    anchors: list[AnchorPoint] = field(default_factory=list)
    sub_anchors: list[AnchorPoint] = field(default_factory=list)

    @property
    def uses_multi_anchor(self) -> bool:
        """True when the profile uses the new multi-anchor positioning."""
        return len(self.anchors) >= 2

    def get_sub_anchor(self, name: str) -> AnchorPoint | None:
        for sa in self.sub_anchors:
            if sa.name == name:
                return sa
        return None

    def to_dict(self) -> dict:
        d: dict = {
            "name": self.name,
            "rois": [r.to_dict() for r in self.rois],
            "search_region": self.search_region,
            "monitor_index": self.monitor_index,
        }
        # Legacy fields
        if self.anchor_template_path:
            d["anchor_template_path"] = self.anchor_template_path
        if self.anchor_match_threshold != 0.7:
            d["anchor_match_threshold"] = self.anchor_match_threshold
        if self.anchor_roi:
            d["anchor_roi"] = self.anchor_roi
        # Multi-anchor fields
        if self.anchors:
            d["anchors"] = [a.to_dict() for a in self.anchors]
        if self.sub_anchors:
            d["sub_anchors"] = [a.to_dict() for a in self.sub_anchors]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Profile:
        return cls(
            name=d["name"],
            anchor_template_path=d.get("anchor_template_path", ""),
            anchor_match_threshold=d.get("anchor_match_threshold", 0.7),
            anchor_roi=d.get("anchor_roi", {}),
            rois=[ROIDefinition.from_dict(r) for r in d.get("rois", [])],
            search_region=d.get("search_region", {"x": 0, "y": 0, "w": 800, "h": 600}),
            monitor_index=d.get("monitor_index", 0),
            anchors=[AnchorPoint.from_dict(a) for a in d.get("anchors", [])],
            sub_anchors=[AnchorPoint.from_dict(a) for a in d.get("sub_anchors", [])],
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


@dataclass
class ROIRef:
    """A leaf reference to a single ROI from one profile.

    ``key`` is the output JSON key; if empty the ROI name is used.
    """

    profile: str
    roi: str
    key: str = ""

    def to_dict(self) -> dict:
        d: dict = {"profile": self.profile, "roi": self.roi}
        if self.key:
            d["key"] = self.key
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ROIRef":
        return cls(profile=d["profile"], roi=d["roi"], key=d.get("key", ""))


@dataclass
class SchemaNode:
    """A recursive tree node in the output schema.

    ``type`` controls how children are serialized:
    - "object" — children become key/value pairs in a dict
    - "array"  — children become elements of a JSON array
    """

    key: str
    type: str = "object"  # "object" | "array"
    children: list = field(default_factory=list)  # list[ROIRef | SchemaNode]

    def to_dict(self) -> dict:
        return {"key": self.key, "type": self.type, "children": [c.to_dict() for c in self.children]}

    @classmethod
    def from_dict(cls, d: dict) -> "SchemaNode":
        children = []
        for c in d.get("children", []):
            children.append(ROIRef.from_dict(c) if "profile" in c else SchemaNode.from_dict(c))
        return cls(key=d["key"], type=d.get("type", "object"), children=children)


@dataclass
class HUDProfile:
    """A named snapshot of all small profiles for a specific ship / HUD layout.

    Storing one HUD profile bundles the anchor, search-region, and ROI settings
    of every individual profile so that switching ships restores the complete setup.

    The optional ``output_schema`` defines how raw profile results are structured
    when committed to the session JSON.  Profiles not covered by any group are
    collected under a fallback ``"misc"`` singleton group so no data is lost.
    """

    name: str
    profiles: dict = field(default_factory=dict)  # profile_name -> profile.to_dict()
    output_schema: list[SchemaNode] = field(default_factory=list)

    def to_dict(self) -> dict:
        d: dict = {"name": self.name, "profiles": self.profiles}
        if self.output_schema:
            d["output_schema"] = [n.to_dict() for n in self.output_schema]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "HUDProfile":
        schema = [SchemaNode.from_dict(n) for n in d.get("output_schema", [])]
        return cls(name=d["name"], profiles=d.get("profiles", {}), output_schema=schema)

    def save(self, hud_profiles_dir: Path) -> None:
        """Save HUD profile as JSON to hud_profiles_dir/{self.name}.json."""
        hud_profiles_dir.mkdir(parents=True, exist_ok=True)
        path = hud_profiles_dir / f"{self.name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "HUDProfile":
        """Load a HUD profile from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @staticmethod
    def list_profiles(hud_profiles_dir: Path) -> list[str]:
        """List available HUD profile names."""
        if not hud_profiles_dir.exists():
            return []
        return [p.stem for p in sorted(hud_profiles_dir.glob("*.json"))]
