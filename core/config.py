"""Application configuration for Rock Capture CNN."""

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class AppConfig:
    """Application-level configuration loaded from data/config.json."""
    tool_version: str = "ocr_tool_0.3"
    user: str = ""
    org: str = ""
    system: str = ""
    gravity_well: str = ""
    active_ship_profile: str = ""  # last selected HUD profile name

    def to_dict(self) -> dict:
        return {
            "tool_version": self.tool_version,
            "user": self.user,
            "org": self.org,
            "system": self.system,
            "gravity_well": self.gravity_well,
            "active_ship_profile": self.active_ship_profile,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AppConfig":
        return cls(
            tool_version=d.get("tool_version", "ocr_tool_0.3"),
            user=d.get("user", ""),
            org=d.get("org", ""),
            system=d.get("system", ""),
            gravity_well=d.get("gravity_well", ""),
            active_ship_profile=d.get("active_ship_profile", ""),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        if not path.exists():
            config = cls()
            config.save(path)
            return config
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
