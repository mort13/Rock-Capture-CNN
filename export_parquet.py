"""
Export captured session JSON files into two Parquet files.

    scans.parquet        — one row per capture (session metadata + scan fields)
    compositions.parquet — one row per material entry, joined by capture_id

The export is driven by the same declarative field specification used in
export_duckdb.py.

Run:
    python export_parquet.py                           # default paths
    python export_parquet.py --captures data/captures --out data/
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("pyarrow is required:  pip install pyarrow", file=sys.stderr)
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
#  Field specification  (mirrors export_duckdb.py)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Field:
    """A single output column derived from the capture JSON.

    source:
        JSON key inside the scan dict (e.g. "mass", "instability_int").
        For composite fields supply the shared prefix (e.g. "instability").
    column:
        Output column name.  Defaults to *source*.
    kind:
        "text"      — raw string value
        "int"       — cast to integer
        "float"     — cast to float
        "composite" — recombine <prefix>_int + <prefix>_dec into one float
    confidence:
        Whether to emit an extra <column>_conf column.
    """
    source: str
    column: str = ""
    kind: str = "text"
    confidence: bool = False

    def __post_init__(self):
        if not self.column:
            self.column = self.source


@dataclass
class MaterialField:
    """A column extracted from each element of the composition array."""
    source: str
    column: str = ""
    kind: str = "text"

    def __post_init__(self):
        if not self.column:
            self.column = self.source


@dataclass
class ExportSpec:
    """Full specification of what to export and how."""
    scan_fields: list[Field] = field(default_factory=list)
    material_fields: list[MaterialField] = field(default_factory=list)
    material_min_confidence: bool = True
    aliases: dict[str, str] = field(default_factory=dict)


# ─── Default spec matching the mole HUD profile ──────────────────

DEFAULT_SPEC = ExportSpec(
    scan_fields=[
        Field("deposit_name", "deposit",     "text",      confidence=True),
        Field("mass",        "mass",        "int",       confidence=True),
        Field("resistance",  "resistance",  "int",       confidence=True),
        Field("instability", "instability", "composite", confidence=True),
        Field("volume",      "volume",      "composite", confidence=True),
    ],
    material_fields=[
        MaterialField("name",    "type",    "text"),
        MaterialField("amount",  "amount",  "composite"),
        MaterialField("quality", "quality", "int"),
    ],
    material_min_confidence=True,
    aliases={
        "stileron":        "Stileron",
        "taranite":        "Taranite",
        "inert_materials": "Inert Materials",
    },
)


# ═══════════════════════════════════════════════════════════════════
#  Value extraction helpers
# ═══════════════════════════════════════════════════════════════════

def _get_entry(scan: dict, key: str) -> dict | None:
    return scan.get(key)


def _extract_value(scan: dict, f: Field) -> Any:
    if f.kind == "composite":
        int_entry = _get_entry(scan, f"{f.source}_int")
        dec_entry = _get_entry(scan, f"{f.source}_dec")
        int_str = int_entry["value"] if int_entry else "0"
        dec_str = dec_entry["value"] if dec_entry else "0"
        try:
            return float(f"{int_str}.{dec_str}")
        except ValueError:
            return None
    entry = _get_entry(scan, f.source)
    if entry is None:
        return None
    raw = entry["value"]
    if f.kind == "int":
        try:
            return int(raw)
        except (ValueError, TypeError):
            return None
    if f.kind == "float":
        try:
            return float(raw)
        except (ValueError, TypeError):
            return None
    return raw


def _extract_confidence(scan: dict, f: Field) -> float | None:
    if f.kind == "composite":
        confs = []
        for suffix in ("_int", "_dec"):
            e = _get_entry(scan, f"{f.source}{suffix}")
            if e and "confidence" in e:
                confs.append(e["confidence"])
        return min(confs) if confs else None
    entry = _get_entry(scan, f.source)
    if entry and "confidence" in entry:
        return entry["confidence"]
    return None


def _extract_material_value(mat: dict, mf: MaterialField) -> Any:
    if mf.kind == "composite":
        int_entry = mat.get(f"{mf.source}_int")
        dec_entry = mat.get(f"{mf.source}_dec")
        int_str = int_entry["value"] if int_entry else "0"
        dec_str = dec_entry["value"] if dec_entry else "0"
        try:
            return float(f"{int_str}.{dec_str}")
        except ValueError:
            return None
    entry = mat.get(mf.source)
    if entry is None:
        return None
    raw = entry["value"]
    if mf.kind == "int":
        try:
            return int(raw)
        except (ValueError, TypeError):
            return None
    if mf.kind == "float":
        try:
            return float(raw)
        except (ValueError, TypeError):
            return None
    return raw


def _material_min_confidence(mat: dict) -> float | None:
    confs = [
        v["confidence"]
        for v in mat.values()
        if isinstance(v, dict) and "confidence" in v
    ]
    return min(confs) if confs else None


# ═══════════════════════════════════════════════════════════════════
#  Row building
# ═══════════════════════════════════════════════════════════════════

def _build_scan_row(session: dict, capture: dict, spec: ExportSpec) -> dict:
    scan = capture.get("scan", {})
    loc = capture.get("location") or {}
    row: dict[str, Any] = {
        "session_id":   session.get("session_id"),
        "user":         session.get("source", {}).get("user"),
        "org":          session.get("source", {}).get("org"),
        "timestamp":    capture.get("timestamp"),
        "capture_id":   capture.get("capture_id"),
        "cluster_id":   capture.get("cluster_id"),
        "system":       loc.get("system"),
        "gravity_well": loc.get("gravity_well"),
        "region":       loc.get("region"),
        "place":        loc.get("place"),
    }
    for f in spec.scan_fields:
        row[f.column] = _extract_value(scan, f)
        if f.confidence:
            row[f"{f.column}_conf"] = _extract_confidence(scan, f)
    return row


def _build_material_rows(capture: dict, spec: ExportSpec) -> list[dict]:
    scan = capture.get("scan", {})
    composition = scan.get("composition", [])
    
    # Extract scan volume for material volume calculation
    scan_volume = _extract_value(scan, Field("volume", "volume", "composite"))
    
    rows = []
    for idx, mat in enumerate(composition):
        # Skip rows where the material name is missing, null, empty, "none", or "?"
        name_entry = mat.get("name")
        name_value = name_entry.get("value") if isinstance(name_entry, dict) else None
        if not name_value or name_value in ("none", "?"):
            continue

        row: dict[str, Any] = {
            "capture_id": capture.get("capture_id"),
            "mat_index":  idx,
        }
        for mf in spec.material_fields:
            val = _extract_material_value(mat, mf)
            if mf.kind == "text" and isinstance(val, str):
                val = spec.aliases.get(val, val)
            row[mf.column] = val
        
        # Calculate material volume from percentage and scan volume
        if scan_volume is not None:
            material_amount = _extract_material_value(mat, MaterialField("amount", "amount", "composite"))
            if material_amount is not None:
                # Assume amount is a percentage (0-100), calculate actual volume
                material_volume = (material_amount / 100.0) * scan_volume
                row["material_volume"] = material_volume
            else:
                row["material_volume"] = None
        else:
            row["material_volume"] = None
        
        if spec.material_min_confidence:
            row["min_confidence"] = _material_min_confidence(mat)
        rows.append(row)
    return rows


# ═══════════════════════════════════════════════════════════════════
#  PyArrow schema helpers
# ═══════════════════════════════════════════════════════════════════

def _scan_schema(spec: ExportSpec) -> pa.Schema:
    fields = [
        pa.field("session_id",   pa.string()),
        pa.field("user",         pa.string()),
        pa.field("org",          pa.string()),
        pa.field("timestamp",    pa.string()),
        pa.field("capture_id",   pa.string()),
        pa.field("cluster_id",   pa.string()),
        pa.field("system",       pa.string()),
        pa.field("gravity_well", pa.string()),
        pa.field("region",       pa.string()),
        pa.field("place",        pa.string()),
    ]
    type_map = {"text": pa.string(), "int": pa.int64(),
                "float": pa.float64(), "composite": pa.float64()}
    for f in spec.scan_fields:
        fields.append(pa.field(f.column, type_map[f.kind]))
        if f.confidence:
            fields.append(pa.field(f"{f.column}_conf", pa.float64()))
    return pa.schema(fields)


def _material_schema(spec: ExportSpec) -> pa.Schema:
    fields = [
        pa.field("capture_id", pa.string()),
        pa.field("mat_index",  pa.int64()),
    ]
    type_map = {"text": pa.string(), "int": pa.int64(),
                "float": pa.float64(), "composite": pa.float64()}
    for mf in spec.material_fields:
        fields.append(pa.field(mf.column, type_map[mf.kind]))
    # Add calculated material volume field
    fields.append(pa.field("material_volume", pa.float64()))
    if spec.material_min_confidence:
        fields.append(pa.field("min_confidence", pa.float64()))
    return pa.schema(fields)


def _rows_to_table(rows: list[dict], schema: pa.Schema) -> pa.Table:
    if not rows:
        return pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)
    columns = {f.name: [row.get(f.name) for row in rows] for f in schema}
    arrays = {name: pa.array(vals, type=schema.field(name).type) for name, vals in columns.items()}
    return pa.table(arrays, schema=schema)


# ═══════════════════════════════════════════════════════════════════
#  Main export
# ═══════════════════════════════════════════════════════════════════

def export(
    captures_dir: Path,
    out_dir: Path,
    spec: ExportSpec | None = None,
) -> None:
    spec = spec or DEFAULT_SPEC
    json_files = sorted(captures_dir.glob("session_*.json"))
    if not json_files:
        print(f"No session files found in {captures_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    scan_rows: list[dict] = []
    mat_rows: list[dict] = []

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            session = json.load(f)

        for capture in session.get("captures", []):
            scan_rows.append(_build_scan_row(session, capture, spec))
            mat_rows.extend(_build_material_rows(capture, spec))

    scan_schema = _scan_schema(spec)
    mat_schema  = _material_schema(spec)

    scans_path = out_dir / "scans.parquet"
    comps_path = out_dir / "compositions.parquet"

    pq.write_table(_rows_to_table(scan_rows, scan_schema), scans_path, compression="snappy")
    pq.write_table(_rows_to_table(mat_rows,  mat_schema),  comps_path, compression="snappy")

    print(
        f"Exported {len(scan_rows)} scans → {scans_path}\n"
        f"Exported {len(mat_rows)} material rows → {comps_path}"
    )


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Export session JSONs to Parquet")
    parser.add_argument(
        "--captures", type=Path,
        default=Path(__file__).parent.parent / "Rock Capture Database" / "captures",
        help="Directory containing session_*.json files",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).parent.parent / "Rock Capture Database",
        help="Output directory for scans.parquet and compositions.parquet",
    )
    args = parser.parse_args()
    export(args.captures, args.out)


if __name__ == "__main__":
    main()
