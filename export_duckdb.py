"""
Export captured session JSON files into a DuckDB database.

The export is driven by a declarative field specification that controls:
- Which JSON fields are extracted
- How *_int / *_dec pairs are recombined into a single number
- Which confidence scores are kept

Run:
    python export_duckdb.py                           # default paths
    python export_duckdb.py --captures data/captures --db data/scans.duckdb
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import duckdb
except ImportError:
    print("duckdb is required:  pip install duckdb", file=sys.stderr)
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
#  Field specification
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Field:
    """A single output column derived from the capture JSON.

    source:
        JSON key inside the scan dict (e.g. "mass", "instability_int").
        For composite fields supply the shared prefix (e.g. "instability").
    column:
        Output column name in the DB.  Defaults to *source*.
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
    material_min_confidence: bool = True  # emit a min_confidence per material row
    aliases: dict[str, str] = field(default_factory=dict)  # raw OCR name -> display name


# ─── Default spec matching the mole HUD profile ──────────────────

DEFAULT_SPEC = ExportSpec(
    scan_fields=[
        Field("deposit",    "deposit",     "text",      confidence=True),
        Field("mass",       "mass",        "int",       confidence=True),
        Field("resistance", "resistance",  "int",       confidence=True),
        Field("instability","instability", "composite", confidence=True),
        Field("volume",     "volume",      "composite", confidence=True),
    ],
    material_fields=[
        MaterialField("type",   "type",   "text"),
        MaterialField("amount", "amount", "composite"),
        MaterialField("quality","quality", "int"),
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
    """Return the {value, confidence} dict for *key*, or None."""
    return scan.get(key)


def _extract_value(scan: dict, f: Field) -> Any:
    """Pull the value for a Field from the scan dict."""
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
    return raw  # text


def _extract_confidence(scan: dict, f: Field) -> float | None:
    """Pull the confidence(s) for a Field.  For composite, return the min."""
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
    """Pull a value from one composition entry."""
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
    """Return the minimum confidence across all entries in a material dict."""
    confs = []
    for v in mat.values():
        if isinstance(v, dict) and "confidence" in v:
            confs.append(v["confidence"])
    return min(confs) if confs else None


# ═══════════════════════════════════════════════════════════════════
#  Row building
# ═══════════════════════════════════════════════════════════════════

def _build_scan_row(
    session: dict, capture: dict, spec: ExportSpec
) -> dict:
    scan = capture.get("scan", {})
    row: dict[str, Any] = {
        "session_id": session.get("session_id"),
        "user": session.get("source", {}).get("user"),
        "org": session.get("source", {}).get("org"),
        "timestamp": capture.get("timestamp"),
        "capture_id": capture.get("capture_id"),
        "cluster_id": capture.get("cluster_id"),
    }
    for f in spec.scan_fields:
        row[f.column] = _extract_value(scan, f)
        if f.confidence:
            row[f"{f.column}_conf"] = _extract_confidence(scan, f)
    return row


def _build_material_rows(
    capture: dict, spec: ExportSpec
) -> list[dict]:
    scan = capture.get("scan", {})
    composition = scan.get("composition", [])
    rows = []
    for idx, mat in enumerate(composition):
        row: dict[str, Any] = {
            "capture_id": capture.get("capture_id"),
            "mat_index": idx,
        }
        for mf in spec.material_fields:
            val = _extract_material_value(mat, mf)
            if mf.kind == "text" and isinstance(val, str):
                val = spec.aliases.get(val, val)
            row[mf.column] = val
        if spec.material_min_confidence:
            row["min_confidence"] = _material_min_confidence(mat)
        rows.append(row)
    return rows


# ═══════════════════════════════════════════════════════════════════
#  DuckDB export
# ═══════════════════════════════════════════════════════════════════

def _create_tables(con: duckdb.DuckDBPyConnection, spec: ExportSpec) -> None:
    # ── scans table ───────────────────────────────────────────────
    cols = [
        "session_id  TEXT",
        "user        TEXT",
        "org         TEXT",
        "timestamp   TEXT",
        "capture_id  TEXT PRIMARY KEY",
        "cluster_id  INTEGER",
    ]
    for f in spec.scan_fields:
        sql_type = {"text": "TEXT", "int": "INTEGER", "float": "DOUBLE",
                     "composite": "DOUBLE"}[f.kind]
        cols.append(f"{f.column}  {sql_type}")
        if f.confidence:
            cols.append(f"{f.column}_conf  DOUBLE")

    con.execute(f"CREATE OR REPLACE TABLE scans (\n  {',\n  '.join(cols)}\n)")

    # ── compositions table ────────────────────────────────────────
    mat_cols = [
        "capture_id      TEXT REFERENCES scans(capture_id)",
        "mat_index        INTEGER",
    ]
    for mf in spec.material_fields:
        sql_type = {"text": "TEXT", "int": "INTEGER", "float": "DOUBLE",
                     "composite": "DOUBLE"}[mf.kind]
        mat_cols.append(f"{mf.column}  {sql_type}")
    if spec.material_min_confidence:
        mat_cols.append("min_confidence  DOUBLE")

    con.execute(
        f"CREATE OR REPLACE TABLE compositions (\n  {',\n  '.join(mat_cols)},\n"
        f"  PRIMARY KEY (capture_id, mat_index)\n)"
    )

    # ── aliases table ─────────────────────────────────────────────
    con.execute(
        "CREATE OR REPLACE TABLE aliases (\n"
        "  raw      TEXT PRIMARY KEY,\n"
        "  display  TEXT NOT NULL\n)"
    )
    for raw, display in spec.aliases.items():
        con.execute("INSERT OR REPLACE INTO aliases VALUES (?, ?)", [raw, display])


def export(
    captures_dir: Path,
    db_path: Path,
    spec: ExportSpec | None = None,
) -> None:
    spec = spec or DEFAULT_SPEC
    json_files = sorted(captures_dir.glob("session_*.json"))
    if not json_files:
        print(f"No session files found in {captures_dir}")
        return

    con = duckdb.connect(str(db_path))
    _create_tables(con, spec)

    scan_count = 0
    mat_count = 0

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            session = json.load(f)

        for capture in session.get("captures", []):
            scan_row = _build_scan_row(session, capture, spec)
            placeholders = ", ".join(["?"] * len(scan_row))
            col_names = ", ".join(scan_row.keys())
            con.execute(
                f"INSERT OR REPLACE INTO scans ({col_names}) VALUES ({placeholders})",
                list(scan_row.values()),
            )
            scan_count += 1

            for mat_row in _build_material_rows(capture, spec):
                placeholders = ", ".join(["?"] * len(mat_row))
                col_names = ", ".join(mat_row.keys())
                con.execute(
                    f"INSERT OR REPLACE INTO compositions ({col_names}) VALUES ({placeholders})",
                    list(mat_row.values()),
                )
                mat_count += 1

    con.close()
    print(f"Exported {scan_count} scans, {mat_count} material rows → {db_path}")


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Export session JSONs to DuckDB")
    parser.add_argument(
        "--captures", type=Path,
        default=Path(__file__).parent / "data" / "captures",
        help="Directory containing session_*.json files",
    )
    parser.add_argument(
        "--db", type=Path,
        default=Path(__file__).parent / "data" / "scans.duckdb",
        help="Output DuckDB file path",
    )
    args = parser.parse_args()
    export(args.captures, args.db)


if __name__ == "__main__":
    main()
