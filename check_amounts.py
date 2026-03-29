"""
Check that material amounts per scan sum to roughly 100%.

Usage:
    python check_amounts.py
    python check_amounts.py --captures path/to/captures --tolerance 2.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _amount(mat: dict) -> float | None:
    int_entry = mat.get("amount_int")
    dec_entry = mat.get("amount_dec")
    if int_entry is None and dec_entry is None:
        return None
    int_str = int_entry["value"] if int_entry else "0"
    dec_str = dec_entry["value"] if dec_entry else "0"
    try:
        return float(f"{int_str}.{dec_str}")
    except ValueError:
        return None


def check(captures_dir: Path, tolerance: float = 3.0) -> None:
    json_files = sorted(captures_dir.glob("session_*.json"))
    if not json_files:
        print(f"No session files found in {captures_dir}")
        return

    total_captures = 0
    bad_captures = 0

    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            session = json.load(f)

        for capture in session.get("captures", []):
            total_captures += 1
            scan = capture.get("scan", {})
            composition = scan.get("composition", [])

            amounts = []
            for mat in composition:
                name_entry = mat.get("name")
                if isinstance(name_entry, dict) and name_entry.get("value") == "none":
                    continue
                a = _amount(mat)
                if a is not None:
                    amounts.append(a)

            if not amounts:
                continue

            total = sum(amounts)
            if abs(total - 100.0) > tolerance:
                bad_captures += 1
                capture_id = capture.get("capture_id", "unknown")
                def _fmt(mat: dict) -> str:
                    name = mat.get("name", {}).get("value", "?") if isinstance(mat.get("name"), dict) else "?"
                    a = _amount(mat)
                    return f"{name}={a:.1f}%" if a is not None else f"{name}=?"
                mat_list = ", ".join(
                    _fmt(mat)
                    for mat in composition
                    if not (isinstance(mat.get("name"), dict) and mat["name"].get("value") == "none")
                )
                print(f"  [{jf.name}] capture {capture_id}: missing={100-total:.2f}%  ({mat_list})")

    print(
        f"\n{bad_captures} / {total_captures} captures have amounts outside "
        f"100 ± {tolerance}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check material amount sums in session JSONs")
    parser.add_argument(
        "--captures", type=Path,
        default=Path(__file__).parent.parent / "Rock Capture Database" / "captures",
        help="Directory containing session_*.json files",
    )
    parser.add_argument(
        "--tolerance", type=float, default=3.0,
        help="Allowed deviation from 100%% (default: 3.0)",
    )
    args = parser.parse_args()
    check(args.captures, args.tolerance)


if __name__ == "__main__":
    main()
