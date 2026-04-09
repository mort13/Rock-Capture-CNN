#!/usr/bin/env python3
"""
Convert all session files to use full UUIDv4 session IDs instead of hex[:8].
"""

import json
import uuid
from pathlib import Path

def convert_session_files():
    captures_dir = Path(__file__).parent.parent / "Rock Capture Database" / "captures"
    
    session_files = sorted(captures_dir.glob("session_*.json"))
    print(f"Found {len(session_files)} session files to convert\n")
    
    id_mapping = {}  # old_id -> new_id
    
    for json_file in session_files:
        with open(json_file, "r", encoding="utf-8") as f:
            session_data = json.load(f)
        
        old_session_id = session_data.get("session_id")
        new_session_id = str(uuid.uuid4())
        
        print(f"File: {json_file.name}")
        print(f"  Old session_id: {old_session_id}")
        print(f"  New session_id: {new_session_id}")
        
        # Store mapping
        id_mapping[old_session_id] = new_session_id
        
        # Update session_id in captures
        session_data["session_id"] = new_session_id
        for capture in session_data.get("captures", []):
            if "session_id" in capture:
                capture["session_id"] = new_session_id
        
        # Write updated JSON to temporary file
        temp_file = json_file.with_suffix(".tmp.json")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)
        
        # Extract timestamp from filename (e.g., "session_20260328_101458_4453da30.json")
        parts = json_file.stem.split("_")
        timestamp = f"{parts[1]}_{parts[2]}"  # "20260328_101458"
        
        # Create new filename with new UUIDv4
        new_filename = f"session_{timestamp}_{new_session_id}.json"
        new_file = captures_dir / new_filename
        
        # Replace old file with new one
        temp_file.replace(new_file)
        print(f"  Renamed: {json_file.name} → {new_filename}\n")
    
    print(f"✓ Successfully converted {len(session_files)} session files")
    print(f"\nID Mapping:")
    for old_id, new_id in sorted(id_mapping.items()):
        print(f"  {old_id} → {new_id}")

if __name__ == "__main__":
    convert_session_files()
