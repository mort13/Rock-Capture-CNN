# Volume & Mass Data Flow Analysis - Rock Capture CNN

## 1. WHERE VOLUME AND MASS VALUES ARE EXTRACTED

### Source Files:
- **[export_parquet.py](export_parquet.py)** - Primary extraction logic
- **[check_amounts.py](check_amounts.py)** - Validation of material amounts
- **[gui/main_window.py](gui/main_window.py)** - Real-time staging and validation

### Extraction Points:

#### In export_parquet.py (Lines 86-100):
```python
DEFAULT_SPEC = ExportSpec(
    scan_fields=[
        Field("mass",        "mass",        "int",       confidence=True),
        Field("volume",      "volume",      "composite", confidence=True),
    ],
)
```

**Field Types:**
- **"int"** - Simple integer values cast from JSON
- **"composite"** - Combines `<field>_int` and `<field>_dec` into a single float
  - Example: `mass_int: 5`, `mass_dec: 2` → `5.2`
  - Example: `volume_int: 100`, `volume_dec: 50` → `100.50`

#### Volume Extraction Flow (Lines 225-256):
1. Extract scan volume from composite fields: `volume_int` + `volume_dec`
2. For each material in composition array:
   - Get material percentage: `amount_int` + `amount_dec`
   - Calculate actual volume: `(material_amount / 100.0) * scan_volume`
   - Store in `material_volume` field

```python
scan_volume = _extract_value(scan, Field("volume", "volume", "composite"))

for mat in composition:
    material_amount = _extract_material_value(mat, MaterialField("amount", "amount", "composite"))
    if material_amount is not None and scan_volume is not None:
        material_volume = (material_amount / 100.0) * scan_volume
        row["material_volume"] = material_volume
```

#### Data Structure in JSON Files:
```json
{
  "scan": {
    "mass_int": {"value": "5", "confidence": 0.95},
    "mass_dec": {"value": "2", "confidence": 0.95},
    "volume_int": {"value": "100", "confidence": 0.92},
    "volume_dec": {"value": "50", "confidence": 0.92},
    "composition": [
      {
        "name": {"value": "Gold", "confidence": 0.99},
        "amount_int": {"value": "45", "confidence": 0.88},
        "amount_dec": {"value": "5", "confidence": 0.88},
        "quality": {"value": "8", "confidence": 0.91}
      }
    ]
  }
}
```

---

## 2. VALIDATION & FLAG CHECKS

### Location: [gui/main_window.py](gui/main_window.py)

#### Two Main Validation Functions:

##### A. `_compute_amount_red_keys()` (Lines 1100-1140)
**Purpose:** Checks if material percentages sum to ~100%

**Logic:**
1. Collects all named materials (those with non-empty names)
2. Sums their amounts: `sum(amount_int.amount_dec for each material)`
3. Checks against tolerance: Flags RED if `total < (100 - tolerance)` OR `total > (100 + tolerance)`
4. Default tolerance: `2.0%` (configurable)

**Validation Checks:**
- Supports both formats:
  - **Structured:** `scan/composition[N]/amount_int`, `scan/composition[N]/amount_dec`
  - **Legacy:** `values/material{N}/int`, `values/material{N}/decimal`
- Ignores materials without names
- Returns set of flat keys to mark RED

**Example:**
```python
# These amounts are VALID (sum = 100%)
45.5 + 30.2 + 24.3 = 100.0  ✓

# These amounts trigger RED flag (sum = 95%, outside tolerance)
45.5 + 30.2 + 19.3 = 95.0   ✗ (below 100 ± 2%)
```

##### B. `_compute_deposit_red_keys()` (Lines 1142-1160)
**Purpose:** Warns when deposit name changes within same cluster

**Logic:**
1. Gets current deposit name from staged values
2. Compares with last committed capture's deposit
3. If same cluster_id but different deposit → RED flag
4. Prevents accidental data inconsistency

**Example Use Case:**
```python
# Cluster ABC-123: First capture has deposit = "Sector 7"
# User changes staged data to "Sector 8" but cluster is still "ABC-123"
# → deposit_name field turns RED
```

---

## 3. REAL-TIME VALIDATION DURING STAGING

### The "Staging" Workflow (F9/F10 Keys):

**F9 Key:** Freeze current frame results
- Calls `_on_stage_pressed()` (Line 1248)
- Captures latest ROI results into `_staged_data`
- Freezes controls panel with editable fields
- Computes red_keys for initial validation
- Enables real-time validation callbacks

**Real-time Validation:** (Lines 1177-1180)
```python
def _validate_staged_values(self, flat_values: dict[str, str]) -> set[str]:
    """Validation callback for real-time updates while editing staged values."""
    return (
        self._compute_amount_red_keys(flat_values)
        | self._compute_deposit_red_keys(flat_values)
    )
```

**How it works:**
- User edits any staged field → `_on_staged_value_changed()` fires
- Calls `_validation_callback()` to get updated red_keys
- Updates all field backgrounds in real-time
- Fields with issues turn RED (#ffaaaa)
- Valid fields stay YELLOW (#ffffcc)

**F10 Key:** Commit staged data to JSON
- Calls `_on_commit_to_json()` (Line 1364)
- Writes to `data/captures/session_*.json`
- Even if red flags present (user can override)
- Creates new session file or appends to existing

---

## 4. WHERE DATA IS WRITTEN TO JSON

### Session JSON Files:

**Location:** `data/captures/session_<TIMESTAMP>_<SESSION_ID>.json`

**Written by:** `_on_commit_to_json()` (Lines 1364-1419)

**Structure:**
```python
session_data = {
    "session_id": str,
    "started_at": ISO-timestamp,
    "tool_version": str,
    "source": {
        "user": str,
        "org": str,
    },
    "captures": [
        {
            "timestamp": ISO-timestamp,
            "capture_id": UUID,
            "cluster_id": str,  # F11 hotkey cycles cluster IDs
            "location": {
                "system": str,       # From location_edit field
                "gravity_well": str,
                "region": str,
                "place": str,
            },
            # Structured OR legacy format based on output_schema
            "scan": {...},  # or "values": {...}
        }
    ]
}
```

### Export Pipeline:

**[export_parquet.py](export_parquet.py)** reads session JSONs and exports to:
- `scans.parquet` - One row per capture (scan-level data)
- `compositions.parquet` - One row per material (joined by capture_id)

**Calculated Fields:**
- `material_volume` - Derived from: `(amount_percentage / 100) * scan_volume`

---

## 5. WHAT "STAGING" MEANS

### Staging = Temporary Frozen Frame Capture

**Purpose:** Allow user review and editing before final commitment

**Workflow:**
```
Live Capture (F9) → Freeze Frame → Edit Values → Commit (F10)
                                  ↑ Validation
                           Red flags for issues
```

**States:**
1. **Live Mode** (default):
   - Pipeline continuously captures frames
   - Results update in real-time
   - No editing capability

2. **Staged Mode** (F9 pressed):
   - Current frame results frozen
   - Results become editable text fields
   - Real-time validation checks active
   - Can cancel (F9 again) to return to live mode

3. **Committed Mode** (F10 pressed):
   - Staged values written to JSON
   - Return to live mode
   - Status bar shows commit details

**Staging Data Structure:** (`_staged_data` dict)
```python
{
    "timestamp": "2024-04-06T14:23:45",
    "_structured": True,  # True if output_schema active, False if legacy
    "scan": {...},        # Structured format
    # or "values": {...}  # Legacy format
}
```

---

## 6. HOW ITEMS ARE MARKED AS "RED" OR FLAGGED AS INVALID

### Red Flag System:

**Visual Representation:**
- RED background (#ffaaaa) = Invalid/Warning state
- YELLOW background (#ffffcc) = Valid state

### Red Flag Triggers:

#### A. Amount Validation (Material Percentages)
**Condition:** Named material amounts don't sum to 100% ± tolerance

**Affected Fields:**
- ALL `amount_int` and `amount_dec` fields for named materials
- Entire amount field pair marked red

**Example:**
```
Material 1: 30.5%
Material 2: 45.2%
Material 3: 20.1%
Total: 95.8% → RED (outside 100 ± 2%)
```

#### B. Deposit Validation
**Condition:** Deposit name changed in same cluster

**Affected Fields:**
- `scan/deposit_name` OR `values/deposit_name/name`
- Only the deposit_name field marked red

**Example:**
```
Cluster "ABC-123" has:
  - Capture 1: deposit = "Sector 7" ✓
  - Capture 2 (staged): deposit = "Sector 8" → RED
```

### Red Key Collection:

```python
red_keys = (
    self._compute_amount_red_keys(flat_values)     # Amount issues
    | self._compute_deposit_red_keys(flat_values)  # Deposit changes
)
```

### Field Styling Logic:

```python
def _update_field_style(self, edit: QLineEdit, is_red: bool) -> None:
    if is_red:
        edit.setStyleSheet(
            "font-size: 13px; background-color: #ffaaaa; color: #000; padding: 2px;"
        )
    else:
        edit.setStyleSheet(
            "font-size: 13px; background-color: #ffffcc; padding: 2px;"
        )
```

### Important: Red Flags Are Warnings, NOT Blockers

- User can commit staged data even with red flags present
- Validation is informational/advisory
- Warnings displayed in status bar:
  - "⚠ amounts ≠ 100%"
  - "⚠ deposit name changed in same cluster"

---

## 7. EXISTING VALIDATION LOGIC SUMMARY

### Validation Functions:

| Function | Purpose | File | Lines |
|----------|---------|------|-------|
| `_compute_amount_red_keys()` | Sum of material amounts ≈ 100% | main_window.py | 1100-1140 |
| `_compute_deposit_red_keys()` | Deposit consistency in cluster | main_window.py | 1142-1160 |
| `_validate_staged_values()` | Real-time callback combiner | main_window.py | 1177-1180 |
| `check()` | Batch validation of JSONs | check_amounts.py | 30-96 |

### Validation Execution:

**During Staging (F9):**
1. Called once: `_validate_staged_values()` to compute initial red_keys
2. Then called on every field edit to update red_keys

**Batch Validation:**
- Run: `python check_amounts.py --captures data/captures/`
- Checks all session JSONs for material amount consistency
- Can optionally remove bad captures: `python check_amounts.py --remove`

---

## 8. CURRENT DATA FLOW DIAGRAM

```
Pipeline (live capture)
    ↓
Frame Results (ROI text + confidence)
    ↓
[F9 - Stage Pressed]
    ↓
Freeze Frame → Build _staged_data
    ↓
Flatten to editable fields
    ↓
Compute red_keys:
  ├─ _compute_amount_red_keys()
  └─ _compute_deposit_red_keys()
    ↓
Freeze Controls Panel (yellow fields with red flags)
    ↓
User Edits Fields (real-time validation updates red status)
    ↓
[F10 - Commit Pressed]
    ↓
Apply edits back to _staged_data
    ↓
Wrap in capture object (timestamp, location, cluster_id)
    ↓
Write to session_*.json
    ↓
[Export Pipeline]
    ↓
Parse JSON → Parquet (scans.parquet + compositions.parquet)
```

---

## 9. TOLERANCE & CONFIGURATION

### Amount Tolerance:
- **Default:** 2.0% (stored in `self._tolerance_percentage`)
- **Location:** Constructor of [gui/main_window.py](gui/main_window.py)
- Currently hardcoded but could be made configurable

### Where Configured:
```python
self._tolerance_percentage = 2.0  # Line ~90 in __init__
```

---

## 10. FILES INVOLVED IN VOLUME/MASS HANDLING

| File | Role |
|------|------|
| [export_parquet.py](export_parquet.py) | Extract/transform volume & mass from JSON for export |
| [check_amounts.py](check_amounts.py) | Batch validate material amounts |
| [gui/main_window.py](gui/main_window.py) | Staging workflow, real-time validation |
| [gui/controls_panel.py](gui/controls_panel.py) | Freeze/unfreeze staged fields, apply red styling |
| [core/profile.py](core/profile.py) | Profile definitions with output_schema |
| [core/config.py](core/config.py) | Load/save config including user/org metadata |

---

## Summary Checklist

✅ **Files involved:** export_parquet.py, check_amounts.py, main_window.py, controls_panel.py  
✅ **Current validation:** Amount sum check (100% ± 2%), Deposit consistency check  
✅ **Data structure:** Composite fields (int + dec), confidence values  
✅ **Flags/errors:** Red background (#ffaaaa) on invalid fields  
✅ **"Red" definition:** Material amounts deviate from 100%, or deposit changes in same cluster  
✅ **Staging:** F9 freezes frame, F10 commits to JSON  
✅ **JSON export:** session_*.json → parquet files for analytics  
