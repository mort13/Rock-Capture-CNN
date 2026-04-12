# Rock Capture CNN - Anchor System Technical Specification

**Last Updated:** April 11, 2026
**For:** Porting anchor system to web/other platforms

---

## Executive Summary

The anchor system uses **TOP-LEFT CORNER coordinates** throughout all operations. This is consistent with OpenCV's convention. ROI positioning is **resolution-independent** through affine/similarity transforms computed from 2-3 anchor points.

---

## 1. COORDINATE SYSTEM: TOP-LEFT CORNER (Not Center)

### Key Point
All anchor and ROI coordinates **use the TOP-LEFT corner**, not the center:
- `ref_x, ref_y` = TOP-LEFT corner of the rectangle
- Templates are matched at their TOP-LEFT corner
- ROIs are extracted from their TOP-LEFT corner
- `width, height` = dimensions extending RIGHT and DOWN from the corner

### Example
If you draw a rectangle from pixels (100, 50) to (200, 150):
- `ref_x = 100` (left edge)
- `ref_y = 50` (top edge)
- `width = 100` (extends right)
- `height = 100` (extends down)
- **NOT** center-based coordinates

---

## 2. MULTI-ANCHOR POSITIONING WORKFLOW

### Step 1: Setting Up Anchors (GUI - anchor_setup_dialog.py)

When the user draws rectangles on a captured frame:

```
User draws rectangle on frame image
    ↓
Rectangle coordinates stored as QRect(x, y, w, h)
    ↓
For each rectangle, save:
  - x_offset = top-left X coordinate
  - y_offset = top-left Y coordinate
  - width, height = dimensions
    ↓
This becomes ref_x, ref_y in AnchorPoint / ROIDefinition
```

**Code Reference:** `gui/anchor_setup_dialog.py` line 558
```python
def get_rois(self) -> list[ROIDefinition]:
    result.append(ROIDefinition(
        name=item.name,
        ref_x=float(item.rect.x()),      # ← TOP-LEFT X
        ref_y=float(item.rect.y()),      # ← TOP-LEFT Y
        width=item.rect.width(),         # ← WIDTH
        height=item.rect.height(),       # ← HEIGHT
        sub_anchor=item.sub_anchor_name,
    ))
    return result
```

### Step 2: Template Matching (core/anchor.py)

When finding anchors in a captured frame:

```python
result = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
_, max_val, _, max_loc = cv2.minMaxLoc(result)

# max_loc is a tuple (x, y) = TOP-LEFT corner of best match
return AnchorResult(
    found=True,
    x=max_loc[0] + offset_x,  # ← TOP-LEFT X in frame coordinates
    y=max_loc[1] + offset_y,  # ← TOP-LEFT Y in frame coordinates
    anchor_w=template_width,
    anchor_h=template_height,
)
```

**Important:** `cv2.matchTemplate()` **always returns the TOP-LEFT corner** of the matched region.

### Step 3: Transform Computation (core/anchor.py)

```python
def find_anchors(frame, anchor_points):
    # For each AnchorPoint:
    #   - ap.ref_x, ap.ref_y = TOP-LEFT in reference frame
    #   - ar.x, ar.y = TOP-LEFT detected in current frame
    
    ref_pts = [(ap.ref_x, ap.ref_y) for ap in matched_anchors]
    det_pts = [(ar.x, ar.y) for ar in detected_anchors]
    
    M, scale = compute_transform(ref_pts, det_pts)
    # M is a 2×3 affine matrix mapping reference → current frame
```

The transform maps **TOP-LEFT corner positions** from the reference frame to the current captured frame.

---

## 3. ROI EXTRACTION & POSITIONING

### Multi-Anchor Mode (core/pipeline.py - _on_frame_multi_anchor)

```python
# 1. Apply transform to ROI's reference coordinates
tx, ty = transform_point(M, roi_def.ref_x, roi_def.ref_y)
roi_x = int(round(tx))
roi_y = int(round(ty))

# 2. Scale dimensions
eff_w = int(round(roi_def.width * scale))
eff_h = int(round(roi_def.height * scale))

# 3. Extract from frame
raw_roi = frame[roi_y : roi_y + eff_h, roi_x : roi_x + eff_w]
```

**Key:** `(roi_x, roi_y)` is the **TOP-LEFT corner** where extraction begins.

### Sub-Anchor Refinement

```python
if roi_def.sub_anchor and roi_def.sub_anchor in sub_anchor_positions:
    sa = profile.get_sub_anchor(roi_def.sub_anchor)
    sa_pos = sub_anchor_positions[roi_def.sub_anchor]  # (x, y) = TOP-LEFT
    
    # Relative offset from sub-anchor's TOP-LEFT
    dx = roi_def.ref_x - sa.ref_x
    dy = roi_def.ref_y - sa.ref_y
    
    roi_x = int(round(sa_pos[0] + dx * scale))
    roi_y = int(round(sa_pos[1] + dy * scale))
```

---

## 4. AFFINE TRANSFORM MATHEMATICS

### Transform Matrix M

```
M = 2×3 matrix [a, b, tx]
                [c, d, ty]

Point (x, y) → (x', y'):
  x' = a*x + b*y + tx
  y' = c*x + d*y + ty

Scale factor = sqrt(a² + c²) = sqrt(b² + d²)
```

### Computing the Transform

**For 2 points (similarity - translate + rotate + uniform scale):**
```python
cv2.estimateAffinePartial2D(src_points, dst_points)
# Computes 2×3 matrix with constraint: uniform scale & rotation
```

**For 3 points (full affine):**
```python
cv2.getAffineTransform(src_points[:3], dst_points[:3])
# Computes general 2×3 matrix (can have non-uniform scale/skew)
```

---

## 5. DATA STRUCTURES

### AnchorPoint (core/profile.py)

```python
@dataclass
class AnchorPoint:
    name: str                          # "anchor_1", "anchor_2", etc.
    template_path: str                 # Path to saved template image
    match_threshold: float = 0.7       # Template matching confidence threshold
    ref_x: float = 0                   # TOP-LEFT X in reference frame
    ref_y: float = 0                   # TOP-LEFT Y in reference frame
    search_region: dict | None = None  # Optional: {"x", "y", "width", "height"}
```

### ROIDefinition (core/profile.py)

```python
@dataclass
class ROIDefinition:
    name: str                  # "mass", "instability", etc.
    width: int = 80            # Dimension (extends right)
    height: int = 24           # Dimension (extends down)
    
    # Multi-anchor mode:
    ref_x: float = 0.0         # TOP-LEFT X in reference frame
    ref_y: float = 0.0         # TOP-LEFT Y in reference frame
    sub_anchor: str = ""       # Name of sub-anchor for refinement (empty = main)
    
    # Legacy mode:
    x_offset: int = 0          # Offset from anchor's x
    y_offset: int = 0          # Offset from anchor's y
```

### AnchorResult (core/anchor.py)

```python
@dataclass
class AnchorResult:
    found: bool
    x: int = 0                 # TOP-LEFT X (or 0 if not found)
    y: int = 0                 # TOP-LEFT Y (or 0 if not found)
    confidence: float = 0.0    # Template match confidence [0, 1]
    anchor_w: int = 0          # Template width
    anchor_h: int = 0          # Template height
```

---

## 6. JSON STORAGE FORMAT

### Example: mole_relative_anchors.json

```json
{
  "profiles": {
    "scan_results": {
      "rois": [
        {
          "name": "mass",
          "ref_x": 123.0,        # TOP-LEFT X
          "ref_y": 78.0,         # TOP-LEFT Y
          "width": 41,           # WIDTH
          "height": 18,          # HEIGHT
          "sub_anchor": ""       # Empty = use main transform only
        },
        {
          "name": "inst_int",
          "ref_x": 115.0,
          "ref_y": 100.0,
          "width": 25,
          "height": 17,
          "sub_anchor": "resistance_anchor"  # Refined by sub-anchor
        }
      ]
    }
  }
}
```

---

## 7. LEGACY SINGLE-ANCHOR MODE (For Reference)

For comparison, legacy mode uses pixel offsets from the anchor match:

```python
def _on_frame_legacy(self, frame):
    anchor = self.anchor_matcher.find_anchor(frame)
    # anchor.x, anchor.y = TOP-LEFT corner of template match
    
    for roi_def in rois:
        roi_x = anchor.x + roi_def.x_offset
        roi_y = anchor.y + roi_def.y_offset
        # Extract from (roi_x, roi_y) with (roi_def.width, roi_def.height)
```

**Still uses TOP-LEFT corners!**

---

## 8. COMMON PITFALLS WHEN PORTING

### ❌ WRONG: Using Center Coordinates
```javascript
// DO NOT DO THIS:
const centerX = rect.x + rect.width / 2;
const centerY = rect.y + rect.height / 2;
// This will shift all ROIs!
```

### ✅ CORRECT: Use Top-Left Directly
```javascript
// DO THIS:
const topLeftX = rect.x;
const topLeftY = rect.y;
```

### ❌ WRONG: Flipping Y-Axis
```python
# If web canvas has Y increasing downward (like OpenCV):
transformed_y = M @ (ref_y, ref_x)  # WRONG ORDER
```

### ✅ CORRECT: Consistent Y-Axis
```python
# Ensure Y-axis direction matches OpenCV (Y increases downward):
x_new = M[0,0] * x + M[0,1] * y + M[0,2]
y_new = M[1,0] * x + M[1,1] * y + M[1,2]
```

### ⚠️ COMMON: Mismatched Template Coordinates
When loading anchors from JSON and applying transforms:
```python
# Reference frame = stored JSON coordinates (TOP-LEFT)
ref_x, ref_y = roi_def["ref_x"], roi_def["ref_y"]

# Apply transform to TOP-LEFT, not center
tx, ty = transform_point(M, ref_x, ref_y)

# Use transformed TOP-LEFT for extraction
frame_crop = frame[int(ty) : int(ty) + height,
                   int(tx) : int(tx) + width]
```

---

## 9. VERIFICATION CHECKLIST FOR WEB PORT

- [ ] Coordinates use **TOP-LEFT corner** throughout
- [ ] `ref_x, ref_y` from JSON are treated as **TOP-LEFT**
- [ ] Rectangle extraction: `frame[y : y+h, x : x+w]` (numpy/OpenCV style)
- [ ] Template matching returns **TOP-LEFT corner**
- [ ] Affine transform matrix is applied consistently
- [ ] Scale extracted correctly: `sqrt(M[0,0]² + M[1,0]²)`
- [ ] Sub-anchor offsets are relative to **sub-anchor's TOP-LEFT**
- [ ] Y-axis direction matches (increases downward)
- [ ] Rounding applied consistently: `int(round(...))`

---

## 10. DEBUGGING: Why ROIs Are Shifted

If ROIs appear shifted in the web version:

**Likely Cause 1: Center vs. Top-Left**
```
Symptom: All ROIs shifted right/down by ~width/2, ~height/2
Fix: Use ref_x, ref_y as TOP-LEFT, not as center
```

**Likely Cause 2: Transform Not Applied Correctly**
```
Symptom: ROIs are at original positions, no scaling
Fix: Verify transform matrix M is computed and applied correctly
     Check: scale = sqrt(M[0][0]² + M[1][0]²)
```

**Likely Cause 3: Y-Axis Flipped**
```
Symptom: ROIs are mirrored vertically
Fix: Ensure Y-axis increases downward consistently
     In canvas: (0,0) is top-left, Y increases downward
```

**Likely Cause 4: Sub-Anchor Offsets Wrong**
```
Symptom: Only sub-anchor ROIs are shifted
Fix: Verify dx = ref_x - sub_anchor.ref_x (not center offset)
     Verify offset is scaled: int(round(offset * scale))
```

---

## 11. CODE REFERENCES

| File | Function | Purpose |
|------|----------|---------|
| `core/anchor.py:248-280` | `find_anchors()` | Multi-anchor detection & transform |
| `core/anchor.py:202-236` | `_match_template_in()` | Single template matching |
| `core/anchor.py:46-70` | `compute_transform()` | Affine matrix from point pairs |
| `core/anchor.py:73-76` | `transform_point()` | Apply matrix to single point |
| `core/pipeline.py:186-245` | `_on_frame_multi_anchor()` | Multi-anchor ROI extraction |
| `core/profile.py:80-160` | `ROIDefinition` dataclass | ROI data structure |
| `gui/anchor_setup_dialog.py:558-575` | `get_rois()` | UI → data serialization |

---

## 12. FAQ

**Q: Why not use center coordinates? Wouldn't that be more intuitive?**
A: OpenCV, NumPy, and image libraries use top-left corners by convention. Using top-left is **consistent with the entire Python image processing ecosystem**.

**Q: The anchors seem to be at corners not centers. Is that intentional?**
A: Yes. Each anchor is a **rectangle template**, and `cv2.matchTemplate()` returns the top-left corner. This is the foundation of the system.

**Q: My ROIs shifted when I ported to web. Where's the bug?**
A: Likely causes (in order):
1. Using center instead of top-left (most common)
2. Transform not scaled by detected scale factor
3. Y-axis direction inconsistency
4. Rounding errors accumulating

**Q: Can I use different coordinate systems for anchors vs. ROIs?**
A: No. The entire system must be consistent. All coordinates must be top-left corners.

---

## Summary

| Aspect | Value |
|--------|-------|
| **Reference Point** | TOP-LEFT CORNER |
| **Coordinates** | (x, y, width, height) |
| **Transform Type** | 2×3 Affine Matrix |
| **Transform Computation** | Reference points → Detected points |
| **Template Matching** | OpenCV `cv2.matchTemplate()` |
| **ROI Extraction** | `frame[y:y+h, x:x+w]` |
| **Sub-Anchor Refinement** | Relative to sub-anchor's TOP-LEFT |
| **Scale Extraction** | `sqrt(M[0,0]² + M[1,0]²)` |

