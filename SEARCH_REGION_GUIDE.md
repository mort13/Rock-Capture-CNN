# Search Region Guide

## Overview

When using sub-anchors, you can now explicitly specify a **search region** — the area where the sub-anchor template should be looked for. This is useful when you want to constrain the search to a specific part of the HUD to improve matching accuracy.

## Workflow

### 1. Create Main Anchors
- In "Setup Anchors & ROIs" dialog, select **Main Anchor** mode
- Draw rectangles around stable HUD landmarks (e.g., top-left corner of UI panel)
- Name each one (e.g., `anchor_top_left`, `anchor_top_right`)
- Click "Add Item"

### 2. Create Sub-Anchors
- Select **Sub-Anchor** mode
- Draw a rectangle around your target (e.g., the `%` symbol)
- Name it (e.g., `percent_sign`)
- Click "Add Item"
- The template is auto-saved

### 3. (Optional) Define Search Region
- Select **Search Region (for sub-anchor)** mode
- Draw a rectangle in the expected area where the sub-anchor might be found
  - This is typically a larger region around the sub-anchor's nominal position
  - In frame reference coordinates (will be transformed at runtime)
- Select the target sub-anchor from the dropdown (e.g., `percent_sign`)
- Click "Add Item"

### 4. Create ROIs
- Select **ROI** mode
- Draw rectangles for each region you want to extract
- Name them
- Select which sub-anchor to relate them to (if any)
- Click "Add Item"

## Example: Number with % Sign

**Goal:** Extract a left-aligned number and a % symbol from the HUD, accounting for single-digit vs multi-digit numbers.

**Setup:**

1. **Main Anchor** at frame (100, 50): Matches a stable HUD element
   - Name: `number_area`
   - Template: Colored screenshot of that HUD element

2. **Sub-Anchor** at frame (150, 60): Matches the `%` symbol
   - Name: `percent_sign`
   - Template: Colored screenshot of just the `%`

3. **Search Region** for `percent_sign`:
   - Draw a box around where the % might appear (accounting for variable digit width)
   - E.g., rectangle at ref (130, 50) with size 80×30
   - In frame coords: (130, 50, 80, 30)

4. **ROIs**:
   - ROI `digit_value`: ref (120, 60) size 30×20 → relative to `percent_sign`
   - ROI `percent`: ref (160, 60) size 20×20 → relative to `percent_sign`

**At Runtime:**

1. Main anchor matches at frame position (100, 50)
2. Affine transform computed (e.g., scale 1.0 if HUD at same resolution)
3. Transform the search region coordinates: (130, 50) → (130, 50) in frame space
4. Search for `percent_sign` template within the region (130, 50, 80, 30)
5. When `percent_sign` is found (say at frame 155, 60):
   - ROI `digit_value`: from (120, 60) ref → (155 - offset_x, 60 + offset_y) in frame
   - ROI `percent`: from (160, 60) ref → (155 + offset_x, 60 + offset_y) in frame

## Benefits

- **Explicit control**: No more mysterious "padding=40" values
- **Resolution-independent**: Search regions scale with the main transform
- **Robust matching**: Constraining the search area reduces false positives
- **Multi-region capture**: Different ROIs can relate to different sub-anchors

## JSON Representation

Sub-anchors with search regions are stored in the profile JSON:

```json
{
  "sub_anchors": [
    {
      "name": "percent_sign",
      "template_path": "percent_sign.png",
      "match_threshold": 0.7,
      "ref_x": 150,
      "ref_y": 60,
      "search_region": {
        "x": 130,
        "y": 50,
        "width": 80,
        "height": 30
      }
    }
  ]
}
```

If no search region is defined, the system falls back to padding-based search (±40 pixels scaled by HUD resolution).
