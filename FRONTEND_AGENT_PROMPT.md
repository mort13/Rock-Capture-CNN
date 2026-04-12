# Frontend OCR Tool Agent Prompt

## Project Overview

Build a web-based OCR (Optical Character Recognition) tool that replicates the advanced recognition techniques from **Rock Capture CNN**. The tool will use multi-anchor positioning, precise ROI extraction, character segmentation, and CNN inference to extract structured text data from images, with fully configurable output schema.

**Key Constraint**: Transfer existing setup from desktop profiles (stored in JSON) to the web application for zero-reconfiguration deployment.

---

## Core Architecture

### 1. Recognition Pipeline

The OCR pipeline executes in this sequence:

```
Image Input → Multi-Anchor Detection → Affine Transform → 
ROI Extraction → Image Filtering → Character Segmentation → 
CNN Inference → Post-Processing → Structured Output
```

#### Phase 1: Multi-Anchor Detection
- **Objective**: Establish resolution-independent positioning using 2-3 anchor templates
- **Implementation**: 
  - Load multiple anchor templates (anchor points) as reference images
  - Use normalized cross-correlation (`matchTemplate` equivalent in JavaScript) to find each anchor in the input image
  - Compute a 2×3 affine transformation matrix from 2-3 matched anchor points
  - Extract a uniform scale factor from the transformation
- **Benefit**: ROI positions automatically scale and translate based on image resolution—users don't need to recalibrate for different quality screenshots

#### Phase 2: ROI Extraction
- Extract rectangular regions of interest (ROIs) from the transformed image using coordinates
- Support two positioning modes:
  - **Legacy**: x_offset/y_offset from primary anchor
  - **Multi-anchor**: ref_x/ref_y in reference frame (recommended)
- Support optional **sub-anchors** for local refinement of specific ROI positions

#### Phase 3: Image Filtering
- Per-ROI configurable filters (from profile):
  - Brightness adjustment
  - Contrast adjustment
  - Grayscale conversion
  - Threshold (binary conversion)
  - Color channel extraction (red/green/blue/none)
  - Inversion
- Apply filters in sequence as specified in ROI configuration

#### Phase 4: Character Segmentation
- Split filtered ROI into individual characters using one of three modes:
  - **Projection**: Vertical histogram analysis (detects valleys between separated chars)
  - **Contour**: OpenCV contour detection (requires clear gaps)
  - **Fixed-width**: Equal-width column slices (handles touching characters optimally)
- Normalize each character to 28×28 grayscale
- Use `format_pattern` to validate character count and insert literals (e.g., "xx%" → 2 digits + %)

#### Phase 5: CNN Inference
- Load ONNX models:
  - `digit_cnn.onnx` - digit/character classification
  - `word_cnn.onnx` - word recognition
- Run normalized 28×28 character images through the appropriate model
- Support ROI-specific recognition modes:
  - **"cnn"** - segment and classify individual characters
  - **"word_cnn"** - recognize entire ROI as a word
  - **"template"** - match against pre-defined template images
- Filter predictions by `allowed_chars` (if set) to only accept valid character classes for that ROI

#### Phase 6: Post-Processing & Output
- Construct structured output using the configurable **output schema**
- Transform raw ROI predictions into hierarchical JSON/objects per schema definition
- Return confidence scores and raw segmented character crops (optional debug output)

---

## Data Models & Configuration

### Profile Structure (JSON Format)

Profiles are portable JSON files that fully define the OCR setup. They contain:

```json
{
  "name": "my_profile",
  "anchor_points": [
    {
      "name": "top_left",
      "template_path": "anchors/top_left.png",
      "match_threshold": 0.75,
      "ref_x": 50.0,
      "ref_y": 50.0,
      "search_region": {
        "x": 0, "y": 0, "width": 640, "height": 480
      }
    },
    {
      "name": "bottom_right",
      "template_path": "anchors/bottom_right.png",
      "match_threshold": 0.75,
      "ref_x": 1200.0,
      "ref_y": 900.0
    }
  ],
  "rois": [
    {
      "name": "material_name",
      "ref_x": 200.0,
      "ref_y": 150.0,
      "width": 120,
      "height": 24,
      "filters": {
        "brightness": 5,
        "contrast": 20,
        "threshold": 127,
        "threshold_enabled": true,
        "grayscale": true,
        "invert": false,
        "channel": "none"
      },
      "seg_mode": "projection",
      "char_count": 15,
      "allowed_chars": "",
      "format_pattern": "",
      "recognition_mode": "cnn",
      "enabled": true
    },
    {
      "name": "mass_value",
      "ref_x": 800.0,
      "ref_y": 300.0,
      "width": 60,
      "height": 20,
      "filters": { /* ... */ },
      "seg_mode": "fixed_width",
      "char_count": 5,
      "allowed_chars": "0123456789",
      "format_pattern": "xxx.xx",
      "dot_width": 4,
      "recognition_mode": "cnn",
      "enabled": true
    }
  ],
  "output_schema": [
    {
      "key": "scan_data",
      "type": "object",
      "children": [
        {
          "key": "material",
          "type": "ref",
          "profile": "my_profile",
          "roi": "material_name"
        },
        {
          "key": "mass",
          "type": "ref",
          "profile": "my_profile",
          "roi": "mass_value"
        }
      ]
    }
  ]
}
```

### Profile Import/Export
- Store profiles as JSON files (serializable, versionable)
- Implement profile import dialog: users upload JSON or select from library
- Generate example profiles with sensible defaults
- Export current configuration as downloadable JSON for backup/sharing

---

## Output Schema System

The output schema defines how raw ROI results are transformed into the final JSON structure. It's a tree of nodes:

### Schema Node Types:

1. **ROIRef (Leaf)**
   ```json
   {
     "key": "material",
     "type": "ref",
     "profile": "profile_name",
     "roi": "roi_name"
   }
   ```
   - Inserts the recognized text/confidence from the specified ROI
   - Result value: `{ "text": "...", "confidence": 0.95 }`

2. **Object Node**
   ```json
   {
     "key": "vessel_info",
     "type": "object",
     "children": [ /* nested nodes */ ]
   }
   ```
   - Collects children into a dictionary

3. **Array Node**
   ```json
   {
     "key": "cargo_holds",
     "type": "array",
     "children": [ /* repeated for each item */ ]
   }
   ```
   - Repeats child structure multiple times (useful for variable-length fields)

### Output Example
Input profile with material ROI → recognized as "Copper"  
Schema: `{ "key": "materials", "type": "object", "children": [{ "key": "primary", "type": "ref", "roi": "material_name" }] }`  
Result:
```json
{
  "materials": {
    "primary": {
      "text": "Copper",
      "confidence": 0.987
    }
  }
}
```

---

## Web Application Requirements

### 1. User Interface

#### Setup/Configuration Panel
- **Profile Manager**:
  - Import profile JSON (drag-drop or file picker)
  - Create/edit profiles visually (no JSON editing required for basic setup)
  - Preview anchors and ROI regions on a sample image
  - Adjust anchor match thresholds in real-time
  - Save modified configurations as new profiles

- **Anchor Setup**:
  - Display image
  - Click to select anchor templates (upload or paste from known locations)
  - Mark reference points visually (adjust thresholds with threshold slider)
  - Preview computed transform and ROI positions
  - Validate that all required anchors are present

- **ROI Editor**:
  - Visual ROI boundary editor (click-and-drag rectangles)
  - Per-ROI filter controls (brightness/contrast/threshold sliders)
  - Segmentation mode selector
  - Character count / fixed-width configuration
  - Format pattern validator (show segmentation preview)
  - Recognition mode selector (cnn/word_cnn/template)
  - Test segmentation on live ROI preview

- **Output Schema Builder**:
  - Tree-based schema editor (Add Group / Add ROI Ref / Delete buttons)
  - Drag-to-reorder schema nodes
  - Visual validation (highlight referenced ROIs)

#### Processing Panel
- **Image Upload**:
  - Drag-drop or file picker for input images
  - Display uploaded image with anchor/ROI overlays (toggleable)
  - Optional: live capture from camera/screen region

- **Processing Status**:
  - Real-time feedback (anchors found, ROI extraction status, segmentation preview)
  - Confidence scores per ROI
  - Debug view: segmented characters, CNN predictions, confidence per character

- **Results Display**:
  - Formatted JSON output
  - Copy-to-clipboard buttons
  - Export as JSON/CSV
  - Side-by-side comparison: input image vs annotated output (anchors, ROIs, character boxes)

### 2. Backend Requirements (Processing Engine)

#### ONNX Inference
- Load `digit_cnn.onnx` and `word_cnn.onnx` models using an ONNX runtime
- Run inference on normalized 28×28 grayscale images
- Return top prediction + confidence score
- Support batching for performance

#### Image Processing
- Template matching (normalized cross-correlation)
- Affine transform computation from point sets
- ROI extraction with boundary checking
- Image filtering (brightness, contrast, threshold, channel extraction, inversion)
- Character segmentation (projection/contour/fixed-width modes)

#### Optional: Server-side Processing
- If processing overhead is high, offload to a backend service (Node.js, Python Flask, or WebAssembly)
- Stream results back to UI in real-time
- Cache ONNX models in memory for fast inference

### 3. Deployment & Storage

- **Profile Storage**:
  - IndexedDB or local file system for client-side profile persistence
  - Optional: server-side profile library (shareable templates)

- **Model Caching**:
  - Cache ONNX models locally (IndexedDB or service worker)
  - Lazy-load on first use (show loading indicator)

- **Data Persistence**:
  - Auto-save current configuration
  - Recovery mode if app crashes

---

## Technical Stack Recommendations

### Frontend
- **Framework**: React (component-based UI, state management) or Vue.js (simpler)
- **Image Processing**: OpenCV.js or custom canvas-based implementation
- **ONNX Runtime**: ONNXRuntime Web
- **UI Components**: Material-UI or Tailwind CSS
- **State Management**: Redux/Zustand for profile + result state

### Optional Backend
- **Node.js + Express**: Handle heavy image processing + ONNX inference
- **Python Flask**: Direct integration with PyTorch-based ONNX runner
- **WebAssembly**: Compile OpenCV/image processing to WASM for performance

---

## Integration with Existing Rock Capture CNN Profiles

1. **Profile Format Compatibility**:
   - Use identical JSON schema as Desktop version
   - Desktop profiles can be directly imported into web app
   - Bidirectional export (web-generated profiles work in desktop app)

2. **Data Transfer**:
   - Profiles live in version control (Git)
   - Import from GitHub/file share for team collaboration
   - Tag profiles by domain ("ship_scanner", "mining_site", etc.)

3. **Anchor/ROI Asset Management**:
   - Store anchor templates in a `data/anchors/` directory
   - Bundle anchors with profile (zip or embedded as base64)
   - On import, extract anchors to browser cache or server storage

---

## Testing & Validation

1. **Unit Tests**:
   - Anchor matching algorithm
   - Transform computation
   - ROI extraction accuracy
   - Segmentation output

2. **Integration Tests**:
   - Full pipeline on sample images
   - Profile import/export round-trip
   - Output schema generation

3. **Visual Validation**:
   - Display annotated images (anchors, ROIs, segmented chars)
   - Compare desktop vs. web results on same input

---

## Deliverables Checklist

- [ ] Profile import/export (JSON format)
- [ ] Multi-anchor detection + affine transform
- [ ] ROI extraction with transform
- [ ] Image filtering pipeline (per-ROI)
- [ ] Character segmentation (3 modes)
- [ ] ONNX model loading + inference (digit + word models)
- [ ] Output schema tree builder and processor
- [ ] Web UI with configuration and processing panels
- [ ] Result display with annotations and export
- [ ] Deployment guide (local or server-based)
- [ ] Documentation for end-users

---

## Success Criteria

1. **Accuracy**: OCR results match desktop version (within confidence threshold)
2. **Usability**: Non-technical users can import profiles and process images without code
3. **Performance**: Process images in <2 seconds (including model inference)
4. **Portability**: Profiles seamlessly move between desktop and web app
5. **Extensibility**: Easy to add new profiles or modify existing ones graphically
