# CRNN Web Deployment Implementation Prompt

## Overview
Extend the existing ONNX export pipeline to support the CRNN digit sequence model for browser-based inference. The current system exports DigitCNN and WordCNN; this adds CRNN export and web-side integration.

---

## Phase 1: Backend ONNX Export (Python)

### File: `cnn_export_onnx.py` — Extend to support CRNN

**Changes to make:**

1. **Import and detection logic**
   - Add import: `from digit_crnn.model import DigitCRNN`
   - Extend the model detection heuristic to identify CRNN checkpoints
     - **Heuristic**: Check for key `"model_state_dict"` and architecture-specific markers
     - DigitCRNN has keys like `rnn.weight_ih_l0`, `rnn.weight_hh_l0` (LSTM weights)
     - DigitCNN has only Conv2d + Linear keys (no RNN)
     - WordCNN has `conv3.weight` (3rd conv layer) that DigitCRNN doesn't have

2. **CRNN export path**
   ```python
   # Pseudo-code structure
   if has_lstm_keys:  # Detected CRNN
       char_classes = checkpoint.get("char_classes", "0123456789.%")
       num_classes = checkpoint.get("num_classes", DigitCRNN.NUM_CLASSES)
       model = DigitCRNN(num_classes=num_classes)
       dummy = torch.randn(1, 1, DigitCRNN.INPUT_H, DigitCRNN.INPUT_W)  # (1, 1, 32, 256)
       print(f"Detected DigitCRNN model ({num_classes} classes, 64 timesteps)")
   ```

3. **ONNX export parameters for CRNN**
   ```python
   torch.onnx.export(
       model, dummy, output_path,
       input_names=["input"],
       output_names=["log_probs"],  # ← Changed from "logits"
       # CRNN output is (T=64, batch, num_classes), not (batch, num_classes)
       dynamic_axes={
           "input": {0: "batch"},
           "log_probs": {0: "T", 1: "batch"}  # ← T is first now
       },
       opset_version=17,
   )
   ```

4. **Metadata JSON structure for CRNN**
   ```json
   {
       "modelType": "crnn",
       "numClasses": 13,
       "inputShape": [1, 1, 32, 256],
       "outputShape": [64, 1, 13],
       "timeSteps": 64,
       "blankIdx": 0,
       "charClasses": "0123456789.%",
       "formats": {
           "decimalPercent": "^\\d{1,2}\\.\\d{2}%$",
           "decimal": "^\\d{1,3}\\.\\d{2}$",
           "percent": "^\\d{1,2}%$",
           "integer": "^\\d{1,6}$"
       },
       "valAccuracy": 0.95
   }
   ```

5. **CLI usage**
   ```bash
   python cnn_export_onnx.py --model data/models/crnn_model.pth --output crnn_model.onnx
   ```

---

## Phase 2: Web-Side Integration (JavaScript/TypeScript)

### Prerequisites
- ONNX Runtime JS: `npm install onnxruntime-web`
- Web canvas/image processing (e.g., OpenCV.js or native Canvas API)

### Core Components Needed

#### 1. Model Loading
```typescript
// Load CRNN ONNX model and metadata JSON
async function loadCRNNModel(modelPath: string, metadataPath: string) {
    // Fetch both crnn_model.onnx and crnn_model.json
    const modelData = await fetch(modelPath).then(r => r.arrayBuffer());
    const metadata = await fetch(metadataPath).then(r => r.json());
    
    const session = await ort.InferenceSession.create(modelData);
    return { session, metadata };
}
```

#### 2. Image Preprocessing
Implement the same preprocessing as `digit_crnn/dataset.py`:
- **Contrast check**: `max(gray) - min(gray) >= 100` (skip if below)
- **Autocrop**: Crop to bounding box of pixels > 35% of peak brightness
- **Resize+pad**: Fit within 32×256 while preserving aspect, left-align, vertically center
- **Normalize**: Divide by 255.0

```typescript
function preprocessStripImage(imageData: Uint8ClampedArray, width: number, height: number): Float32Array {
    // 1. Convert to grayscale if color
    // 2. Check contrast
    // 3. Autocrop text content
    // 4. Resize+pad to 32×256
    // 5. Normalize and return as Float32Array
    return normalized32x256;
}
```

#### 3. Inference & CTC Decoding
```typescript
async function inferCRNN(session: ort.InferenceSession, imageData: Float32Array) {
    // Input: (1, 1, 32, 256)
    const input = new ort.Tensor("float32", imageData, [1, 1, 32, 256]);
    const result = await session.run({ input });
    
    // Output shape: [T=64, batch=1, num_classes=13]
    const logProbs = result.log_probs;
    const data = logProbs.data as Float32Array;
    
    // Greedy decode: argmax per timestep
    const decodedIndices = greedyCTCDecode(data);
    const text = indicesToText(decodedIndices, metadata.charClasses);
    
    return { text, confidence: computeConfidence(logProbs, decodedIndices) };
}

function greedyCTCDecode(logProbs: Float32Array): number[] {
    // logProbs is [T=64, batch=1, num_classes=13] flattened row-major
    const T = 64, batch = 1, C = 13;
    const result: number[] = [];
    let prevIdx = -1;
    
    for (let t = 0; t < T; t++) {
        // Find argmax for this timestep, batch 0
        let maxIdx = 0, maxVal = -Infinity;
        for (let c = 0; c < C; c++) {
            const val = logProbs[t * batch * C + 0 * C + c];
            if (val > maxVal) { maxVal = val; maxIdx = c; }
        }
        
        // Collapse consecutive duplicates
        // Remove blanks (idx 0)
        if (maxIdx !== prevIdx && maxIdx !== 0) {
            result.push(maxIdx);
        }
        prevIdx = maxIdx;
    }
    return result;
}

function indicesToText(indices: number[], charClasses: string): string {
    // charClasses = "0123456789.%"
    // indices are 1-based (blank=0, '0'=1, '9'=10, '.'=11, '%'=12)
    return indices.map(i => i > 0 && i <= charClasses.length ? charClasses[i - 1] : '?').join('');
}

function computeConfidence(logProbs: ort.Tensor, decodedIndices: number[]): number {
    // Mean of max softmax probabilities over non-blank timesteps
    const data = logProbs.data as Float32Array;
    const T = 64, C = 13;
    let sumProbs = 0, count = 0;
    
    for (let t = 0; t < T; t++) {
        let maxLogProb = -Infinity;
        for (let c = 0; c < C; c++) {
            const val = data[t * C + c];
            if (val > maxLogProb) maxLogProb = val;
        }
        if (t in decodedIndices || decodedIndices.includes(/* timestep maxIdx */)) {
            sumProbs += Math.exp(maxLogProb);  // Convert log-softmax back to probability
            count++;
        }
    }
    return count > 0 ? sumProbs / count : 0;
}
```

#### 4. Format Validation
Implement regex matching on web-side to reject invalid sequences:
```typescript
function validateFormat(text: string, formatPattern: string, metadata: any): boolean {
    // formatPattern examples: "{1,2}.{2}%", "{1,3}.xx", "{1,2}%", "{1,6}"
    // Use metadata.formats regex list to validate
    
    // Heuristic matching (same as Python):
    if (formatPattern.includes('.') && formatPattern.includes('%')) {
        // Decimal percent: ^\\d{1,2}\\.\\d{2}%$
        return /^\d{1,2}\.\d{2}%$/.test(text);
    }
    if (formatPattern.includes('.')) {
        // Decimal: ^\\d{1,3}\\.\\d{2}$
        return /^\d{1,3}\.\d{2}$/.test(text);
    }
    if (formatPattern.includes('%')) {
        // Percent: ^\\d{1,2}%$
        return /^\d{1,2}%$/.test(text);
    }
    // Integer: ^\\d{1,6}$
    return /^\d{1,6}$/.test(text);
}

async function predictROI(roi: HTMLCanvasElement, formatPattern: string) {
    const imageData = extractAndPreprocess(roi);
    const { text, confidence } = await inferCRNN(session, imageData);
    
    // Validation: return empty string on invalid format
    if (!validateFormat(text, formatPattern, metadata)) {
        return { text: "", confidence: 0 };
    }
    return { text, confidence };
}
```

---

## Phase 3: Integration with Existing Web Platform

### Assumptions (adjust as needed)
- Your web app already loads DigitCNN and WordCNN models
- You have a ROI processing dispatch function

### Changes to ROI dispatcher

Add a new case in the recognition mode switch:
```typescript
switch (roi.recognition_mode) {
    case "cnn":
        return await predictDigitCNN(roi, roi.allowed_chars);
    case "digit_crnn":
        return await predictCRNN(roi, roi.format_pattern);  // ← NEW
    case "template":
        return await predictTemplate(roi);
    case "word_cnn":
        return await predictWordCNN(roi);
}
```

### Metadata-driven setup
Store CRNN model path in config and auto-detect at startup:
```typescript
// At initialization
const models = {
    digitCnn: { model, metadata },
    crnn: { model, metadata },  // ← NEW
    wordCnn: { model, metadata },
};

// Auto-load from manifest
const manifest = await fetch("models/manifest.json").then(r => r.json());
for (const { type, path, metadataPath } of manifest) {
    if (type === "crnn") {
        models.crnn = await loadCRNNModel(path, metadataPath);
    }
}
```

---

## Phase 4: Testing Checklist

- [ ] Export CRNN checkpoint: `python cnn_export_onnx.py --model data/models/crnn_model.pth --output crnn_model.onnx`
- [ ] Verify `crnn_model.json` metadata is correct (13 classes, 64 timesteps, format regexes)
- [ ] Load ONNX model in browser console without errors
- [ ] Test preprocessing on known strip images (compare with Python preprocessing)
- [ ] Test greedy decode on known sequences (e.g., ["5", ".", "2", "5", "%"] → "5.25%")
- [ ] Test format validation rejects invalid outputs (e.g., "5.25" when expecting `{1,2}.{2}%`)
- [ ] Compare web inference output with Python `CRNNPredictor.predict()` on same images
- [ ] Benchmark inference latency (typical: <50ms per ROI on modern browser)

---

## Output Artifacts

After implementation, you should have:

1. **Backend**
   - `cnn_export_onnx.py` — Updated with CRNN support
   - `crnn_model.onnx` — Exported model (binary)
   - `crnn_model.json` — Metadata (schema defined above)

2. **Web**
   - `crnnModel.ts` — Model loading & inference
   - `ctcDecode.ts` — Greedy CTC decoder + format validation
   - Updated ROI dispatcher in main inference loop
   - Updated model manifest/config to register CRNN

---

## Notes

- **Input shape**: CRNN expects 32×256 (H×W), different from DigitCNN (28×28). Ensure preprocessor handles this.
- **Output shape**: (T=64, batch, 13) vs DigitCNN output (batch, num_classes). Mind the transpose when indexing.
- **Log-softmax**: CRNN outputs log-softmax (from `F.log_softmax`), so convert back via `exp()` for confidence.
- **Blank handling**: CTC blank index (0) must be stripped during decode. No characters should be labeled as blank.
- **Format patterns**: Web-side regex must exactly match Python validation in `digit_crnn/predictor.py`.
- **Metadata JSON**: Include both `formats` (regex map) and `charClasses` for flexibility.

---

## References

- CRNN model arch: [`digit_crnn/model.py`](digit_crnn/model.py)
- Python predictor: [`digit_crnn/predictor.py`](digit_crnn/predictor.py) — see `_validate_format()` and CTC decode logic
- ONNX Runtime JS docs: https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js
- CTC decoding reference: https://towardsdatascience.com/beam-search-decoding-in-ctc-for-end-to-end-speech-recognition-part-2-d36a2645f65e
