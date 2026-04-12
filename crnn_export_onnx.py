"""
Export CRNN digit sequence model to ONNX format for web deployment.
Generates both .onnx binary and .json metadata for browser inference.

Usage:
    python crnn_export_onnx.py --model data/models/crnn_model.pth --output crnn_model.onnx
"""

import argparse
import json
import torch
import onnx

from pathlib import Path
from digit_crnn.model import DigitCRNN


def export_crnn_to_onnx(
    model_path: str | Path,
    output_onnx: str | Path,
    output_json: str | Path | None = None,
) -> None:
    """
    Export a trained CRNN checkpoint to ONNX + metadata JSON.
    
    Args:
        model_path: Path to crnn_model.pth checkpoint
        output_onnx: Path to write .onnx file
        output_json: Path to write .json metadata (default: same dir as output_onnx)
    """
    model_path = Path(model_path)
    output_onnx = Path(output_onnx)
    
    if output_json is None:
        output_json = output_onnx.with_suffix('.json')
    else:
        output_json = Path(output_json)
    
    # Create output directory if needed
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[CRNN Export] Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    
    # Extract metadata from checkpoint
    char_classes = checkpoint.get("char_classes", "0123456789.%")
    num_classes = checkpoint.get("num_classes", DigitCRNN.NUM_CLASSES)
    val_accuracy = checkpoint.get("val_accuracy", 0.0)
    
    print(f"[CRNN Export] Model config:")
    print(f"  - Num classes: {num_classes}")
    print(f"  - Char classes: {char_classes}")
    print(f"  - Validation accuracy: {val_accuracy:.4f}")
    
    # Create and load model
    model = DigitCRNN(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Create dummy input (batch=1, channels=1, height=32, width=256)
    dummy_input = torch.randn(1, 1, DigitCRNN.INPUT_H, DigitCRNN.INPUT_W)
    
    print(f"[CRNN Export] Input shape: {tuple(dummy_input.shape)}")
    
    # Run dummy forward pass to verify
    with torch.no_grad():
        dummy_output = model(dummy_input)
    print(f"[CRNN Export] Output shape: {tuple(dummy_output.shape)}")
    print(f"[CRNN Export] Expected output shape: (T=64, batch=1, num_classes={num_classes})")
    
    # Export to ONNX
    print(f"[CRNN Export] Exporting to ONNX: {output_onnx}")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_onnx),
        input_names=["input"],
        output_names=["log_probs"],
        dynamic_axes={
            "input": {0: "batch"},
            "log_probs": {0: "T", 1: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
    )
    
    # Verify ONNX model can be loaded
    try:
        onnx_model = onnx.load(str(output_onnx))
        onnx.checker.check_model(onnx_model)
        print(f"[CRNN Export] ✓ ONNX model validated")
    except Exception as e:
        print(f"[CRNN Export] ⚠ ONNX validation warning: {e}")
    
    # Create metadata JSON
    metadata = {
        "modelType": "crnn",
        "numClasses": num_classes,
        "inputShape": [1, 1, DigitCRNN.INPUT_H, DigitCRNN.INPUT_W],
        "outputShape": [DigitCRNN.T, 1, num_classes],
        "timeSteps": DigitCRNN.T,
        "blankIdx": DigitCRNN.BLANK_IDX,
        "charClasses": char_classes,
        "formats": {
            "decimalPercent": "^\\d{1,2}\\.\\d{2}%$",
            "decimal": "^\\d{1,3}\\.\\d{2}$",
            "percent": "^\\d{1,2}%$",
            "integer": "^\\d{1,6}$",
        },
        "valAccuracy": float(val_accuracy),
    }
    
    print(f"[CRNN Export] Writing metadata: {output_json}")
    with open(output_json, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[CRNN Export] ✓ Export complete!")
    print(f"  - ONNX model: {output_onnx}")
    print(f"  - Metadata:   {output_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Export CRNN model to ONNX format for web deployment"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to CRNN checkpoint (e.g., data/models/crnn_model.pth)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output ONNX path (e.g., crnn_model.onnx)",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Output JSON metadata path (default: same dir as output with .json extension)",
    )
    
    args = parser.parse_args()
    
    try:
        export_crnn_to_onnx(args.model, args.output, args.metadata)
    except Exception as e:
        print(f"[CRNN Export] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
