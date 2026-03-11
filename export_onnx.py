"""
Export a trained DigitCNN checkpoint (.pth) to ONNX format for browser inference.

Usage:
    python export_onnx.py                          # uses default paths
    python export_onnx.py --model path/to/model.pth --output digit_cnn.onnx
"""

import argparse
import json
from pathlib import Path

import torch

from cnn.model import DigitCNN


def export(model_path: str, output_path: str) -> None:
    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    char_classes = checkpoint.get("char_classes", "0123456789.-%")
    num_classes = checkpoint.get("num_classes", len(char_classes))
    val_accuracy = checkpoint.get("val_accuracy", None)

    model = DigitCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Dummy input: batch of 1, single channel, 28x28
    dummy = torch.randn(1, 1, 28, 28)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    # Write a companion metadata JSON next to the ONNX file
    meta_path = Path(output_path).with_suffix(".json")
    meta = {
        "charClasses": char_classes,
        "numClasses": num_classes,
        "inputShape": [1, 1, 28, 28],
        "valAccuracy": val_accuracy,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Exported ONNX model to: {output_path}")
    print(f"Metadata written to:    {meta_path}")
    print(f"  Classes ({num_classes}): {char_classes}")
    if val_accuracy is not None:
        print(f"  Validation accuracy:  {val_accuracy:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DigitCNN to ONNX")
    parser.add_argument(
        "--model",
        default="data/models/mass_model.pth",
        help="Path to the .pth checkpoint (default: data/models/mass_model.pth)",
    )
    parser.add_argument(
        "--output",
        default="digit_cnn.onnx",
        help="Output ONNX file path (default: digit_cnn.onnx)",
    )
    args = parser.parse_args()
    export(args.model, args.output)
