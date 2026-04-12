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
from word_cnn.model import WordCNN


def export(model_path: str, output_path: str) -> None:
    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    state_dict = checkpoint["model_state_dict"]
    has_conv3 = "conv3.weight" in state_dict

    if has_conv3:
        # WordCNN model: classification over word labels
        word_classes: list[str] = checkpoint.get("word_classes", [])
        num_classes = checkpoint.get("num_classes", len(word_classes))
        model = WordCNN(num_classes=num_classes)
        dummy = torch.randn(1, 1, 32, 256)
        print(f"Detected WordCNN model ({num_classes} word classes)")
    else:
        # DigitCNN model: classification over character classes
        char_classes: str = checkpoint.get("char_classes", "0123456789.-%empty")
        num_classes = checkpoint.get("num_classes", len(char_classes))
        word_classes = []
        model = DigitCNN(num_classes=num_classes)
        dummy = torch.randn(1, 1, 28, 28)
        print(f"Detected DigitCNN model ({num_classes} char classes)")
    
    model.load_state_dict(state_dict)
    model.eval()

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
    val_accuracy = checkpoint.get("val_accuracy", None)
    meta: dict = {
        "numClasses": num_classes,
        "inputShape": list(dummy.shape),
        "valAccuracy": val_accuracy,
    }
    if has_conv3:
        # WordCNN: store label list under wordClasses; charClasses left empty
        meta["charClasses"] = ""
        meta["wordClasses"] = word_classes
        print(f"  Word classes ({num_classes}): {word_classes}")
    else:
        meta["charClasses"] = char_classes
        print(f"  Char classes ({num_classes}): {char_classes}")
    if val_accuracy is not None:
        print(f"  Validation accuracy:  {val_accuracy:.2%}")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Exported ONNX model to: {output_path}")
    print(f"Metadata written to:    {meta_path}")


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
