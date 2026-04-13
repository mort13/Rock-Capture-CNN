"""
Export a trained WordCNN checkpoint (.pth) to ONNX format for inference.

Usage:
    python cnn_word_export_onnx.py                    # uses defaults
    python cnn_word_export_onnx.py --model data/models/word_model.pth --output word_cnn.onnx
"""

import argparse
import json
from pathlib import Path

import torch

from word_cnn.model import WordCNN


def export(model_path: str, output_path: str) -> None:
    """
    Load a WordCNN checkpoint and export to ONNX format.
    
    Args:
        model_path: Path to the .pth checkpoint
        output_path: Path to write the .onnx file
    """
    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract metadata from checkpoint
    state_dict = checkpoint["model_state_dict"]
    word_classes: list[str] = checkpoint.get("word_classes", [])
    num_classes = checkpoint.get("num_classes", len(word_classes))
    val_accuracy = checkpoint.get("val_accuracy", None)

    if num_classes < 1:
        raise ValueError("No classes found in checkpoint")

    # Create model and load weights
    model = WordCNN(num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.eval()

    # Dummy input for ONNX export (batch=1, 1 channel, 32×256 grayscale)
    dummy_input = torch.randn(1, 1, WordCNN.INPUT_H, WordCNN.INPUT_W)

    print(f"Exporting WordCNN with {num_classes} classes...")
    print(f"  Word classes: {word_classes}")
    if val_accuracy is not None:
        print(f"  Validation accuracy: {val_accuracy:.2%}")

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )

    # Write companion metadata JSON
    meta_path = Path(output_path).with_suffix(".json")
    metadata = {
        "numClasses": num_classes,
        "inputShape": [1, 1, WordCNN.INPUT_H, WordCNN.INPUT_W],
        "wordClasses": word_classes,
        "valAccuracy": val_accuracy,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Exported ONNX model to: {output_path}")
    print(f"Metadata written to:    {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export WordCNN to ONNX format")
    parser.add_argument(
        "--model",
        default="data/models/word_model.pth",
        help="Path to the WordCNN .pth checkpoint (default: data/models/word_model.pth)",
    )
    parser.add_argument(
        "--output",
        default="word_cnn.onnx",
        help="Output ONNX file path (default: word_cnn.onnx)",
    )
    args = parser.parse_args()
    export(args.model, args.output)
