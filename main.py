"""
Rock Capture CNN - Entry point.
Game HUD OCR using dynamic anchor + CNN digit recognition.
"""

import sys
import argparse
from pathlib import Path

# Ensure the project root is on sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# torch must be imported before PyQt6 to avoid a DLL conflict on Windows
import torch  # noqa: F401

from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow


def main():
    parser = argparse.ArgumentParser(description="Rock Capture CNN")
    parser.add_argument(
        "--onnx", action="store_true",
        help="Use ONNX Runtime for CRNN inference instead of PyTorch"
    )
    args, qt_args = parser.parse_known_args()

    if args.onnx:
        from digit_crnn.onnx_predictor import CRNNOnnxPredictor
        crnn_predictor = CRNNOnnxPredictor()
        print("[main] CRNN backend: ONNX Runtime")
    else:
        from digit_crnn.predictor import CRNNPredictor
        crnn_predictor = CRNNPredictor()
        print("[main] CRNN backend: PyTorch")

    app = QApplication([sys.argv[0]] + qt_args)
    app.setStyle("Fusion")
    window = MainWindow(crnn_predictor=crnn_predictor)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
