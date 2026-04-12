"""
train_crnn.py — CLI entry point for training the CRNN digit sequence model.

Usage:
    python train_crnn.py
    python train_crnn.py --epochs 120 --batch-size 64
    python train_crnn.py --data-dir data/strip_training_data --output data/models/crnn_model.pth
    python train_crnn.py --device cpu
"""

import argparse
from digit_crnn.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the CRNN digit sequence model."
    )
    parser.add_argument(
        "--data-dir", default="data/strip_training_data",
        help="Folder containing strip training images (default: data/strip_training_data)",
    )
    parser.add_argument(
        "--output", default="data/models/crnn_model.pth",
        help="Path to save the best model checkpoint (default: data/models/crnn_model.pth)",
    )
    parser.add_argument(
        "--epochs", type=int, default=60,
        help="Number of training epochs (default: 60)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Mini-batch size (default: 32)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Adam learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.15,
        help="Fraction of data held out for validation (default: 0.15)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device string, e.g. 'cuda' or 'cpu' (default: auto-detect)",
    )
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        device=args.device,
    )


if __name__ == "__main__":
    main()
