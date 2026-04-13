"""
train_word_cnn.py — CLI entry point for training the word-classification CNN.

Usage:
    python train_word_cnn.py
    python train_word_cnn.py --epochs 120 --batch-size 64
    python train_word_cnn.py --data-dir data/word_training_data --output data/models/word_model.pth
    python train_word_cnn.py --classes iron gold copper empty
    python train_word_cnn.py --device cpu
"""

import argparse
from word_cnn.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the word-classification CNN."
    )
    parser.add_argument(
        "--data-dir", default="data/word_training_data",
        help="Folder containing per-class image subdirectories (default: data/word_training_data)",
    )
    parser.add_argument(
        "--output", default="data/models/word_model.pth",
        help="Path to save the best model checkpoint (default: data/models/word_model.pth)",
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
        "--classes", nargs="+", default=None,
        help="Limit training to specific class names (default: auto-discover all subdirs)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device string, e.g. 'cuda' or 'cpu' (default: auto-detect)",
    )
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_path=args.output,
        word_classes=args.classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
