"""
PyTorch Dataset for labeled character images.
Loads from training_data/{class_label}/ directory structure.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


def parse_char_classes(chars_str: str) -> list[str]:
    """
    Parse character class string into tokens, handling multi-character tokens.
    
    Multi-character tokens:
    - "empty": represents empty fields
    """
    multi_char_tokens = {"empty"}
    tokens = []
    i = 0
    while i < len(chars_str):
        matched = False
        for token in multi_char_tokens:
            if chars_str[i:i+len(token)] == token:
                tokens.append(token)
                i += len(token)
                matched = True
                break
        if not matched:
            tokens.append(chars_str[i])
            i += 1
    return tokens


class CharacterDataset(Dataset):
    """
    Loads character images from directory structure:
        training_data/
            0/   -> img_001.png, ...
            1/   -> ...
            dot/ -> ...  (maps to '.')
            dash/ -> ... (maps to '-')
            percent/ -> ... (maps to '%')

    Each image is loaded as grayscale, resized to 28x28, normalized to [0,1].
    """

    CHAR_DIR_MAP = {
        ".": "dot",
        "-": "dash",
        "%": "percent",
        ",": "comma",
        "empty": "empty",
    }
    DIR_CHAR_MAP = {v: k for k, v in CHAR_DIR_MAP.items()}

    def __init__(self, data_dir: str | Path, char_classes: str = "0123456789.-%empty"):
        self.data_dir = Path(data_dir)
        # Parse char_classes to handle multi-char tokens like "empty"
        self.char_list = parse_char_classes(char_classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.char_list)}
        self.samples: list[tuple[Path, int]] = []
        self._scan_directory()

    def _scan_directory(self) -> None:
        """Scan training_data directory and populate samples list."""
        for char in self.char_list:
            dir_name = self.CHAR_DIR_MAP.get(char, char)
            class_dir = self.data_dir / dir_name
            if not class_dir.exists():
                continue
            idx = self.class_to_idx[char]
            for img_path in sorted(class_dir.glob("*.png")):
                self.samples.append((img_path, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((28, 28), dtype=np.uint8)
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(img).float() / 255.0
        tensor = tensor.unsqueeze(0)  # (1, 28, 28)
        return tensor, label

    def get_class_counts(self) -> dict[str, int]:
        """Return count of samples per class."""
        counts = {c: 0 for c in self.char_list}
        for _, idx in self.samples:
            counts[self.char_list[idx]] += 1
        return counts
