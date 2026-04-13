"""
PyTorch Dataset for CRNN strip training data.

Loads grayscale strip images from data/strip_training_data/*.png.
Filename convention: <label>_<n:06d>.png  (label split on last underscore).

Examples:
    42.50_000001.png  → label "42.50"
    75%_000042.png    → label "75%"
    100_000123.png    → label "100"
"""

import cv2
import numpy as np
import random
import torch
from torch import Tensor
from torch.utils.data import Dataset
from pathlib import Path

from digit_crnn.model import DigitCRNN

_IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")

# CTC alphabet — index 0 is reserved for blank
CHAR_TO_IDX: dict[str, int] = {c: i + 1 for i, c in enumerate(DigitCRNN.CHAR_CLASSES)}
IDX_TO_CHAR: dict[int, str] = {v: k for k, v in CHAR_TO_IDX.items()}


def _autocrop_text(img: np.ndarray) -> np.ndarray:
    """Crop to the bounding box of bright text content (35% of peak brightness)."""
    peak = int(img.max())
    if peak < 10:
        return img
    bright = max(8, round(peak * 0.35))
    col_max = img.max(axis=0)
    row_max = img.max(axis=1)
    cols = np.where(col_max > bright)[0]
    rows = np.where(row_max > bright)[0]
    if cols.size == 0 or rows.size == 0:
        return img
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    return img[y0:y1, x0:x1]


def resize_pad(img: np.ndarray, target_w: int = DigitCRNN.INPUT_W,
               target_h: int = DigitCRNN.INPUT_H) -> np.ndarray:
    """Auto-crop to text, scale to fit within target_w × target_h (aspect-preserving),
    left-align horizontally, vertically centre."""
    img = _autocrop_text(img)
    h, w = img.shape[:2]
    scale = min(target_h / max(h, 1), target_w / max(w, 1))
    new_h = max(1, round(h * scale))
    new_w = max(1, round(w * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    canvas[y_off: y_off + new_h, 0:new_w] = resized
    return canvas


def _augment(img: np.ndarray) -> np.ndarray:
    """Random augmentations for a grayscale strip."""
    h, w = img.shape[:2]

    alpha = random.uniform(0.7, 1.3)
    beta = random.randint(-30, 30)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    if random.random() < 0.5:
        sigma = random.uniform(3, 15)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if random.random() < 0.4:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    if random.random() < 0.3:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        if random.random() < 0.5:
            img = cv2.erode(img, kernel, iterations=1)
        else:
            img = cv2.dilate(img, kernel, iterations=1)

    if random.random() < 0.6:
        dx = random.uniform(-0.05, 0.05) * w
        dy = random.uniform(-0.05, 0.05) * h
        sx = random.uniform(0.93, 1.07)
        sy = random.uniform(0.95, 1.05)
        M = np.array([[sx, 0, dx], [0, sy, dy]], dtype=np.float32)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return img


def encode_label(text: str) -> list[int]:
    """Encode a label string to a list of CTC indices (1-based, 0=blank)."""
    return [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]


def decode_ctc_greedy(indices: list[int] | torch.Tensor) -> str:
    """Greedy CTC decode: collapse consecutive duplicates then remove blanks."""
    if isinstance(indices, torch.Tensor):
        indices = indices.tolist()
    result = []
    prev = -1
    for idx in indices:
        if idx != prev:
            if idx != DigitCRNN.BLANK_IDX:
                char = IDX_TO_CHAR.get(idx)
                if char is not None:
                    result.append(char)
            prev = idx
    return "".join(result)


class StripDataset(Dataset):
    """
    Dataset of 32px-tall grayscale strip images for CRNN training.

    Directory layout (flat):
        strip_training_data/
            42.50_000001.png
            75%_000042.png
            100_000123.png
            ...

    Label is extracted from the filename by splitting on the *last* underscore.
    """

    def __init__(
        self,
        data_dir: str | Path,
        augment: bool = False,
        oversample: int = 1,
    ):
        self.data_dir = Path(data_dir)
        self.augment = augment
        self._base_samples: list[tuple[Path, str]] = []
        self._scan_directory()
        self.samples = self._base_samples * max(1, oversample)

    def _scan_directory(self) -> None:
        if not self.data_dir.exists():
            return
        for ext in _IMG_EXTS:
            for path in sorted(self.data_dir.glob(ext)):
                # Parse label: split on last '_'
                parts = path.stem.rsplit("_", 1)
                if len(parts) != 2:
                    continue
                label = parts[0]
                # Validate all chars are in our alphabet
                if all(c in CHAR_TO_IDX for c in label):
                    self._base_samples.append((path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        path, label = self.samples[idx]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((DigitCRNN.INPUT_H, DigitCRNN.INPUT_W), dtype=np.uint8)
        img = resize_pad(img)
        if self.augment:
            img = _augment(img)
        tensor = torch.from_numpy(img).float() / 255.0
        tensor = tensor.unsqueeze(0)  # (1, H, W)
        label_tensor = torch.tensor(encode_label(label), dtype=torch.long)
        return tensor, label_tensor

    @property
    def base_len(self) -> int:
        return len(self._base_samples)


def collate_fn(
    batch: list[tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Collate variable-length label sequences for CTCLoss.

    Returns:
        images:         (batch, 1, H, W)
        targets:        (sum of target lengths,)  — flat concatenation
        input_lengths:  (batch,)  — all equal to DigitCRNN.T
        target_lengths: (batch,)
    """
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    target_lengths = torch.tensor([len(lb) for lb in labels], dtype=torch.long)
    targets = torch.cat(labels, dim=0)
    input_lengths = torch.full((len(images),), DigitCRNN.T, dtype=torch.long)
    return images, targets, input_lengths, target_lengths
