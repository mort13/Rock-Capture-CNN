"""
PyTorch Dataset for labeled word / resource-name images.
Loads from a directory structure: word_training_data/{class_label}/*.{png,jpg}

Supports online data augmentation for few-shot training (even 1 image
per class) via random brightness, noise, blur, erosion/dilation, and
small affine shifts.
"""

import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from pathlib import Path

from word_cnn.model import WordCNN

_IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")


def _autocrop_text(img: np.ndarray) -> np.ndarray:
    """Crop to the bounding box containing the bright text core.

    Uses a fraction of the image's own peak brightness as the cut-off so
    the crop adapts to dim vs bright images. Bleed/glow typically sits at
    10-20% of peak, while actual text strokes are at 60-100% — so picking
    anything above 35% of peak reliably separates text from bleed without
    needing a hard fixed threshold.
    """
    peak = int(img.max())
    if peak < 10:
        return img  # blank image
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


def resize_pad(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Auto-crop to text content, then scale to fit within target_w×target_h
    while preserving aspect ratio.  Short words (tin) still fill the canvas
    height and leave plenty of black on the right; long words (inert materials)
    scale down to fit the width without being clipped.  The text is left-aligned
    horizontally and vertically centred on the canvas.
    """
    img = _autocrop_text(img)
    h, w = img.shape[:2]
    scale = min(target_h / max(h, 1), target_w / max(w, 1))
    new_h = max(1, round(h * scale))
    new_w = max(1, round(w * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, 0:new_w] = resized
    return canvas


def _augment(img: np.ndarray) -> np.ndarray:
    """Apply random augmentations to a grayscale uint8 image."""
    h, w = img.shape[:2]

    # --- brightness / contrast ---
    alpha = random.uniform(0.7, 1.3)   # contrast
    beta = random.randint(-30, 30)      # brightness
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # --- Gaussian noise ---
    if random.random() < 0.5:
        sigma = random.uniform(3, 15)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # --- blur ---
    if random.random() < 0.4:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # --- erosion / dilation (simulate font weight variation) ---
    if random.random() < 0.3:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        if random.random() < 0.5:
            img = cv2.erode(img, kernel, iterations=1)
        else:
            img = cv2.dilate(img, kernel, iterations=1)

    # --- small affine shift / scale ---
    if random.random() < 0.6:
        dx = random.uniform(-0.08, 0.08) * w
        dy = random.uniform(-0.08, 0.08) * h
        sx = random.uniform(0.92, 1.08)
        sy = random.uniform(0.95, 1.05)
        M = np.array([[sx, 0, dx], [0, sy, dy]], dtype=np.float32)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return img


class WordDataset(Dataset):
    """
    Loads word images from directory structure:
        word_training_data/
            iron/     -> img_001.png, ...
            gold/     -> ...
            titanium/ -> ...

    Each image is loaded as grayscale, resized to INPUT_H × INPUT_W,
    normalised to [0, 1].

    Parameters
    ----------
    data_dir : path to the root training-data folder.
    word_classes : explicit class list; auto-discovered if None.
    augment : apply random augmentation on each __getitem__ call.
    oversample : repeat the sample list this many times so that even a
        single image per class yields enough updates per epoch.
    """

    def __init__(
        self,
        data_dir: str | Path,
        word_classes: list[str] | None = None,
        augment: bool = False,
        oversample: int = 1,
    ):
        self.data_dir = Path(data_dir)
        if word_classes is not None:
            self.word_classes = list(word_classes)
        else:
            self.word_classes = self._discover_classes()
        self.class_to_idx = {c: i for i, c in enumerate(self.word_classes)}
        self.augment = augment
        self._base_samples: list[tuple[Path, int]] = []
        self._scan_directory()
        # oversample: repeat the sample list so small datasets get enough
        # gradient updates per epoch
        self.samples = self._base_samples * max(1, oversample)

    def _discover_classes(self) -> list[str]:
        """Discover classes from subdirectory names that contain at least one image."""
        classes = []
        if not self.data_dir.exists():
            return classes
        for d in sorted(self.data_dir.iterdir()):
            if d.is_dir() and any(
                f for ext in _IMG_EXTS for f in d.glob(ext)
            ):
                classes.append(d.name)
        return classes

    def _scan_directory(self) -> None:
        for cls in self.word_classes:
            class_dir = self.data_dir / cls
            if not class_dir.exists():
                continue
            idx = self.class_to_idx[cls]
            for ext in _IMG_EXTS:
                for img_path in sorted(class_dir.glob(ext)):
                    self._base_samples.append((img_path, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((WordCNN.INPUT_H, WordCNN.INPUT_W), dtype=np.uint8)
        img = resize_pad(img, WordCNN.INPUT_W, WordCNN.INPUT_H)
        if self.augment:
            img = _augment(img)
        tensor = torch.from_numpy(img).float() / 255.0
        tensor = tensor.unsqueeze(0)  # (1, H, W)
        return tensor, label

    @property
    def base_len(self) -> int:
        """Number of unique (non-oversampled) samples."""
        return len(self._base_samples)

    def get_class_counts(self) -> dict[str, int]:
        counts = {c: 0 for c in self.word_classes}
        for _, idx in self._base_samples:
            counts[self.word_classes[idx]] += 1
        return counts
