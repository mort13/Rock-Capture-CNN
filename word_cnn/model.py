"""
CNN for whole-word / resource-name classification.

Input:  1-channel grayscale image, height normalised to 32px, width up to 256px.
Output: logits over N word classes (e.g. resource names).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WordCNN(nn.Module):
    """
    Small CNN for whole-word classification.

    Images are height-normalised (text always 32px tall) then padded/cropped
    to INPUT_W. This preserves word length as a discriminating feature —
    short words (Tin, Iron) occupy a small fraction of the canvas while
    long words (Copper, Beryl) occupy more, making them easy to tell apart.

    Architecture:
        Input:  1 × 32 × 256   (grayscale, H × W)
        Conv1:  1→32, 3×3, pad=1  → 32×32×256  → pool → 32×16×128
        Conv2:  32→64, 3×3, pad=1 → 64×16×128  → pool → 64×8×64
        Conv3:  64→128, 3×3, pad=1→ 128×8×64   → pool → 128×4×32
        Flatten → 128*4*32 = 16384
        FC1:  16384→256, ReLU, Dropout(0.3)
        FC2:  256→num_classes
    """

    INPUT_H = 32
    INPUT_W = 256

    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        _flat = 128 * (self.INPUT_H // 8) * (self.INPUT_W // 8)  # 128*4*32 = 16384
        self.fc1 = nn.Linear(_flat, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # → 32×16×128
        x = self.pool(F.relu(self.conv2(x)))   # → 64×8×64
        x = self.pool(F.relu(self.conv3(x)))   # → 128×4×32
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
