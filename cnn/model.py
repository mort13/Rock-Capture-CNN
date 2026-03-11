"""
Small CNN for single-character recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):
    """
    Small CNN for single-character recognition.

    Architecture:
        Input: 1x28x28 grayscale
        Conv1: 1->32 channels, 3x3, padding=1  -> 32x28x28
        ReLU + MaxPool(2)                        -> 32x14x14
        Conv2: 32->64 channels, 3x3, padding=1  -> 64x14x14
        ReLU + MaxPool(2)                        -> 64x7x7
        Flatten                                  -> 3136
        FC1: 3136->128, ReLU, Dropout(0.25)
        FC2: 128->num_classes

    Default classes: 0-9 plus '.', '-', '%' = 13 classes
    """

    def __init__(self, num_classes: int = 13):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
