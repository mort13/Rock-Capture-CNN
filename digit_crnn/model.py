"""
CRNN (Convolutional Recurrent Neural Network) for digit sequence recognition.

Takes a full strip image (no explicit segmentation) and outputs a variable-length
digit string via CTC decoding.

Input:  (batch, 1, 32, 256)  — grayscale, H=32, W=256
Output: (T=64, batch, num_classes)  — log-softmax over CTC alphabet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCRNN(nn.Module):
    """
    CNN backbone + BiLSTM + CTC head for variable-length digit string recognition.

    CTC alphabet (blank=0):
        blank  = 0
        '0'–'9'= 1–10
        '.'    = 11
        '%'    = 12

    Architecture:
        CNN block 1: Conv(1→32,  3×3, pad=1) + BN + ReLU → MaxPool(2,2)  → 32×16×128
        CNN block 2: Conv(32→64, 3×3, pad=1) + BN + ReLU → MaxPool(2,2)  → 64×8×64
        CNN block 3: Conv(64→128,3×3, pad=1) + BN + ReLU → MaxPool((2,1))→ 128×4×64
        CNN block 4: Conv(128→128,3×3,pad=1) + BN + ReLU                  → 128×4×64
        Reshape: (batch, 128, 4, 64) → (batch, 512, 64) → permute → (64, batch, 512)
        BiLSTM × 2: input 512, hidden 256, dropout 0.3  → (64, batch, 512)
        FC:  512 → num_classes
    """

    INPUT_H = 32
    INPUT_W = 256
    # Number of time-steps produced by the CNN backbone (W // 4 due to two
    # maxpool-width halvings in blocks 1 and 2; blocks 3 and 4 do not pool width)
    T = INPUT_W // 4  # = 64

    BLANK_IDX = 0
    CHAR_CLASSES = "0123456789.%"   # indices 1-12; blank is 0
    NUM_CLASSES = len(CHAR_CLASSES) + 1  # 13

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # ── CNN backbone ───────────────────────────────────────
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # → 32 × 16 × 128

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # → 64 × 8 × 64

            # Block 3 — pool height only to preserve time-steps
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),        # → 128 × 4 × 64

            # Block 4 — no spatial pooling
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
                                         # → 128 × 4 × 64
        )

        # ── BiLSTM ────────────────────────────────────────────
        # Input per time-step: 128 channels × 4 height rows = 512
        rnn_input_size = 128 * 4  # 512
        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
            batch_first=False,  # (T, batch, features)
        )

        # ── CTC head ──────────────────────────────────────────
        self.fc = nn.Linear(256 * 2, num_classes)  # bidirectional → 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, H, W)
        Returns:
            log_probs: (T, batch, num_classes)  — for nn.CTCLoss
        """
        # CNN: (batch, 1, 32, 256) → (batch, 128, 4, 64)
        features = self.cnn(x)

        # Reshape to sequence: (batch, 128, 4, T) → (batch, 512, T) → (T, batch, 512)
        b, c, h, t = features.shape
        features = features.view(b, c * h, t)     # (batch, 512, T)
        features = features.permute(2, 0, 1)       # (T, batch, 512)

        # BiLSTM: (T, batch, 512) → (T, batch, 512)
        rnn_out, _ = self.rnn(features)

        # FC + log-softmax: (T, batch, num_classes)
        logits = self.fc(rnn_out)
        log_probs = F.log_softmax(logits, dim=2)
        return log_probs
