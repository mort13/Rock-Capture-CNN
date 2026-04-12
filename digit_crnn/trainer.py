"""
Standalone training function for the CRNN digit sequence model.

Designed to be called from train_crnn.py (CLI) or imported directly.
Uses CTCLoss + greedy decode for validation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from digit_crnn.model import DigitCRNN
from digit_crnn.dataset import StripDataset, collate_fn, decode_ctc_greedy


def train(
    data_dir: str | Path = "data/strip_training_data",
    output_path: str | Path = "data/models/crnn_model.pth",
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 0.001,
    val_split: float = 0.15,
    device: str | None = None,
) -> None:
    """
    Train the CRNN and save the best checkpoint to *output_path*.

    Validation metric: full-string accuracy (strict) on a held-out split.
    Progress is printed to stdout.
    """
    data_dir = Path(data_dir)
    output_path = Path(output_path)

    # ── Device ────────────────────────────────────────────────
    if device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        _device = torch.device(device)
    print(f"[CRNN trainer] device: {_device}")

    # ── Dataset ───────────────────────────────────────────────
    full_ds = StripDataset(data_dir, augment=False)
    n_total = full_ds.base_len
    if n_total < 4:
        raise RuntimeError(
            f"Not enough training strips in {data_dir} (found {n_total}). "
            "Run generate_synth_strip_data.py first."
        )

    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    train_base, val_base = random_split(
        StripDataset(data_dir, augment=False),
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Augmented train set
    oversample = max(1, 400 // n_train)
    train_ds = StripDataset(data_dir, augment=True, oversample=oversample)
    # Only use the same indices as train_base (no leaking val indices)
    train_ds.samples = [train_ds._base_samples[i] for i in train_base.indices] * oversample

    val_ds = StripDataset(data_dir, augment=False, oversample=1)
    val_ds.samples = [val_ds._base_samples[i] for i in val_base.indices]

    use_cuda = _device.type == "cuda"
    num_workers = 4 if use_cuda else 2

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=use_cuda,
    )

    # ── Model ─────────────────────────────────────────────────
    model = DigitCRNN(num_classes=DigitCRNN.NUM_CLASSES).to(_device)
    criterion = nn.CTCLoss(blank=DigitCRNN.BLANK_IDX, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Mixed precision on GPU
    use_amp = use_cuda
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    print(
        f"[CRNN trainer] {n_train} train + {n_val} val strips "
        f"(oversample ×{oversample}), {epochs} epochs, batch {batch_size}, lr {lr}"
    )

    best_val_acc = -1.0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # ── Training ──────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for images, targets, input_lengths, target_lengths in train_loader:
            images = images.to(_device)
            targets = targets.to(_device)
            input_lengths = input_lengths.to(_device)
            target_lengths = target_lengths.to(_device)

            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    log_probs = model(images)   # (T, batch, C)
                    loss = criterion(log_probs, targets, input_lengths, target_lengths)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                log_probs = model(images)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)

        # ── Validation ────────────────────────────────────────
        model.eval()
        exact_correct = 0
        char_correct = 0
        char_total = 0
        total_seqs = 0

        with torch.no_grad():
            for images, targets, input_lengths, target_lengths in val_loader:
                images = images.to(_device)
                log_probs = model(images)  # (T, batch, C)
                # Greedy decode per sample
                probs = log_probs.argmax(dim=2)  # (T, batch)
                probs = probs.permute(1, 0)      # (batch, T)

                tgt_offset = 0
                for b_idx in range(images.size(0)):
                    tgt_len = target_lengths[b_idx].item()
                    tgt_seq = targets[tgt_offset: tgt_offset + tgt_len].tolist()
                    tgt_offset += tgt_len

                    pred_seq = decode_ctc_greedy(probs[b_idx])
                    # Ground truth string
                    from digit_crnn.dataset import IDX_TO_CHAR
                    gt_str = "".join(IDX_TO_CHAR.get(i, "?") for i in tgt_seq)

                    if pred_seq == gt_str:
                        exact_correct += 1

                    # Character-level accuracy (aligned by min length)
                    for p, g in zip(pred_seq, gt_str):
                        char_correct += int(p == g)
                    char_total += max(len(pred_seq), len(gt_str))

                    total_seqs += 1

        val_exact_acc = exact_correct / max(total_seqs, 1)
        val_char_acc = char_correct / max(char_total, 1)

        print(
            f"Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  "
            f"val_exact={val_exact_acc:.3f}  val_char={val_char_acc:.3f}"
        )

        if val_exact_acc > best_val_acc:
            best_val_acc = val_exact_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "char_classes": DigitCRNN.CHAR_CLASSES,
                    "num_classes": DigitCRNN.NUM_CLASSES,
                    "val_accuracy": val_exact_acc,
                },
                str(output_path),
            )
            print(f"  → saved best model (val_exact={val_exact_acc:.3f})")

    print(f"\n[CRNN trainer] Done. Best val_exact={best_val_acc:.3f}  →  {output_path}")
