"""
Training thread for the word-classification CNN.
Mirrors cnn.trainer but uses WordCNN + WordDataset.

Also exposes a standalone train() function for CLI use (no Qt required).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path

# Qt is only needed by WordTrainerThread (GUI mode); import lazily so the
# standalone train() function works without PyQt6 installed.
try:
    from PyQt6.QtCore import QThread, pyqtSignal
    _QT_AVAILABLE = True
except ImportError:  # pragma: no cover
    QThread = object  # type: ignore[misc,assignment]
    pyqtSignal = None  # type: ignore[assignment]
    _QT_AVAILABLE = False

from word_cnn.model import WordCNN
from word_cnn.dataset import WordDataset


def discover_word_classes(data_dir: str | Path) -> list[str]:
    """
    Auto-discover available word classes from word_training_data subdirectories.
    Returns a sorted list of class names (includes directories even if empty).
    """
    data_dir = Path(data_dir)
    classes = []
    if data_dir.exists():
        for subdir in sorted(data_dir.iterdir()):
            if subdir.is_dir():
                classes.append(subdir.name)
    return sorted(classes)


def train(
    data_dir: str | Path = "data/word_training_data",
    output_path: str | Path = "data/models/word_model.pth",
    word_classes: list[str] | None = None,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 0.001,
    device: str | None = None,
) -> None:
    """
    Train a WordCNN and save the best checkpoint to *output_path*.

    When *word_classes* is None, all subdirectories of *data_dir* are used.
    Progress is printed to stdout; no Qt required.
    """
    data_dir = Path(data_dir)
    output_path = Path(output_path)

    # ── Device ────────────────────────────────────────────────────────────────
    if device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        _device = torch.device(device)
    print(f"[WordCNN trainer] device: {_device}")

    # ── Classes ───────────────────────────────────────────────────────────────
    if word_classes is None:
        word_classes = discover_word_classes(data_dir)
    if len(word_classes) < 2:
        raise RuntimeError(f"Need at least 2 classes, found {len(word_classes)}.")

    # ── Dataset ───────────────────────────────────────────────────────────────
    probe = WordDataset(data_dir, word_classes, augment=False, oversample=1)
    n_real = probe.base_len
    if n_real < 2:
        raise RuntimeError(
            f"Not enough samples ({n_real}) in {data_dir}. "
            "Run generate_synth_word_data.py first."
        )
    word_classes = probe.word_classes  # update in case dataset filtered any
    num_classes = len(word_classes)

    oversample = max(1, 200 // n_real)

    train_ds = WordDataset(data_dir, word_classes, augment=True, oversample=oversample)
    val_ds   = WordDataset(data_dir, word_classes, augment=False, oversample=1)

    use_cuda = _device.type == "cuda"
    num_workers = 4 if use_cuda else 2

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_cuda,
    )

    print(
        f"[WordCNN trainer] {n_real} real images "
        f"(×{oversample} oversample), {num_classes} classes, "
        f"{epochs} epochs, batch {batch_size}, lr {lr}"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = WordCNN(num_classes=num_classes).to(_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler    = GradScaler() if use_cuda else None

    best_val_acc = 0.0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(_device)
            labels = labels.to(_device)
            optimizer.zero_grad()
            if scaler:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(_device)
                labels = labels.to(_device)
                _, predicted = torch.max(model(images), 1)
                total   += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = correct / total if total > 0 else 0.0

        marker = " *" if val_acc > best_val_acc else ""
        print(
            f"  epoch {epoch:3d}/{epochs}  loss {avg_loss:.4f}  "
            f"val_acc {val_acc:.4f}{marker}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "word_classes": word_classes,
                    "num_classes": num_classes,
                    "val_accuracy": val_acc,
                },
                str(output_path),
            )

    print(f"\n[WordCNN trainer] best val_acc {best_val_acc:.4f} -> {output_path}")


class WordTrainerThread(QThread):
    """
    Trains a WordCNN in a background thread.

    Signals:
        epoch_completed(int, float, float): (epoch, train_loss, val_accuracy)
        training_finished(str): path to saved model
        training_failed(str): error message
        progress_update(str): status text
    """
    epoch_completed = pyqtSignal(int, float, float)
    training_finished = pyqtSignal(str)
    training_failed = pyqtSignal(str)
    progress_update = pyqtSignal(str)

    def __init__(
        self,
        data_dir: str | Path,
        output_model_path: str | Path,
        word_classes: list[str] | None = None,
        num_epochs: int = 60,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        val_split: float = 0.2,
        parent=None,
    ):
        super().__init__(parent)
        self.data_dir = Path(data_dir)
        self.output_path = Path(output_model_path)
        self.word_classes = word_classes
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.val_split = val_split
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            self.progress_update.emit("Loading dataset...")
            # Auto-discover classes if None
            if self.word_classes is None:
                self.word_classes = discover_word_classes(self.data_dir)
            
            # Load without augmentation first to discover classes + count
            raw_dataset = WordDataset(self.data_dir, self.word_classes)

            n_real = raw_dataset.base_len
            if n_real < 2:
                self.training_failed.emit(
                    f"Not enough samples ({n_real}). Need at least 2."
                )
                return

            word_classes = raw_dataset.word_classes
            num_classes = len(word_classes)
            if num_classes < 2:
                self.training_failed.emit(
                    f"Need at least 2 classes, found {num_classes}."
                )
                return

            # Compute oversample factor so that effective dataset has
            # at least ~200 samples per epoch even from a handful of images
            oversample = max(1, 200 // n_real)

            # Build train set (augmented + oversampled) and val set (clean)
            train_ds = WordDataset(
                self.data_dir, word_classes, augment=True, oversample=oversample
            )
            val_ds = WordDataset(
                self.data_dir, word_classes, augment=False, oversample=1
            )

            # Use num_workers for parallel CPU data loading
            # pin_memory speeds up GPU transfer if CUDA is available
            use_cuda = torch.cuda.is_available()
            num_workers = 4 if use_cuda else 2  # Adjust based on your CPU cores
            
            train_loader = DataLoader(
                train_ds, batch_size=self.batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=use_cuda
            )
            val_loader = DataLoader(
                val_ds, batch_size=self.batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=use_cuda
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = WordCNN(num_classes=num_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            
            # Mixed precision training for GPU speedup
            scaler = GradScaler() if device.type == 'cuda' else None

            self.progress_update.emit(
                f"Training on {n_real} images (×{oversample} oversample, augmented), "
                f"validating on {n_real} clean ({num_classes} classes, device: {device})..."
            )

            best_val_acc = 0.0
            for epoch in range(self.num_epochs):
                if self._stop_requested:
                    self.progress_update.emit("Training stopped by user.")
                    break

                model.train()
                total_loss = 0.0
                for images, labels in train_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    
                    if scaler:  # Mixed precision on GPU
                        with autocast():
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:  # Standard training on CPU
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                    
                    total_loss += loss.item()
                avg_loss = total_loss / len(train_loader)

                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                val_acc = correct / total if total > 0 else 0.0

                self.epoch_completed.emit(epoch + 1, avg_loss, val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.output_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "word_classes": word_classes,
                            "num_classes": num_classes,
                            "val_accuracy": val_acc,
                        },
                        str(self.output_path),
                    )

            self.training_finished.emit(str(self.output_path))

        except Exception as e:
            self.training_failed.emit(str(e))
