"""
CNN training thread for Rock Capture CNN.
Runs the training loop in a QThread with progress signals.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from PyQt6.QtCore import QThread, pyqtSignal
from pathlib import Path

from cnn.model import DigitCNN
from cnn.dataset import CharacterDataset


def discover_char_classes(data_dir: str | Path) -> tuple[list[str], str]:
    """
    Auto-discover available character classes from training_data subdirectories.
    Returns (class_list, class_string) where:
    - class_list: list of individual class tokens including multi-char tokens like 'empty'
    - class_string: string representation for CharacterDataset
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return [], ""
    
    # Reverse mapping for special chars
    char_dir_map = {
        "dot": ".",
        "dash": "-",
        "percent": "%",
        "comma": ",",
        "empty": "empty",
    }
    
    classes = []
    for subdir in sorted(data_dir.iterdir()):
        if subdir.is_dir():
            dir_name = subdir.name
            # Convert directory name back to character/token
            char = char_dir_map.get(dir_name, dir_name)
            classes.append(char)
    
    # Sort: digits first, then special chars
    digits = [c for c in classes if c.isdigit()]
    special = [c for c in classes if not c.isdigit()]
    sorted_classes = sorted(digits) + sorted(special)
    
    # Build string: single chars directly, 'empty' as special token
    class_str = "".join(c for c in sorted_classes if c != "empty")
    if "empty" in sorted_classes:
        class_str += "empty"
    
    return sorted_classes, class_str


class TrainerThread(QThread):
    """
    Trains the DigitCNN in a background thread.

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
        char_classes: str = "0123456789.-%empty",
        num_epochs: int = 30,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        val_split: float = 0.2,
        parent=None,
    ):
        super().__init__(parent)
        self.data_dir = Path(data_dir)
        self.output_path = Path(output_model_path)
        self.char_classes = char_classes
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
            
            # Auto-discover classes from training_data directory if using default
            num_classes = None
            if not self.char_classes or self.char_classes == "0123456789.-%empty":
                discovered_list, discovered_str = discover_char_classes(self.data_dir)
                if discovered_list:
                    self.char_classes = discovered_str
                    num_classes = len(discovered_list)
            
            dataset = CharacterDataset(self.data_dir, self.char_classes)
            
            # If not auto-discovered, count actual classes in dataset
            if num_classes is None:
                # For non-auto-discovered case, we need to count carefully
                # If "empty" is in the string, it's one class, not 5
                if "empty" in self.char_classes:
                    base_chars = self.char_classes.replace("empty", "")
                    num_classes = len(base_chars) + 1
                else:
                    num_classes = len(self.char_classes)

            if len(dataset) < 10:
                self.training_failed.emit(
                    f"Not enough samples ({len(dataset)}). Need at least 10."
                )
                return

            val_size = max(1, int(len(dataset) * self.val_split))
            train_size = len(dataset) - val_size
            train_set, val_set = random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(
                train_set, batch_size=self.batch_size, shuffle=True,
                num_workers=4, pin_memory=True
            )
            val_loader = DataLoader(
                val_set, batch_size=self.batch_size, shuffle=False,
                num_workers=4, pin_memory=True
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = DigitCNN(num_classes=num_classes).to(device)
            
            # Mixed precision training for GPU speedup
            scaler = GradScaler() if device.type == 'cuda' else None
            
            # Class weights to help distinguish confusing digits (0, 8, 9)
            class_weights = torch.ones(num_classes, device=device)
            # Increase weight on easily-confused digits
            if num_classes > 0:
                class_weights[0] = 2.0  # '0'
            if num_classes > 8:
                class_weights[8] = 2.0  # '8'
            if num_classes > 9:
                class_weights[9] = 2.0  # '9'
            
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

            self.progress_update.emit(
                f"Training on {train_size} samples, validating on {val_size} "
                f"(device: {device})..."
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
                            "char_classes": self.char_classes,
                            "num_classes": num_classes,
                            "val_accuracy": val_acc,
                        },
                        str(self.output_path),
                    )

            self.training_finished.emit(str(self.output_path))

        except Exception as e:
            self.training_failed.emit(str(e))
