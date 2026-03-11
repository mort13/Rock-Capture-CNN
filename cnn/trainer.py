"""
CNN training thread for Rock Capture CNN.
Runs the training loop in a QThread with progress signals.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from PyQt6.QtCore import QThread, pyqtSignal
from pathlib import Path

from cnn.model import DigitCNN
from cnn.dataset import CharacterDataset


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
        char_classes: str = "0123456789.-%",
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
            dataset = CharacterDataset(self.data_dir, self.char_classes)

            if len(dataset) < 10:
                self.training_failed.emit(
                    f"Not enough samples ({len(dataset)}). Need at least 10."
                )
                return

            val_size = max(1, int(len(dataset) * self.val_split))
            train_size = len(dataset) - val_size
            train_set, val_set = random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(
                train_set, batch_size=self.batch_size, shuffle=True
            )
            val_loader = DataLoader(
                val_set, batch_size=self.batch_size, shuffle=False
            )

            num_classes = len(self.char_classes)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = DigitCNN(num_classes=num_classes).to(device)
            criterion = nn.CrossEntropyLoss()
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
