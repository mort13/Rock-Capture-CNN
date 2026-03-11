"""
Training progress dialog for Rock Capture CNN.
Modal dialog shown during CNN training with progress bar and epoch log.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QProgressBar, QPushButton, QTextEdit,
)

from cnn.trainer import TrainerThread


class TrainingDialog(QDialog):
    """Modal dialog that shows CNN training progress."""

    def __init__(self, trainer: TrainerThread, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Training CNN")
        self.setMinimumSize(450, 350)
        self.setModal(True)
        self._trainer = trainer

        layout = QVBoxLayout(self)

        self.status_label = QLabel("Initializing...")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, trainer.num_epochs)
        layout.addWidget(self.progress_bar)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.clicked.connect(self._on_stop)
        layout.addWidget(self.stop_btn)

        trainer.epoch_completed.connect(self._on_epoch)
        trainer.training_finished.connect(self._on_finished)
        trainer.training_failed.connect(self._on_failed)
        trainer.progress_update.connect(self._on_status)

        trainer.start()

    def _on_epoch(self, epoch: int, loss: float, acc: float) -> None:
        self.progress_bar.setValue(epoch)
        self.log_text.append(
            f"Epoch {epoch}: loss={loss:.4f}, val_acc={acc:.1%}"
        )
        self.status_label.setText(
            f"Epoch {epoch} - Loss: {loss:.4f} - Accuracy: {acc:.1%}"
        )

    def _on_finished(self, model_path: str) -> None:
        self.status_label.setText(f"Training complete! Saved: {model_path}")
        self.stop_btn.setText("Close")
        self.stop_btn.clicked.disconnect()
        self.stop_btn.clicked.connect(self.accept)

    def _on_failed(self, error: str) -> None:
        self.status_label.setText(f"Training failed: {error}")
        self.log_text.append(f"\nERROR: {error}")
        self.stop_btn.setText("Close")
        self.stop_btn.clicked.disconnect()
        self.stop_btn.clicked.connect(self.reject)

    def _on_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _on_stop(self) -> None:
        self._trainer.request_stop()
        self.status_label.setText("Stopping...")
