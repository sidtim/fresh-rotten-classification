import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
import pytorch_lightning as pl
from typing import Any, Dict, Optional

from .model import create_model


class FreshRottenClassifier(pl.LightningModule):
    """LightningModule для классификации свежих/гнилых продуктов."""

    def __init__(
        self,
        model_type: str = "resnet18",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        num_classes: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Создаем модель
        self.model = create_model(
            model_type=model_type,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, y)

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": y}

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, y)

        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": y}

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.test_accuracy(preds, y)

        # Log metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": y}

    def configure_optimizers(self) -> Dict[str, Any]:
        """Настройка оптимизатора и scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # Используем StepLR вместо ReduceLROnPlateau чтобы избежать проблем с verbose
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,  # Уменьшать LR каждые 10 эпох
            gamma=0.5,  # Уменьшать LR в 2 раза
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }

    def on_train_epoch_end(self) -> None:
        """Логирование learning rate в конце эпохи."""
        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]["lr"]
        self.log("learning_rate", current_lr, on_epoch=True, prog_bar=True)
