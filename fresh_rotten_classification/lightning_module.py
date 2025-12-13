from typing import Any

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch import nn
from torchmetrics import Accuracy

from .model import create_model


class FreshRottenClassifier(pl.LightningModule):
    """LightningModule for fresh/rotten classification."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        # self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        # self.cfg = cfg  # Оригинал для использования

        # Create model
        self.model = create_model(
            model_type=cfg.model.type,
            num_classes=cfg.model.num_classes,
            pretrained=cfg.model.pretrained,
            freeze_backbone=cfg.model.freeze_backbone,
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=cfg.model.num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=cfg.model.num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=cfg.model.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> dict[str, torch.Tensor]:
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

    def validation_step(self, batch: Any, batch_idx: int) -> dict[str, torch.Tensor]:
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

    def test_step(self, batch: Any, batch_idx: int) -> dict[str, torch.Tensor]:
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

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )

        # Use StepLR instead of ReduceLROnPlateau to avoid verbose issues
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_train_epoch_end(self) -> None:
        """Log learning rate at the end of epoch."""
        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]["lr"]
        self.log("learning_rate", current_lr, on_epoch=True, prog_bar=True)
