# train.py
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from .dataset import FreshRottenDataModule
from .lightning_module import FreshRottenClassifier

# Автоматически определяем путь к конфигам
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  # поднимаемся на уровень выше
configs_dir = project_root / "configs"


@hydra.main(
    version_base=None,
    config_path=str(configs_dir),
    config_name="main",
)
def main(cfg: DictConfig):
    """Main training function."""
    # Set seed for reproducibility
    pl.seed_everything(cfg.seed)

    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("\n" + "=" * 50 + "\n")

    # Setup CUDA precision if using GPU
    if cfg.training.gpu and torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    # Initialize DataModule
    data_module = FreshRottenDataModule(cfg)

    # Initialize model
    model = FreshRottenClassifier(cfg)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.training.checkpoint_dir,
        filename="{epoch:02d}-{val_acc:.2f}",
        monitor="val_acc",
        mode="max",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=cfg.training.early_stopping_patience,
        mode="min",
        verbose=True,
    )

    progress_bar_callback = RichProgressBar()

    # Logger
    logger = TensorBoardLogger(cfg.training.log_dir, name=cfg.name)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="gpu" if cfg.training.gpu and torch.cuda.is_available() else "cpu",
        devices=1 if cfg.training.gpu and torch.cuda.is_available() else None,
        callbacks=[checkpoint_callback, early_stopping_callback, progress_bar_callback],
        logger=logger,
        log_every_n_steps=10,
        enable_model_summary=True,
        deterministic=True,
        enable_progress_bar=True,
    )

    # Training
    print(f"Starting training with model: {cfg.model.type}")
    print(f"Using GPU: {cfg.training.gpu and torch.cuda.is_available()}")
    print(f"Use augmentation: {cfg.data.use_augmentation}")
    print(f"Batch size: {cfg.data.batch_size}")
    print(f"Learning rate: {cfg.training.learning_rate}")

    trainer.fit(model, datamodule=data_module)

    # Testing
    print("\n" + "=" * 50)
    print("Testing the model...")
    trainer.test(model, datamodule=data_module, ckpt_path="best")

    print("\nTraining completed!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
