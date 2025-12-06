from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar

from .dataset import FreshRottenDataModule
from .dvc_utils import check_and_download_data
from .lightning_module import FreshRottenClassifier

# from .dvc_utils import check_and_download_data
from .mlflow_logger import setup_mlflow

# # Разрешить DictConfig для безопасной загрузки чекпоинтов
# torch.serialization.add_safe_globals([DictConfig])

# logger = logging.getLogger(__name__)

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

    if not check_and_download_data(cfg.data.dir):
        logger.error("Can't download data from DVC")
        return

    # Настройка MLflow
    mlflow_logger, mlflow_callbacks = setup_mlflow(cfg)

    # Set seed for reproducibility
    pl.seed_everything(cfg.seed)

    # Print configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Setup CUDA precision if using GPU
    if cfg.training.gpu and torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    # Initialize DataModule
    data_module = FreshRottenDataModule(cfg)

    # Initialize model
    model = FreshRottenClassifier(cfg)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.training.checkpoint_dir,
            filename="{epoch:02d}-{val_acc:.2f}",
            monitor="val_acc",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=cfg.training.early_stopping_patience,
            mode="min",
        ),
        RichProgressBar(),
    ]

    # Добавляем MLflow callbacks если есть
    if mlflow_callbacks:
        callbacks.extend(mlflow_callbacks)

    # Настройка логгеров
    loggers = []

    # MLflow logger (для гиперпараметров и git commit)
    if mlflow_logger:
        loggers.append(mlflow_logger)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="gpu" if cfg.training.gpu and torch.cuda.is_available() else "cpu",
        devices=1 if cfg.training.gpu and torch.cuda.is_available() else None,
        callbacks=callbacks,
        logger=loggers if loggers else True,
        log_every_n_steps=10,
        enable_model_summary=True,
        enable_progress_bar=True,
    )

    # Training
    logger.info(f"Starting training with model: {cfg.model.type}")
    trainer.fit(model, datamodule=data_module)

    # Testing
    logger.info("Testing the model...")
    trainer.test(model, datamodule=data_module, ckpt_path="best")

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
