import argparse
import os
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from .lightning_module import FreshRottenClassifier
from .dataset import FreshRottenDataModule


def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(description="Train Fresh/Rotten classifier")

    # Data arguments
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to data directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.15,
        help="Fraction of training data to use for validation",
    )
    parser.add_argument(
        "--use_augmentation", action="store_true", help="Use data augmentation during training"
    )
    parser.add_argument(
        "--image_size", type=int, nargs=2, default=[224, 224], help="Image size (height, width)"
    )

    # Model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        default="resnet18",
        choices=["simple_cnn", "resnet18", "resnet34", "resnet50"],
        help="Type of model to use",
    )
    parser.add_argument(
        "--pretrained", action="store_true", help="Use pretrained weights (for ResNet models)"
    )
    parser.add_argument(
        "--freeze_backbone", action="store_true", help="Freeze backbone weights (for ResNet models)"
    )
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")

    # Training arguments
    parser.add_argument(
        "--max_epochs", type=int, default=50, help="Maximum number of epochs to train"
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training if available")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to save logs")

    return parser.parse_args()


def main():
    """Основная функция для обучения модели."""
    args = parse_args()

    # Настройка precision для CUDA для лучшей производительности
    if args.gpu and torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    # Создаем директории если их нет
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Инициализируем DataModule
    data_module = FreshRottenDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        use_augmentation=args.use_augmentation,
        image_size=tuple(args.image_size),
    )

    # Инициализируем модель
    model = FreshRottenClassifier(
        model_type=args.model_type,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        num_classes=2,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="{epoch:02d}-{val_acc:.2f}",
        monitor="val_acc",
        mode="max",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=10, mode="min", verbose=True
    )

    progress_bar_callback = RichProgressBar()

    # Logger
    logger = TensorBoardLogger(args.log_dir, name="fresh_rotten_classification")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if args.gpu and torch.cuda.is_available() else "cpu",
        devices=1 if args.gpu and torch.cuda.is_available() else None,
        callbacks=[checkpoint_callback, early_stopping_callback, progress_bar_callback],
        logger=logger,
        log_every_n_steps=10,
        enable_model_summary=True,
        deterministic=True,
        enable_progress_bar=True,
    )

    # Обучение
    print(f"Starting training with model: {args.model_type}")
    print(f"Using GPU: {args.gpu and torch.cuda.is_available()}")
    print(f"Use augmentation: {args.use_augmentation}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")

    trainer.fit(model, datamodule=data_module)

    # Тестирование
    print("\n" + "=" * 50)
    print("Testing the model...")
    trainer.test(model, datamodule=data_module, ckpt_path="best")

    print(f"\nTraining completed!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
