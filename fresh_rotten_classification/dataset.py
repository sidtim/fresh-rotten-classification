from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


class FreshRottenDataset(Dataset):
    """Dataset for fresh/rotten vegetables and fruits classification."""

    def __init__(self, root_dir: str, transform: transforms.Compose | None = None):
        """
        Args:
            root_dir: Path to data directory
            transform: Transformations for data augmentation
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = ["fresh_product", "rotten_product"]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            for img_name in class_dir.iterdir():
                if img_name.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]:
                    img_path = class_dir / img_name.name
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class FreshRottenDataModule(pl.LightningDataModule):
    """Lightning DataModule for managing data."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        # Define transforms
        self.train_transform = self._get_train_transform()
        self.val_test_transform = self._get_val_test_transform()

    def _get_train_transform(self) -> transforms.Compose:
        """Transformations for training data."""
        if not self.cfg.data.use_augmentation:
            # Without augmentation - only basic transforms
            return transforms.Compose(
                [
                    transforms.Resize(self.cfg.data.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=self.cfg.data.augmentation.normalize.mean,
                        std=self.cfg.data.augmentation.normalize.std,
                    ),
                ]
            )

        # With augmentation
        return transforms.Compose(
            [
                transforms.Resize(self.cfg.data.image_size),
                transforms.RandomHorizontalFlip(p=self.cfg.data.augmentation.horizontal_flip_prob),
                transforms.RandomRotation(degrees=self.cfg.data.augmentation.rotation_degrees),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.cfg.data.augmentation.normalize.mean,
                    std=self.cfg.data.augmentation.normalize.std,
                ),
            ]
        )

    def _get_val_test_transform(self) -> transforms.Compose:
        """Transformations for validation and test data."""
        return transforms.Compose(
            [
                transforms.Resize(self.cfg.data.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.cfg.data.augmentation.normalize.mean,
                    std=self.cfg.data.augmentation.normalize.std,
                ),
            ]
        )

    def setup(self, stage: str | None = None):
        """Setup datasets for different stages."""
        # Full training dataset
        full_train_dir = Path(self.cfg.data.dir) / "train"
        full_dataset = FreshRottenDataset(full_train_dir, transform=self.train_transform)

        # Split into train and val
        val_size = int(len(full_dataset) * self.cfg.data.val_split)
        train_size = len(full_dataset) - val_size

        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

        # Test dataset
        test_dir = Path(self.cfg.data.dir) / "test"
        self.test_dataset = FreshRottenDataset(test_dir, transform=self.val_test_transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
        )
