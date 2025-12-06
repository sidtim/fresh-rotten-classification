import os
from typing import Optional, Tuple, Dict, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl


class FreshRottenDataset(Dataset):
    """Датасет для классификации свежих и гнилых овощей/фруктов."""

    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None):
        """
        Args:
            root_dir: Путь к папке с данными (должна содержать fresh_product/ и rotten_product/)
            transform: Трансформации для аугментации данных
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ["fresh_product", "rotten_product"]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Загрузка изображения
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class FreshRottenDataModule(pl.LightningDataModule):
    """Lightning DataModule для управления данными."""

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.15,
        use_augmentation: bool = True,
        image_size: Tuple[int, int] = (224, 224),
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.use_augmentation = use_augmentation
        self.image_size = image_size

        # Определяем трансформации
        self.train_transform = self._get_train_transform()
        self.val_test_transform = self._get_val_test_transform()

    def _get_train_transform(self) -> transforms.Compose:
        """Трансформации для тренировочных данных."""
        transform_list = [
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        if not self.use_augmentation:
            # Без аугментации - только базовые преобразования
            transform_list = [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]

        return transforms.Compose(transform_list)

    def _get_val_test_transform(self) -> transforms.Compose:
        """Трансформации для валидационных и тестовых данных."""
        return transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def setup(self, stage: Optional[str] = None):
        """Настройка датасетов для разных стадий."""
        # Полный тренировочный датасет
        full_train_dir = os.path.join(self.data_dir, "train")
        full_dataset = FreshRottenDataset(full_train_dir, transform=self.train_transform)

        # Разделяем на train и val
        val_size = int(len(full_dataset) * self.val_split)
        train_size = len(full_dataset) - val_size

        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

        # Тестовый датасет
        test_dir = os.path.join(self.data_dir, "test")
        self.test_dataset = FreshRottenDataset(test_dir, transform=self.val_test_transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
