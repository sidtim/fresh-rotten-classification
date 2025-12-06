from typing import Literal

import torch
from torch import nn
from torchvision import models


class SimpleCNN(nn.Module):
    """Простая CNN модель с нуля."""

    def __init__(self, num_classes: int = 2):
        super().__init__()

        # Feature extractor
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2)

        # Classifier
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 14 * 14, 512)  # Для 224x224 входного изображения
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Classification
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class ResNetClassifier(nn.Module):
    """Классификатор на основе предобученного ResNet."""

    def __init__(
        self,
        model_name: str = "resnet18",
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # Загружаем предобученную модель
        if model_name == "resnet18":
            self.backbone = models.resnet18(weights="DEFAULT" if pretrained else None)
        elif model_name == "resnet34":
            self.backbone = models.resnet34(weights="DEFAULT" if pretrained else None)
        elif model_name == "resnet50":
            self.backbone = models.resnet50(weights="DEFAULT" if pretrained else None)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Замораживаем веса backbone если нужно
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Заменяем последний полносвязный слой
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def create_model(
    model_type: Literal["simple_cnn", "resnet18", "resnet34", "resnet50"] = "resnet18",
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Фабрика для создания модели."""
    if model_type == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)

    if model_type in ["resnet18", "resnet34", "resnet50"]:
        return ResNetClassifier(
            model_name=model_type,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    raise ValueError(f"Unknown model type: {model_type}")
