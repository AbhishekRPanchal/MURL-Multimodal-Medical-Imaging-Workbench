"""
Ultrasound Classification Models
Implements:
1. Baseline custom CNN (UltrasoundBaselineCNN)
2. Transfer Learning Architecture (UltrasoundTransferClassifier) with ResNet18/DenseNet121
"""

import torch
import torch.nn as nn
import torchvision.models as models

class UltrasoundBaselineCNN(nn.Module):
    """
    Custom 4-stage convolutional baseline network designed specifically
    for 2D ultrasound texture and speckle pattern extraction.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 3, dropout_rate: float = 0.3):
        super().__init__()
        
        self.features = nn.Sequential(
            # Stage 1: Initial texture feature extraction
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 256 -> 128

            # Stage 2: Intermediate speckle abstraction
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128 -> 64

            # Stage 3: Lesion margin and acoustic shadow detection
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64 -> 32

            # Stage 4: High-level semantic representation
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        pooled = self.global_pool(feats)
        flattened = torch.flatten(pooled, 1)
        logits = self.classifier(flattened)
        return logits


class UltrasoundTransferClassifier(nn.Module):
    """
    Transfer learning classifier leveraging pretrained ResNet18 backbone
    with ultrasound-adapted fine-tuning head.
    """
    def __init__(self, architecture: str = "resnet18", num_classes: int = 3, pretrained: bool = True, dropout_rate: float = 0.3):
        super().__init__()
        self.architecture = architecture
        
        if architecture == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_features, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(128, num_classes)
            )
        elif architecture == "densenet121":
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            self.backbone = models.densenet121(weights=weights)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_features, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(128, num_classes)
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}. Choose 'resnet18' or 'densenet121'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
