"""
v1/src/models.py
Model definitions for v1:
  1. MobileNetV2  – pretrained ImageNet backbone, fine-tuned classifier head
  2. CustomCNN    – built from scratch for road damage classification
"""

import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 5  # D00, D10, D20, D40, Normal


# ── 1. MobileNetV2 ─────────────────────────────────────────────────────────────
def build_mobilenetv2(num_classes: int = NUM_CLASSES, freeze_backbone: bool = True):
    """
    MobileNetV2 pretrained on ImageNet.
    Only the classifier head is replaced and trained by default.
    Set freeze_backbone=False to fine-tune the full network.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return model


# ── 2. Custom CNN ──────────────────────────────────────────────────────────────
class CustomCNN(nn.Module):
    """
    A compact CNN built from scratch.
    Input: (B, 3, 224, 224)
    Output: (B, num_classes)

    Architecture:
      Block 1: Conv 32  → BN → ReLU → MaxPool
      Block 2: Conv 64  → BN → ReLU → MaxPool
      Block 3: Conv 128 → BN → ReLU → MaxPool
      Block 4: Conv 256 → BN → ReLU → AdaptiveAvgPool
      Head   : FC 512 → Dropout → FC num_classes
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )

        self.features = nn.Sequential(
            conv_block(3,   32),   # → 112x112
            conv_block(32,  64),   # → 56x56
            conv_block(64,  128),  # → 28x28
            conv_block(128, 256),  # → 14x14
            nn.AdaptiveAvgPool2d((4, 4)),  # → 4x4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ── Model factory ──────────────────────────────────────────────────────────────
def get_model(name: str, num_classes: int = NUM_CLASSES):
    """
    Factory function.
    name: 'mobilenetv2' | 'customcnn'
    """
    name = name.lower()
    if name == "mobilenetv2":
        return build_mobilenetv2(num_classes)
    elif name == "customcnn":
        return CustomCNN(num_classes)
    else:
        raise ValueError(f"Unknown model '{name}'. Choose 'mobilenetv2' or 'customcnn'.")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
