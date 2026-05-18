from __future__ import annotations

import importlib

import torch
import torch.nn as nn


MODEL_CHOICES = ("tiny-cnn", "resnet50", "vit-tiny")


class TinyCNN(nn.Module):
    def __init__(self, channels: int = 3, classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(96 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _adapt_first_conv(model: nn.Module, channels: int) -> None:
    if channels == 3:
        return
    conv = model.conv1
    model.conv1 = nn.Conv2d(
        channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )


def build_model(name: str, channels: int, classes: int, image_size: int) -> nn.Module:
    if name == "tiny-cnn":
        return TinyCNN(channels=channels, classes=classes)

    if name == "resnet50":
        torchvision = importlib.import_module("torchvision")
        model = torchvision.models.resnet50(weights=None, num_classes=classes)
        _adapt_first_conv(model, channels)
        return model

    if name == "vit-tiny":
        try:
            timm = importlib.import_module("timm")
        except ImportError as exc:
            raise RuntimeError(
                "vit-tiny requires timm. Install benchmark dev dependencies with `uv add timm --dev`."
            ) from exc
        return timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            num_classes=classes,
            img_size=image_size,
            in_chans=channels,
        )

    raise ValueError(f"Unknown model: {name}. Choose one of {MODEL_CHOICES}.")
