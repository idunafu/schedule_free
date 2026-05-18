from __future__ import annotations

import importlib
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset


class PrototypeImageDataset(Dataset):
    def __init__(
        self,
        size: int,
        channels: int = 3,
        height: int = 32,
        width: int = 32,
        classes: int = 10,
        noise: float = 0.20,
        seed: int = 1,
        prototypes: Optional[torch.Tensor] = None,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        labels = torch.arange(size) % classes
        labels = labels[torch.randperm(size, generator=generator)]

        if prototypes is None:
            prototypes = make_prototypes(classes, channels, height, width, seed)

        self.prototypes = prototypes
        self.labels = labels.long()
        self.noise = noise
        self.seed = seed

    def __len__(self) -> int:
        return self.labels.numel()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        label = self.labels[index]
        image = self.prototypes[label].clone()
        if self.noise != 0.0:
            generator = torch.Generator().manual_seed(self.seed + int(index))
            image.add_(torch.randn(image.shape, generator=generator, dtype=image.dtype), alpha=self.noise)
        return image, label


def make_prototypes(classes: int, channels: int, height: int, width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    prototypes = torch.randn(classes, channels, height, width, generator=generator)
    smooth = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    return smooth(prototypes).mul_(0.7)


def import_torchvision():
    try:
        return importlib.import_module("torchvision")
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required for --dataset mnist/cifar10. "
            "Use synthetic-cifar10 or install torchvision."
        ) from exc


def build_datasets(args) -> Tuple[Dataset, Dataset, int, int]:
    image_size = args.image_size
    if args.dataset == "synthetic-cifar10":
        prototypes = make_prototypes(classes=10, channels=3, height=image_size, width=image_size, seed=args.seed)
        train = PrototypeImageDataset(
            args.train_size,
            channels=3,
            height=image_size,
            width=image_size,
            classes=10,
            seed=args.seed + 1,
            prototypes=prototypes,
        )
        eval_set = PrototypeImageDataset(
            args.eval_size,
            channels=3,
            height=image_size,
            width=image_size,
            classes=10,
            seed=args.seed + 2,
            prototypes=prototypes,
        )
        return train, eval_set, 3, 10

    torchvision = import_torchvision()
    datasets = torchvision.datasets
    transforms = torchvision.transforms

    if args.dataset == "mnist":
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_full = datasets.MNIST(args.data_root, train=True, download=args.download, transform=transform)
        eval_full = datasets.MNIST(args.data_root, train=False, download=args.download, transform=transform)
        channels = 1
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train_full = datasets.CIFAR10(args.data_root, train=True, download=args.download, transform=transform)
        eval_full = datasets.CIFAR10(args.data_root, train=False, download=args.download, transform=transform)
        channels = 3

    train = Subset(train_full, range(min(args.train_size, len(train_full))))
    eval_set = Subset(eval_full, range(min(args.eval_size, len(eval_full))))
    return train, eval_set, channels, 10


def make_loader(dataset: Dataset, args, shuffle: bool, seed: int, device: torch.device) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
