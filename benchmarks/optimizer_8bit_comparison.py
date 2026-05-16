from __future__ import annotations

import argparse
import csv
import importlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import schedulefree


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
        images = prototypes[labels] + noise * torch.randn(size, channels, height, width, generator=generator)

        self.images = images
        self.labels = labels.long()

    def __len__(self) -> int:
        return self.labels.numel()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


def make_prototypes(classes: int, channels: int, height: int, width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    prototypes = torch.randn(classes, channels, height, width, generator=generator)
    smooth = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    return smooth(prototypes).mul_(0.7)


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


@dataclass
class BenchResult:
    optimizer: str
    status: str
    device: str
    steps: int
    epochs: int
    seconds: float
    samples_per_second: float
    final_train_loss: Optional[float]
    eval_loss: Optional[float]
    eval_accuracy: Optional[float]
    optimizer_state_mb: Optional[float]
    cuda_peak_allocated_mb: Optional[float]
    cuda_peak_reserved_mb: Optional[float]
    error: Optional[str] = None


class OptionalOptimizerUnavailable(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Schedule-Free AdamW, integrated 8-bit AdamW, and bitsandbytes AdamW8bit wrapped with ScheduleFreeWrapper."
    )
    parser.add_argument("--dataset", choices=["synthetic-cifar10", "mnist", "cifar10"], default="synthetic-cifar10")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--download", action="store_true", help="Download torchvision datasets when using mnist/cifar10.")
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--eval-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0, help="Stop each optimizer after this many training steps. 0 means full epochs.")
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--min-8bit-size", type=int, default=4096)
    parser.add_argument("--wrapper-beta1", type=float, default=0.0, help="Inner AdamW8bit beta1. 0 avoids double momentum with ScheduleFreeWrapper.")
    parser.add_argument("--optimizers", default="adamw,adamw8bit,wrapped-bnb-adamw8bit")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--fail-on-skip", action="store_true")
    return parser.parse_args()


def get_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if name == "mps" and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(name)


def import_torchvision():
    try:
        return importlib.import_module("torchvision")
    except ImportError as exc:
        raise RuntimeError("torchvision is required for --dataset mnist/cifar10. Use synthetic-cifar10 or install torchvision.") from exc


def build_datasets(args: argparse.Namespace) -> Tuple[Dataset, Dataset, int, int]:
    if args.dataset == "synthetic-cifar10":
        prototypes = make_prototypes(classes=10, channels=3, height=32, width=32, seed=args.seed)
        train = PrototypeImageDataset(
            args.train_size, channels=3, height=32, width=32, classes=10,
            seed=args.seed + 1, prototypes=prototypes)
        eval_set = PrototypeImageDataset(
            args.eval_size, channels=3, height=32, width=32, classes=10,
            seed=args.seed + 2, prototypes=prototypes)
        return train, eval_set, 3, 10

    torchvision = import_torchvision()
    datasets = torchvision.datasets
    transforms = torchvision.transforms

    if args.dataset == "mnist":
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_full = datasets.MNIST(args.data_root, train=True, download=args.download, transform=transform)
        eval_full = datasets.MNIST(args.data_root, train=False, download=args.download, transform=transform)
        channels = 1
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train_full = datasets.CIFAR10(args.data_root, train=True, download=args.download, transform=transform)
        eval_full = datasets.CIFAR10(args.data_root, train=False, download=args.download, transform=transform)
        channels = 3

    train = Subset(train_full, range(min(args.train_size, len(train_full))))
    eval_set = Subset(eval_full, range(min(args.eval_size, len(eval_full))))
    return train, eval_set, channels, 10


def make_loader(dataset: Dataset, args: argparse.Namespace, shuffle: bool, seed: int, device: torch.device) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_bytes(obj) -> int:
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(v) for v in obj)
    return 0


def optimizer_state_mb(optimizer) -> float:
    return tensor_bytes(optimizer.state) / (1024 ** 2)


def make_optimizer(name: str, params: Iterable[torch.Tensor], args: argparse.Namespace, device: torch.device):
    betas = (args.beta1, args.beta2)
    if name == "adamw":
        return schedulefree.AdamWScheduleFree(
            params,
            lr=args.lr,
            betas=betas,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            foreach=False,
        )
    if name == "adamw8bit":
        return schedulefree.AdamWScheduleFree8bit(
            params,
            lr=args.lr,
            betas=betas,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            block_size=args.block_size,
            min_8bit_size=args.min_8bit_size,
        )
    if name == "wrapped-bnb-adamw8bit":
        if device.type != "cuda":
            raise OptionalOptimizerUnavailable("bitsandbytes AdamW8bit comparison requires a CUDA device.")
        try:
            bnb = importlib.import_module("bitsandbytes")
        except ImportError as exc:
            raise OptionalOptimizerUnavailable("bitsandbytes is not installed; skipping wrapped-bnb-adamw8bit.") from exc
        base = bnb.optim.AdamW8bit(
            params,
            lr=args.lr,
            betas=(args.wrapper_beta1, args.beta2),
            weight_decay=0.0,
        )
        return schedulefree.ScheduleFreeWrapper(
            base,
            momentum=args.beta1,
            weight_decay_at_y=args.weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {name}")


def train_one(
    optimizer_name: str,
    initial_state: Dict[str, torch.Tensor],
    train_set: Dataset,
    eval_set: Dataset,
    channels: int,
    classes: int,
    args: argparse.Namespace,
    device: torch.device,
) -> BenchResult:
    torch.manual_seed(args.seed)
    model = TinyCNN(channels=channels, classes=classes).to(device)
    model.load_state_dict(initial_state)
    optimizer = make_optimizer(optimizer_name, model.parameters(), args, device)

    train_loader = make_loader(train_set, args, shuffle=True, seed=args.seed, device=device)
    eval_loader = make_loader(eval_set, args, shuffle=False, seed=args.seed + 1, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    total_loss = 0.0
    total_samples = 0
    steps = 0
    sync(device)
    start = time.perf_counter()

    for _epoch in range(args.epochs):
        model.train()
        optimizer.train()
        for data, target in train_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(data)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()

            batch = target.numel()
            total_loss += loss.detach().item() * batch
            total_samples += batch
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break
        if args.max_steps and steps >= args.max_steps:
            break

    sync(device)
    seconds = time.perf_counter() - start

    model.eval()
    optimizer.eval()
    eval_loss = 0.0
    correct = 0
    eval_samples = 0
    with torch.no_grad():
        for data, target in eval_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model(data)
            eval_loss += F.cross_entropy(logits, target, reduction="sum").item()
            correct += logits.argmax(dim=1).eq(target).sum().item()
            eval_samples += target.numel()

    peak_allocated = None
    peak_reserved = None
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    return BenchResult(
        optimizer=optimizer_name,
        status="ok",
        device=str(device),
        steps=steps,
        epochs=args.epochs,
        seconds=seconds,
        samples_per_second=total_samples / seconds if seconds > 0 else 0.0,
        final_train_loss=total_loss / total_samples if total_samples else None,
        eval_loss=eval_loss / eval_samples if eval_samples else None,
        eval_accuracy=correct / eval_samples if eval_samples else None,
        optimizer_state_mb=optimizer_state_mb(optimizer),
        cuda_peak_allocated_mb=peak_allocated,
        cuda_peak_reserved_mb=peak_reserved,
    )


def skipped_result(name: str, device: torch.device, error: str) -> BenchResult:
    return BenchResult(
        optimizer=name,
        status="skipped",
        device=str(device),
        steps=0,
        epochs=0,
        seconds=0.0,
        samples_per_second=0.0,
        final_train_loss=None,
        eval_loss=None,
        eval_accuracy=None,
        optimizer_state_mb=None,
        cuda_peak_allocated_mb=None,
        cuda_peak_reserved_mb=None,
        error=error,
    )


def print_results(results: List[BenchResult]) -> None:
    headers = [
        "optimizer",
        "status",
        "steps",
        "sec",
        "samples/s",
        "train loss",
        "eval loss",
        "eval acc",
        "opt state MB",
        "cuda peak MB",
    ]
    rows = []
    for result in results:
        rows.append([
            result.optimizer,
            result.status,
            str(result.steps),
            f"{result.seconds:.3f}",
            f"{result.samples_per_second:.1f}",
            "" if result.final_train_loss is None else f"{result.final_train_loss:.4f}",
            "" if result.eval_loss is None else f"{result.eval_loss:.4f}",
            "" if result.eval_accuracy is None else f"{100 * result.eval_accuracy:.2f}%",
            "" if result.optimizer_state_mb is None else f"{result.optimizer_state_mb:.2f}",
            "" if result.cuda_peak_allocated_mb is None else f"{result.cuda_peak_allocated_mb:.2f}",
        ])

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    print(" | ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("-|-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(cell.ljust(width) for cell, width in zip(row, widths)))

    for result in results:
        if result.error:
            print(f"\n{result.optimizer}: {result.error}")


def write_outputs(results: List[BenchResult], args: argparse.Namespace) -> None:
    records = [asdict(result) for result in results]
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    if args.output_csv:
        path = Path(args.output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    train_set, eval_set, channels, classes = build_datasets(args)

    torch.manual_seed(args.seed)
    initial_model = TinyCNN(channels=channels, classes=classes)
    initial_state = {key: value.detach().clone() for key, value in initial_model.state_dict().items()}

    requested = [name.strip() for name in args.optimizers.split(",") if name.strip()]
    results: List[BenchResult] = []
    for name in requested:
        try:
            results.append(train_one(name, initial_state, train_set, eval_set, channels, classes, args, device))
        except OptionalOptimizerUnavailable as exc:
            if args.fail_on_skip or name != "wrapped-bnb-adamw8bit":
                raise
            results.append(skipped_result(name, device, str(exc)))

    print_results(results)
    write_outputs(results, args)


if __name__ == "__main__":
    main()
