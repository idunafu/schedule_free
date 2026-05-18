from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .datasets import make_loader
from .models import build_model
from .optimizers import make_optimizer


@dataclass
class BenchResult:
    model: str
    dataset: str
    image_size: int
    optimizer: str
    quant_backend: str
    status: str
    device: str
    steps: int
    epochs: int
    seconds: float
    train_seconds: float
    eval_seconds: float
    samples_per_second: float
    train_samples_per_second: float
    final_train_loss: Optional[float]
    eval_loss: Optional[float]
    eval_accuracy: Optional[float]
    optimizer_state_mb: Optional[float]
    cuda_peak_allocated_mb: Optional[float]
    cuda_peak_reserved_mb: Optional[float]
    error: Optional[str] = None
    history: Optional[List[Dict[str, float]]] = None


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


def evaluate(model: nn.Module, optimizer, eval_loader, device: torch.device) -> tuple[float, float]:
    was_training = model.training
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

    if was_training:
        model.train()
        optimizer.train()

    return eval_loss / eval_samples, correct / eval_samples


def train_one(
    model_name: str,
    optimizer_name: str,
    initial_state: Dict[str, torch.Tensor],
    train_set: Dataset,
    eval_set: Dataset,
    channels: int,
    classes: int,
    args,
    device: torch.device,
) -> BenchResult:
    torch.manual_seed(args.seed)
    model = build_model(model_name, channels=channels, classes=classes, image_size=args.image_size).to(device)
    model.load_state_dict(initial_state)
    optimizer = make_optimizer(optimizer_name, model.parameters(), args, device)

    train_loader = make_loader(train_set, args, shuffle=True, seed=args.seed, device=device)
    eval_loader = make_loader(eval_set, args, shuffle=False, seed=args.seed + 1, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    total_loss = 0.0
    total_samples = 0
    interval_loss = 0.0
    interval_samples = 0
    steps = 0
    train_seconds = 0.0
    eval_seconds = 0.0
    history: List[Dict[str, float]] = []
    sync(device)
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.train()
        for data, target in train_loader:
            sync(device)
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            sync(device)
            train_start = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)
            logits = model(data)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            sync(device)
            train_seconds += time.perf_counter() - train_start

            batch = target.numel()
            loss_value = loss.detach().item()
            total_loss += loss_value * batch
            total_samples += batch
            interval_loss += loss_value * batch
            interval_samples += batch
            steps += 1

            if not args.no_history and args.eval_interval and steps % args.eval_interval == 0:
                sync(device)
                eval_start = time.perf_counter()
                eval_loss, eval_accuracy = evaluate(model, optimizer, eval_loader, device)
                sync(device)
                eval_seconds += time.perf_counter() - eval_start
                history.append({
                    "epoch": float(epoch),
                    "step": float(steps),
                    "seconds": time.perf_counter() - start,
                    "train_seconds": train_seconds,
                    "eval_seconds": eval_seconds,
                    "train_loss": interval_loss / interval_samples,
                    "eval_loss": eval_loss,
                    "eval_accuracy": eval_accuracy,
                })
                interval_loss = 0.0
                interval_samples = 0

            if args.max_steps and steps >= args.max_steps:
                break

        if not args.no_history and not args.eval_interval and interval_samples > 0:
            sync(device)
            eval_start = time.perf_counter()
            eval_loss, eval_accuracy = evaluate(model, optimizer, eval_loader, device)
            sync(device)
            eval_seconds += time.perf_counter() - eval_start
            history.append({
                "epoch": float(epoch),
                "step": float(steps),
                "seconds": time.perf_counter() - start,
                "train_seconds": train_seconds,
                "eval_seconds": eval_seconds,
                "train_loss": interval_loss / interval_samples,
                "eval_loss": eval_loss,
                "eval_accuracy": eval_accuracy,
            })
            interval_loss = 0.0
            interval_samples = 0

        if args.max_steps and steps >= args.max_steps:
            break

    sync(device)
    eval_start = time.perf_counter()
    eval_loss, eval_accuracy = evaluate(model, optimizer, eval_loader, device)
    sync(device)
    eval_seconds += time.perf_counter() - eval_start
    seconds = time.perf_counter() - start

    peak_allocated = None
    peak_reserved = None
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    return BenchResult(
        model=model_name,
        dataset=args.dataset,
        image_size=args.image_size,
        optimizer=optimizer_name,
        quant_backend=args.quant_backend,
        status="ok",
        device=str(device),
        steps=steps,
        epochs=args.epochs,
        seconds=seconds,
        train_seconds=train_seconds,
        eval_seconds=eval_seconds,
        samples_per_second=total_samples / seconds if seconds > 0 else 0.0,
        train_samples_per_second=total_samples / train_seconds if train_seconds > 0 else 0.0,
        final_train_loss=total_loss / total_samples if total_samples else None,
        eval_loss=eval_loss,
        eval_accuracy=eval_accuracy,
        optimizer_state_mb=optimizer_state_mb(optimizer),
        cuda_peak_allocated_mb=peak_allocated,
        cuda_peak_reserved_mb=peak_reserved,
        history=history if not args.no_history else None,
    )


def skipped_result(model_name: str, optimizer_name: str, device: torch.device, args, error: str) -> BenchResult:
    return BenchResult(
        model=model_name,
        dataset=args.dataset,
        image_size=args.image_size,
        optimizer=optimizer_name,
        quant_backend=args.quant_backend,
        status="skipped",
        device=str(device),
        steps=0,
        epochs=0,
        seconds=0.0,
        train_seconds=0.0,
        eval_seconds=0.0,
        samples_per_second=0.0,
        train_samples_per_second=0.0,
        final_train_loss=None,
        eval_loss=None,
        eval_accuracy=None,
        optimizer_state_mb=None,
        cuda_peak_allocated_mb=None,
        cuda_peak_reserved_mb=None,
        error=error,
    )
