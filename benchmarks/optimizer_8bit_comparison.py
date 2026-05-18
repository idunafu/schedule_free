from __future__ import annotations

import argparse
from typing import List

import torch

from optimizer_bench.datasets import build_datasets
from optimizer_bench.models import MODEL_CHOICES, build_model
from optimizer_bench.optimizers import OptionalOptimizerUnavailable
from optimizer_bench.reporting import print_results, write_outputs
from optimizer_bench.runner import BenchResult, get_device, skipped_result, train_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Schedule-Free optimizers, integrated 8-bit variants, and "
            "bitsandbytes AdamW8bit wrapped with Schedule-Free wrapper logic."
        )
    )
    parser.add_argument("--dataset", choices=["synthetic-cifar10", "mnist", "cifar10"], default="synthetic-cifar10")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--download", action="store_true", help="Download torchvision datasets when using mnist/cifar10.")
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--eval-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", "--epoch", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0, help="Stop each optimizer after this many training steps. 0 means full epochs.")
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--min-8bit-size", type=int, default=4096)
    parser.add_argument("--quant-backend", choices=["bnb_dynamic", "torch_linear"], default="bnb_dynamic")
    parser.add_argument("--wrapper-beta1", type=float, default=0.0, help="Inner AdamW8bit beta1. 0 avoids double momentum with ScheduleFreeWrapper.")
    parser.add_argument("--optimizers", default="adamw,adamw8bit,wrapped-bnb-adamw8bit")
    parser.add_argument("--models", default="tiny-cnn", help=f"Comma-separated model list. Choices: {', '.join(MODEL_CHOICES)}.")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--eval-interval", type=int, default=0, help="Evaluate every N optimizer steps. 0 evaluates once per epoch.")
    parser.add_argument("--no-history", action="store_true", help="Do not store per-epoch/per-interval convergence history.")
    parser.add_argument("--fail-on-skip", action="store_true")
    return parser.parse_args()


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def validate_models(model_names: List[str]) -> None:
    invalid = [name for name in model_names if name not in MODEL_CHOICES]
    if invalid:
        raise ValueError(f"Unknown model(s): {', '.join(invalid)}. Choose from {', '.join(MODEL_CHOICES)}.")


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    train_set, eval_set, channels, classes = build_datasets(args)

    requested_models = parse_csv(args.models)
    validate_models(requested_models)
    requested_optimizers = parse_csv(args.optimizers)

    results: List[BenchResult] = []
    for model_name in requested_models:
        torch.manual_seed(args.seed)
        initial_model = build_model(model_name, channels=channels, classes=classes, image_size=args.image_size)
        initial_state = {key: value.detach().clone() for key, value in initial_model.state_dict().items()}

        for optimizer_name in requested_optimizers:
            try:
                results.append(
                    train_one(
                        model_name,
                        optimizer_name,
                        initial_state,
                        train_set,
                        eval_set,
                        channels,
                        classes,
                        args,
                        device,
                    )
                )
            except OptionalOptimizerUnavailable as exc:
                if args.fail_on_skip or optimizer_name != "wrapped-bnb-adamw8bit":
                    raise
                results.append(skipped_result(model_name, optimizer_name, device, args, str(exc)))

    print_results(results)
    write_outputs(results, args)


if __name__ == "__main__":
    main()
