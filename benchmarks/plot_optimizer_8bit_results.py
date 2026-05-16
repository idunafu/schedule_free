from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot optimizer_8bit_comparison.py JSON/CSV results.")
    parser.add_argument("input", help="Path to a JSON or CSV result file produced by optimizer_8bit_comparison.py.")
    parser.add_argument("--output", default="", help="Output image path. Defaults to <input>.png.")
    parser.add_argument("--title", default="Schedule-Free optimizer comparison")
    parser.add_argument("--show", action="store_true", help="Show the plot interactively after saving.")
    return parser.parse_args()


def parse_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def load_results(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported input format: {path.suffix}. Use JSON or CSV.")


def values(records: List[Dict[str, Any]], key: str) -> List[Optional[float]]:
    return [parse_float(record.get(key)) for record in records]


def plot_bar(ax, labels: List[str], data: List[Optional[float]], title: str, ylabel: str) -> None:
    xs = list(range(len(labels)))
    present = [value is not None for value in data]
    heights = [0.0 if value is None else value for value in data]
    colors = ["#4C78A8" if ok else "#B8B8B8" for ok in present]

    ax.bar(xs, heights, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)

    for x, value, ok in zip(xs, data, present):
        if ok and value is not None:
            ax.text(x, value, f"{value:.3g}", ha="center", va="bottom", fontsize=8)
        else:
            ax.text(x, 0, "skipped", ha="center", va="bottom", fontsize=8, rotation=90)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".png")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting. Install it with `pip install matplotlib`.") from exc

    records = load_results(input_path)
    if not records:
        raise SystemExit("No benchmark records found.")

    labels = [str(record.get("optimizer", "")) for record in records]
    statuses = [str(record.get("status", "")) for record in records]
    labels = [f"{label}\n({status})" if status and status != "ok" else label for label, status in zip(labels, statuses)]

    samples_per_second = values(records, "samples_per_second")
    optimizer_state_mb = values(records, "optimizer_state_mb")
    cuda_peak_allocated_mb = values(records, "cuda_peak_allocated_mb")
    eval_loss = values(records, "eval_loss")
    eval_accuracy = values(records, "eval_accuracy")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(args.title)

    plot_bar(axes[0][0], labels, samples_per_second, "Throughput", "samples/s")
    plot_bar(axes[0][1], labels, optimizer_state_mb, "Optimizer State", "MiB")
    plot_bar(axes[0][2], labels, cuda_peak_allocated_mb, "CUDA Peak Allocated", "MiB")
    plot_bar(axes[1][0], labels, eval_loss, "Eval Loss", "loss")
    plot_bar(
        axes[1][1],
        labels,
        [None if value is None else value * 100.0 for value in eval_accuracy],
        "Eval Accuracy",
        "%",
    )

    axes[1][2].axis("off")
    error_lines = [
        f"{record.get('optimizer')}: {record.get('error')}"
        for record in records
        if record.get("error")
    ]
    if error_lines:
        axes[1][2].text(0, 1, "\n".join(error_lines), va="top", fontsize=9, wrap=True)
    else:
        axes[1][2].text(0, 1, "No skipped or failed benchmark records.", va="top", fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    print(f"Wrote {output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
