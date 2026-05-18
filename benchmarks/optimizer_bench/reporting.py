from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import List

from .runner import BenchResult


def print_results(results: List[BenchResult]) -> None:
    headers = [
        "model",
        "optimizer",
        "status",
        "steps",
        "sec",
        "train sec",
        "samples/s",
        "train samples/s",
        "train loss",
        "eval loss",
        "eval acc",
        "opt state MB",
        "cuda peak MB",
    ]
    rows = []
    for result in results:
        rows.append([
            result.model,
            result.optimizer,
            result.status,
            str(result.steps),
            f"{result.seconds:.3f}",
            f"{result.train_seconds:.3f}",
            f"{result.samples_per_second:.1f}",
            f"{result.train_samples_per_second:.1f}",
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
            print(f"\n{result.model}/{result.optimizer}: {result.error}")


def write_outputs(results: List[BenchResult], args) -> None:
    records = [asdict(result) for result in results]
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    if args.output_csv:
        path = Path(args.output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        csv_records = []
        for record in records:
            csv_record = dict(record)
            if csv_record.get("history") is not None:
                csv_record["history"] = json.dumps(csv_record["history"])
            csv_records.append(csv_record)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_records[0].keys()))
            writer.writeheader()
            writer.writerows(csv_records)
