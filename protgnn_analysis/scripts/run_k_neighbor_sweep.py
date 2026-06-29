#!/usr/bin/env python3
"""
Run the fixed prototype-loss k-neighbor comparison.

Each run uses:
  - prototype learning enabled
  - clst=0.02
  - sep=0.0
  - explain_n=0, so the sweep measures model performance only

The script archives each training run and writes a compact comparison table to:
  outputs/k_neighbor_sweep/k_neighbor_results.csv
  outputs/k_neighbor_sweep/k_neighbor_results.txt
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "k_neighbor_sweep"
RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"
SCRIPT = PROJECT_ROOT / "scripts" / "train_and_explain.py"
K_VALUES = [5, 10, 15, 20]


def latest_run_for_dataset(dataset: str, started_at: float) -> Path:
    candidates = []
    for path in RUNS_DIR.glob(f"*_{dataset}_gcn_prototype_clst0.02_sep0_k-neighbor-sweep"):
        if path.is_dir() and path.stat().st_mtime >= started_at:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No archived run found for {dataset}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_one(k: int) -> dict:
    dataset = f"mimic_patient_sim_k{k}"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'external' / 'GraphXAI-main'}:{PROJECT_ROOT}"
    command = [
        sys.executable,
        str(SCRIPT),
        "--dataset",
        dataset,
        "--clst",
        "0.02",
        "--sep",
        "0.0",
        "--explain_n",
        "0",
        "--archive_tag",
        "k-neighbor-sweep",
    ]

    started_at = datetime.now().timestamp()
    print(f"\n=== k={k} | {dataset} ===", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)

    run_dir = latest_run_for_dataset(dataset, started_at)
    metrics = load_json(run_dir / "test_metrics.json")
    config = load_json(run_dir / "model_config.json")

    return {
        "k": k,
        "dataset": dataset,
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        "threshold": metrics.get("threshold", config.get("decision_threshold")),
        "pr_auc": metrics.get("pr_auc"),
        "acc": metrics.get("acc"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "specificity": metrics.get("specificity"),
        "f1": metrics.get("f1"),
        "balanced_acc": metrics.get("balanced_acc"),
        "tn": metrics.get("confusion_matrix", {}).get("tn"),
        "fp": metrics.get("confusion_matrix", {}).get("fp"),
        "fn": metrics.get("confusion_matrix", {}).get("fn"),
        "tp": metrics.get("confusion_matrix", {}).get("tp"),
    }


def write_csv(rows: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "k",
        "dataset",
        "pr_auc",
        "acc",
        "precision",
        "recall",
        "specificity",
        "f1",
        "balanced_acc",
        "threshold",
        "tn",
        "fp",
        "fn",
        "tp",
        "run_dir",
    ]
    with (OUTPUT_DIR / "k_neighbor_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_report(rows: list[dict]) -> None:
    headers = ["k", "PR-AUC", "F1", "Recall", "Specificity", "Balanced Acc", "Accuracy", "Threshold", "Run"]
    lines = [
        "# k-neighbor sweep",
        "",
        "Fixed configuration: prototype GCN, clst=0.02, sep=0.0, LOS included, explain_n=0.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in sorted(rows, key=lambda item: item["k"]):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["k"]),
                    fmt(row["pr_auc"]),
                    fmt(row["f1"]),
                    fmt(row["recall"]),
                    fmt(row["specificity"]),
                    fmt(row["balanced_acc"]),
                    fmt(row["acc"]),
                    fmt(row["threshold"]),
                    row["run_dir"],
                ]
            )
            + " |"
        )

    best = max(rows, key=lambda item: float(item["pr_auc"]))
    lines.extend(
        [
            "",
            f"Best by PR-AUC: k={best['k']} ({best['pr_auc']:.6f}).",
            "",
        ]
    )
    (OUTPUT_DIR / "k_neighbor_results.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = [run_one(k) for k in K_VALUES]
    rows = sorted(rows, key=lambda item: item["k"])
    write_csv(rows)
    write_report(rows)
    print(f"\nWrote {OUTPUT_DIR / 'k_neighbor_results.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'k_neighbor_results.txt'}")


if __name__ == "__main__":
    main()
