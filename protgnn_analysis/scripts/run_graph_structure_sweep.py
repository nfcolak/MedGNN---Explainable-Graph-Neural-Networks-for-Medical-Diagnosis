#!/usr/bin/env python3
"""
Compare patient-similarity graph structures under fixed prototype settings.

Default comparison keeps k=5 because the k-neighbor sweep selected it as the
best PR-AUC setting.
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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "graph_structure_sweep"
RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"
SCRIPT = PROJECT_ROOT / "scripts" / "train_and_explain.py"

DATASETS = [
    ("star", "mimic_patient_sim_k5"),
    ("weighted_star", "mimic_patient_sim_weighted_k5"),
    ("local_knn", "mimic_patient_sim_local_k5"),
]


def latest_run_for_dataset(dataset: str, started_at: float) -> Path:
    pattern = f"*_{dataset}_gcn_prototype_clst0.02_sep0_graph-structure-sweep"
    candidates = [
        path for path in RUNS_DIR.glob(pattern)
        if path.is_dir() and path.stat().st_mtime >= started_at
    ]
    if not candidates:
        raise FileNotFoundError(f"No archived run found for {dataset}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_one(graph_mode: str, dataset: str) -> dict:
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
        "graph-structure-sweep",
    ]

    started_at = datetime.now().timestamp()
    print(f"\n=== {graph_mode} | {dataset} ===", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)

    run_dir = latest_run_for_dataset(dataset, started_at)
    metrics = load_json(run_dir / "test_metrics.json")
    metadata = load_json(run_dir / "dataset_metadata.json")
    confusion = metrics.get("confusion_matrix", {})

    return {
        "graph_mode": graph_mode,
        "dataset": dataset,
        "k": metadata.get("k", 5),
        "pr_auc": metrics.get("pr_auc"),
        "acc": metrics.get("acc"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "specificity": metrics.get("specificity"),
        "f1": metrics.get("f1"),
        "balanced_acc": metrics.get("balanced_acc"),
        "threshold": metrics.get("threshold"),
        "tn": confusion.get("tn"),
        "fp": confusion.get("fp"),
        "fn": confusion.get("fn"),
        "tp": confusion.get("tp"),
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
    }


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_outputs(rows: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "graph_mode",
        "dataset",
        "k",
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
    with (OUTPUT_DIR / "graph_structure_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    headers = ["Graph", "k", "PR-AUC", "F1", "Recall", "Specificity", "Balanced Acc", "Accuracy", "Threshold", "Run"]
    lines = [
        "# Graph structure sweep",
        "",
        "Fixed configuration: prototype GCN, k=5, clst=0.02, sep=0.0, LOS included, explain_n=0.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["graph_mode"],
                    fmt(row["k"]),
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
    lines.extend(["", f"Best by PR-AUC: {best['graph_mode']} ({best['pr_auc']:.6f}).", ""])
    (OUTPUT_DIR / "graph_structure_results.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = [run_one(graph_mode, dataset) for graph_mode, dataset in DATASETS]
    write_outputs(rows)
    print(f"\nWrote {OUTPUT_DIR / 'graph_structure_results.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'graph_structure_results.txt'}")


if __name__ == "__main__":
    main()
