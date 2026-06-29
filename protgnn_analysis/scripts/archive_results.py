#!/usr/bin/env python3
"""
Archive the current outputs/results directory into a timestamped run folder.

The archive includes a machine-readable metadata file and a plain-text summary
with the model, parameters, command, metrics, and copied artefact counts.
"""

import argparse
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"


def load_json(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_part(value):
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value)
    return value.strip("-") or "run"


def format_float(value):
    if value is None:
        return "NA"
    return f"{float(value):g}"


def build_run_name(timestamp, model_config, tag):
    dataset = safe_part(model_config.get("dataset", "dataset"))
    model = safe_part(model_config.get("model", "model"))
    prot = "prototype" if model_config.get("enable_prototypes") else "standard"
    clst = format_float(model_config.get("clst_weight"))
    sep = format_float(model_config.get("sep_weight"))
    parts = [timestamp, dataset, model, prot, f"clst{clst}", f"sep{sep}"]
    if tag:
        parts.append(safe_part(tag))
    return "_".join(parts)


def should_skip(path):
    return path.name == ".DS_Store"


def copy_tree_contents(source_dir, target_dir):
    copied_files = []
    for source_path in source_dir.rglob("*"):
        if should_skip(source_path):
            continue
        relative_path = source_path.relative_to(source_dir)
        target_path = target_dir / relative_path
        if source_path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        copied_files.append(str(relative_path))
    return sorted(copied_files)


def count_files(target_dir, dirname):
    folder = target_dir / dirname
    if not folder.exists():
        return 0
    return sum(1 for path in folder.rglob("*") if path.is_file())


def derived_dataset_info(dataset_metadata):
    feature_cols = dataset_metadata.get("feature_cols", [])
    return {
        "input_dim": len(feature_cols) if feature_cols else "unknown",
        "los_hours_included": "los_hours" in feature_cols,
    }


def write_json(path, data):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_summary(path, metadata):
    cfg = metadata["model_config"]
    metrics = metadata["test_metrics"]
    dataset_meta = metadata["dataset_metadata"]
    derived_meta = metadata["derived_dataset_info"]

    lines = [
        "RUN SUMMARY",
        "===========",
        f"Run id: {metadata['run_id']}",
        f"Created at: {metadata['created_at']}",
        f"Source results dir: {metadata['source_results_dir']}",
        f"Command: {metadata['command'] or 'not provided'}",
        "",
        "MODEL",
        "-----",
        f"Dataset: {cfg.get('dataset', 'unknown')}",
        f"Model: {cfg.get('model', 'unknown')}",
        f"Prototype enabled: {cfg.get('enable_prototypes', 'unknown')}",
        f"LOS included: {derived_meta.get('los_hours_included', 'unknown')}",
        f"Input dim: {derived_meta.get('input_dim', 'unknown')}",
        f"Output dim: {cfg.get('output_dim', 'unknown')}",
        f"Latent dim: {cfg.get('latent_dim', 'unknown')}",
        f"MLP hidden: {cfg.get('mlp_hidden', 'unknown')}",
        f"Readout: {cfg.get('readout', 'unknown')}",
        "",
        "PARAMETERS",
        "----------",
        f"learning_rate: {cfg.get('learning_rate', 'unknown')}",
        f"batch_size: {cfg.get('batch_size', 'unknown')}",
        f"weight_decay: {cfg.get('weight_decay', 'unknown')}",
        f"dropout: {cfg.get('dropout', 'unknown')}",
        f"max_epochs: {cfg.get('max_epochs', 'unknown')}",
        f"early_stopping: {cfg.get('early_stopping', 'unknown')}",
        f"warm_epochs: {cfg.get('warm_epochs', 'unknown')}",
        f"proj_epochs: {cfg.get('proj_epochs', 'unknown')}",
        f"num_prototypes_per_class: {cfg.get('num_prototypes_per_class', 'unknown')}",
        f"clst_weight: {cfg.get('clst_weight', 'unknown')}",
        f"sep_weight: {cfg.get('sep_weight', 'unknown')}",
        f"decision_threshold: {cfg.get('decision_threshold', 'unknown')}",
        f"seed: {cfg.get('seed', 'unknown')}",
        "",
        "TEST METRICS",
        "------------",
        f"loss: {metrics.get('loss', 'unknown')}",
        f"accuracy: {metrics.get('acc', 'unknown')}",
        f"precision: {metrics.get('precision', 'unknown')}",
        f"recall: {metrics.get('recall', 'unknown')}",
        f"specificity: {metrics.get('specificity', 'unknown')}",
        f"f1: {metrics.get('f1', 'unknown')}",
        f"balanced_acc: {metrics.get('balanced_acc', 'unknown')}",
        f"pr_auc: {metrics.get('pr_auc', 'unknown')}",
        f"threshold_used: {metrics.get('threshold', metrics.get('threshold_used', 'unknown'))}",
        "",
        "ARTEFACTS",
        "---------",
        f"Copied files: {metadata['copied_file_count']}",
        f"Explanation files: {metadata['explanation_file_count']}",
        f"Clinical explanation files: {metadata['clinical_explanation_file_count']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Archive outputs/results into a timestamped outputs/runs folder."
    )
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--tag", default="")
    parser.add_argument("--command", default="")
    parser.add_argument(
        "--timestamp",
        default="",
        help="Optional timestamp override, e.g. 2026-05-16_16-19-28.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the archive folder if it already exists.",
    )
    return parser.parse_args()


def archive_results(results_dir, runs_dir, tag="", command="", timestamp="", overwrite=False):
    results_dir = Path(results_dir).resolve()
    runs_dir = Path(runs_dir).resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_config = load_json(results_dir / "model_config.json")
    test_metrics = load_json(results_dir / "test_metrics.json")
    dataset_metadata = load_json(results_dir / "dataset_metadata.json")

    run_id = build_run_name(timestamp, model_config, tag)
    target_dir = runs_dir / run_id

    if target_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Archive folder already exists: {target_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True)
    copied_files = copy_tree_contents(results_dir, target_dir)

    metadata = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_results_dir": str(results_dir),
        "archive_dir": str(target_dir),
        "command": command,
        "tag": tag,
        "model_config": model_config,
        "test_metrics": test_metrics,
        "dataset_metadata": dataset_metadata,
        "derived_dataset_info": derived_dataset_info(dataset_metadata),
        "copied_files": copied_files,
        "copied_file_count": len(copied_files),
        "explanation_file_count": count_files(target_dir, "explanations"),
        "clinical_explanation_file_count": count_files(target_dir, "clinical_explanations"),
    }

    write_json(target_dir / "run_metadata.json", metadata)
    write_summary(target_dir / "run_summary.txt", metadata)
    return target_dir


def main():
    args = parse_args()
    target_dir = archive_results(
        results_dir=args.results_dir,
        runs_dir=args.runs_dir,
        tag=args.tag,
        command=args.command,
        timestamp=args.timestamp,
        overwrite=args.overwrite,
    )
    print(target_dir)


if __name__ == "__main__":
    main()
