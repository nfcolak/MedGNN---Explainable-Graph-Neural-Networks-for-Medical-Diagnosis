"""Run and validate the three-method fixed-cohort explanation phase."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.lib.benchmark_contract import BenchmarkSpec
from shared.lib.explanation_contract import (
    COHORT_SIZE,
    EXPLANATION_METRIC_KEYS,
    STANDARDIZED_METHODS,
    load_and_validate_method_output,
    load_explanation_cohort,
    validate_cohort_size,
)


DEFAULT_COHORT = Path("comparison/standardized/explanation_subjects.json")
DEFAULT_SPLIT = Path("comparison/canonical_split.json")
DEFAULT_DATASET = Path("data/merged_ed.csv")
DEFAULT_OUTPUT_ROOT = Path("comparison/standardized/explanations")
_METHOD_MODULES = {
    "protgnn": "protgnn_analysis.explainability.explain_standardized",
    "gsat": "gsat_analysis.explainability.explain_gsat",
    "graphcare": "graphcare_analysis.explainability.explain_graphcare",
}
_METHOD_PYTHON = {
    "protgnn": "python3",
    "gsat": "python3",
    "graphcare": ".venv-graphcare/bin/python3",
}
_CHECKPOINTS = {
    "protgnn": Path("checkpoints/mimic_intra_patient_disease/gcn_best.pth"),
    "gsat": Path("gsat_model.pt"),
    "graphcare": Path("graphcare_model.pt"),
}


def explanation_set_dir(output_root: Path, topology: str, seed: int) -> Path:
    return Path(output_root) / topology / f"seed_{seed}"


def checkpoint_for(method: str, topology: str, seed: int) -> Path:
    run_dir = (
        Path("comparison/standardized/results")
        / method
        / topology
        / f"seed_{seed}"
    )
    return run_dir / _CHECKPOINTS[method]


def build_commands(
    *,
    topology: str,
    seed: int,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    cohort_path: Path = DEFAULT_COHORT,
    split_path: Path = DEFAULT_SPLIT,
    dataset_path: Path = DEFAULT_DATASET,
    require_checkpoints: bool = True,
    cohort_size: int = COHORT_SIZE,
) -> list[list[str]]:
    """Build three explicit isolated commands without creating output paths."""
    validate_cohort_size(cohort_size)
    set_dir = explanation_set_dir(output_root, topology, seed)
    commands: list[list[str]] = []
    for method in STANDARDIZED_METHODS:
        spec = BenchmarkSpec(method, topology, seed)
        checkpoint = checkpoint_for(method, spec.structure, spec.seed)
        absolute_checkpoint = PROJECT_ROOT / checkpoint
        if require_checkpoints and not absolute_checkpoint.is_file():
            raise FileNotFoundError(
                f"standardized {method} checkpoint missing: {checkpoint}"
            )
        command = [
            _METHOD_PYTHON[method],
            "-u",
            "-m",
            _METHOD_MODULES[method],
            "--checkpoint",
            str(checkpoint),
            "--graph-structure",
            spec.structure,
            "--canonical-split",
            str(split_path),
            "--cohort-size",
            str(cohort_size),
            "--cohort",
            str(cohort_path),
            "--dataset",
            str(dataset_path),
            "--seed",
            str(spec.seed),
            "--out-dir",
            str(set_dir / method),
        ]
        commands.append(command)
    return commands


def validate_cross_method_outputs(
    set_dir: Path,
    *,
    cohort_path: Path,
    split_path: Path,
    dataset_path: Path,
    topology: str,
    seed: int,
    require_checkpoints: bool = False,
    cohort_size: int = COHORT_SIZE,
) -> dict[str, Any]:
    """Require exact cohort/schema/top-k parity across all three methods."""
    cohort = load_explanation_cohort(
        cohort_path, split_path=split_path, dataset_path=dataset_path, expected_count=cohort_size
    )
    records_by_method: dict[str, dict[str, dict[str, Any]]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for method in STANDARDIZED_METHODS:
        manifest, records = load_and_validate_method_output(
            Path(set_dir) / method,
            method=method,
            topology=topology,
            seed=seed,
            cohort=cohort,
        )
        if require_checkpoints:
            checkpoint = manifest.get("checkpoint")
            if not isinstance(checkpoint, dict) or not checkpoint.get("sha256"):
                raise ValueError(f"{method} checkpoint provenance is missing.")
        manifests[method] = manifest
        records_by_method[method] = {
            record["subject_id"]: record for record in records
        }

    reference_method = STANDARDIZED_METHODS[0]
    for subject in cohort.subject_ids:
        reference = records_by_method[reference_method][subject]
        reference_keys = set(reference["node_explanation"])
        reference_top_k = reference["node_explanation"]["top_k"]
        reference_true_class = reference["true_class_id"]
        for method in STANDARDIZED_METHODS[1:]:
            current = records_by_method[method][subject]
            if set(current["node_explanation"]) != reference_keys:
                raise ValueError(
                    f"cross-method metric schema mismatch for subject {subject}."
                )
            if current["node_explanation"]["top_k"] != reference_top_k:
                raise ValueError(
                    f"cross-method top-k resolution mismatch for subject {subject}."
                )
            if current["true_class_id"] != reference_true_class:
                raise ValueError(
                    f"cross-method true class mismatch for subject {subject}."
                )

    return {
        "schema": "medgnn.standardized_explanation_validation",
        "schema_version": 1,
        "topology": topology,
        "seed": seed,
        "methods": list(STANDARDIZED_METHODS),
        "subject_count": len(cohort.subject_ids),
        "subject_ids": list(cohort.subject_ids),
        "metric_keys": list(EXPLANATION_METRIC_KEYS),
        "cohort_sha256": cohort.artifact_sha256,
        "split_sha256": cohort.split_sha256,
        "dataset_sha256": cohort.dataset_sha256,
        "checkpoint_sha256": {
            method: (
                manifests[method].get("checkpoint") or {}
            ).get("sha256")
            for method in STANDARDIZED_METHODS
        },
    }


def main(
    *,
    topology: str,
    seed: int,
    output_root: Path,
    cohort_path: Path,
    split_path: Path,
    dataset_path: Path,
    dry_run: bool = False,
    cohort_size: int = COHORT_SIZE,
) -> int:
    commands = build_commands(
        topology=topology,
        seed=seed,
        output_root=output_root,
        cohort_path=cohort_path,
        split_path=split_path,
        dataset_path=dataset_path,
        require_checkpoints=not dry_run,
        cohort_size=cohort_size,
    )
    if dry_run:
        print(json.dumps(commands, indent=2))
        return 0

    load_explanation_cohort(
        PROJECT_ROOT / cohort_path,
        split_path=PROJECT_ROOT / split_path,
        dataset_path=PROJECT_ROOT / dataset_path,
        expected_count=cohort_size,
    )
    set_dir = explanation_set_dir(output_root, topology, seed)
    if (PROJECT_ROOT / set_dir).exists():
        raise FileExistsError(
            f"standardized explanation set already exists: {set_dir}"
        )
    for command in commands:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    validation = validate_cross_method_outputs(
        PROJECT_ROOT / set_dir,
        cohort_path=PROJECT_ROOT / cohort_path,
        split_path=PROJECT_ROOT / split_path,
        dataset_path=PROJECT_ROOT / dataset_path,
        cohort_size=cohort_size,
        topology=topology,
        seed=seed,
        require_checkpoints=True,
    )
    (PROJECT_ROOT / set_dir / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in validation.items() if key != 'subject_ids'}, indent=2))
    return 0


def cli(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", "--structure", dest="topology", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cohort", dest="cohort_path", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--split", dest="split_path", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--dataset", dest="dataset_path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cohort-size", type=int, default=COHORT_SIZE, help="Expected test subjects (default: 500).")
    parser.add_argument("--dry-run", action="store_true")
    return main(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(cli())
