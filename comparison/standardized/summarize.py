"""Validate and aggregate standardized benchmark run manifests."""
from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
import csv
from dataclasses import dataclass
import json
import math
from numbers import Real
from pathlib import Path
import statistics
import sys
from typing import Any, Optional


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.lib.benchmark_contract import (
    ALLOWED_SEEDS,
    EXPECTED_CLASSES,
    PRIMARY_STRUCTURES,
    BenchmarkSpec,
)
from shared.lib.graph_structures import GRAPH_STRUCTURES, SUPPORTED_METHODS
from shared.lib.run_manifest import (
    SCHEMA_VERSION,
    STANDARDIZED_METRICS,
    topology_policy_fingerprint,
)


_METHOD_ORDER = {name: index for index, name in enumerate(SUPPORTED_METHODS)}
_TOPOLOGY_ORDER = {name: index for index, name in enumerate(GRAPH_STRUCTURES)}
_REQUIRED_COUNTS = {"train": 59607, "validation": 7448, "test": 7456}
_PRIMARY_TOPOLOGY_PARAMETERS = {
    "star": {
        "patient_hub": True,
        "patient_concept_edges": "bidirectional",
        "concept_concept_edges": "none",
    },
    "cooccur": {
        "patient_hub": True,
        "patient_concept_edges": "bidirectional",
        "concept_concept_edges": "training_fold_pmi",
        "pmi_threshold": 2.0,
    },
}
_HEX_DIGITS = frozenset("0123456789abcdef")


def _topology_category(topology: str) -> str:
    metadata = GRAPH_STRUCTURES[topology]
    if metadata["primary"]:
        return "primary"
    supported_methods = sum(bool(metadata[method]) for method in SUPPORTED_METHODS)
    if supported_methods == 1:
        return "method_specific"
    return "common_secondary"


@dataclass(frozen=True)
class ValidatedRun:
    manifest_path: Path
    method: str
    topology: str
    seed: int
    metrics: dict[str, float]
    parameter_count: int
    dataset_hash: str
    split_hash: str
    topology_policy_fingerprint: str

    @property
    def primary(self) -> bool:
        return self.topology in PRIMARY_STRUCTURES

    @property
    def category(self) -> str:
        return _topology_category(self.topology)


@dataclass(frozen=True)
class IncompleteRun:
    manifest_path: Path
    method: str
    topology: str
    seed: int
    status: str
    error: Optional[str]

    @property
    def category(self) -> str:
        return _topology_category(self.topology)


class RunCollection(Sequence[ValidatedRun]):
    """Stable completed-run sequence with separately retained non-completed runs."""

    def __init__(
        self,
        completed: Sequence[ValidatedRun],
        incomplete: Sequence[IncompleteRun] = (),
    ) -> None:
        self._completed = tuple(completed)
        self.incomplete = tuple(incomplete)
        self.common_secondary = tuple(
            run for run in completed if run.category == "common_secondary"
        )
        self.method_specific = tuple(
            run for run in completed if run.category == "method_specific"
        )

    def __getitem__(self, index):
        return self._completed[index]

    def __len__(self) -> int:
        return len(self._completed)

    def __iter__(self) -> Iterator[ValidatedRun]:
        return iter(self._completed)


@dataclass(frozen=True)
class SummaryRows:
    per_run_rows: tuple[dict[str, Any], ...]
    aggregate_rows: tuple[dict[str, Any], ...]
    incomplete_cells: tuple[dict[str, Any], ...]
    incomplete_run_rows: tuple[dict[str, Any], ...]
    common_secondary_rows: tuple[dict[str, Any], ...]
    method_specific_rows: tuple[dict[str, Any], ...]


def _run_key(run: ValidatedRun | IncompleteRun) -> tuple[int, int, int]:
    return (
        _METHOD_ORDER[run.method],
        _TOPOLOGY_ORDER[run.topology],
        run.seed,
    )


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be readable UTF-8 JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _metric_values(
    payload: dict[str, Any],
    method: str,
    path: Path,
    manifest_parameter_count: int,
) -> tuple[dict[str, float], int]:
    if method == "protgnn":
        metrics = payload
    else:
        if set(payload) != {"parameter_count", "test"}:
            raise ValueError(f"Metrics for {method} have unexpected top-level fields: {path}")
        metrics = payload.get("test")
        if not isinstance(metrics, dict):
            raise ValueError(f"Metrics for {method} must contain a test object: {path}")
    metric_names = set(metrics) - {"parameter_count", "loss"}
    if metric_names != set(STANDARDIZED_METRICS):
        raise ValueError(f"Metrics must contain the exact six standardized metrics: {path}")
    values = {}
    for name in STANDARDIZED_METRICS:
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(f"Metric {name} must be a finite numeric non-boolean value: {path}")
        numeric = float(value)
        if not 0.0 <= numeric <= 1.0:
            raise ValueError(f"Metric {name} must be in [0, 1]: {path}")
        values[name] = numeric
    parameter_count = payload.get("parameter_count", metrics.get("parameter_count"))
    if type(parameter_count) is not int or parameter_count < 0:
        raise ValueError(f"Metrics parameter_count must be a nonnegative exact integer: {path}")
    nested_count = metrics.get("parameter_count")
    if nested_count is not None and (
        type(nested_count) is not int or nested_count < 0
    ):
        raise ValueError(f"Metrics parameter_count must be a nonnegative exact integer: {path}")
    if nested_count is not None and nested_count != parameter_count:
        raise ValueError(f"Metrics parameter_count values disagree: {path}")
    if parameter_count != manifest_parameter_count:
        raise ValueError(f"Metrics parameter_count disagrees with the manifest: {path}")
    return values, parameter_count


def _validate_hash(value: Any, label: str, path: Path) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest: {path}")
    return value


def _require_one_hash(runs: Sequence[ValidatedRun], field: str, label: str) -> None:
    values = {getattr(run, field) for run in runs}
    if len(values) > 1:
        raise ValueError(f"Completed runs have incompatible {label} values.")


def _validate_manifest_contract(path: Path, manifest: dict[str, Any]) -> tuple[str, str, int]:
    if manifest.get("schema_version") != SCHEMA_VERSION or type(
        manifest.get("schema_version")
    ) is not int:
        raise ValueError(f"schema_version must be exactly {SCHEMA_VERSION}: {path}")
    method = manifest.get("method")
    topology = manifest.get("topology")
    seed = manifest.get("seed")
    if type(method) is not str or type(topology) is not str or type(seed) is not int:
        raise ValueError(f"Invalid manifest method/topology/seed: {path}")
    try:
        BenchmarkSpec(method, topology, seed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid manifest method/topology/seed: {path}: {exc}") from exc
    if manifest.get("class_ordering") != list(EXPECTED_CLASSES):
        raise ValueError(f"class_ordering must exactly match the benchmark contract: {path}")
    for field in ("canonical_fold_counts", "effective_fold_counts"):
        counts = manifest.get(field)
        if (
            not isinstance(counts, dict)
            or counts != _REQUIRED_COUNTS
            or any(type(value) is not int for value in counts.values())
        ):
            raise ValueError(f"{field} must exactly match the benchmark contract: {path}")
    parameters = manifest.get("topology_parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"Invalid topology_parameters: {path}")
    if topology in _PRIMARY_TOPOLOGY_PARAMETERS and parameters != _PRIMARY_TOPOLOGY_PARAMETERS[topology]:
        raise ValueError(f"Manifest does not use the canonical topology policy for {topology}: {path}")
    try:
        expected_fingerprint = topology_policy_fingerprint(parameters)
    except ValueError as exc:
        raise ValueError(f"Invalid topology_parameters: {path}: {exc}") from exc
    actual_fingerprint = _validate_hash(
        manifest.get("topology_policy_fingerprint"),
        "topology-policy fingerprint",
        path,
    )
    if actual_fingerprint != expected_fingerprint:
        raise ValueError(f"Manifest topology-policy fingerprint does not match parameters: {path}")
    parameter_count = manifest.get("parameter_count")
    if type(parameter_count) is not int or parameter_count < 0:
        raise ValueError(f"parameter_count must be a nonnegative exact integer: {path}")
    return method, topology, seed


def _completed_run(path: Path, manifest: dict[str, Any]) -> ValidatedRun:
    method, topology, seed = _validate_manifest_contract(path, manifest)
    metrics_path = path.parent / manifest["metrics_path"]
    metrics_payload = _load_json_object(metrics_path, "metrics artifact")
    metrics, parameter_count = _metric_values(
        metrics_payload, method, metrics_path, manifest["parameter_count"]
    )
    return ValidatedRun(
        manifest_path=path,
        method=method,
        topology=topology,
        seed=seed,
        metrics=metrics,
        parameter_count=parameter_count,
        dataset_hash=_validate_hash(
            manifest["dataset"]["sha256"], "dataset SHA-256", path
        ),
        split_hash=_validate_hash(manifest["split"]["sha256"], "split SHA-256", path),
        topology_policy_fingerprint=manifest["topology_policy_fingerprint"],
    )


def load_compatible_runs(results_dir: Path) -> RunCollection:
    """Load completed manifests in stable benchmark order."""
    root = Path(results_dir)
    completed = []
    incomplete = []
    for path in sorted(root.rglob("run_manifest.json")):
        manifest = _load_json_object(path, "run manifest")
        status = manifest.get("status")
        if status == "completed":
            completed.append(_completed_run(path, manifest))
        elif status in {"running", "failed"}:
            method = manifest.get("method")
            topology = manifest.get("topology")
            seed = manifest.get("seed")
            if type(method) is not str or type(topology) is not str or type(seed) is not int:
                raise ValueError(f"Invalid incomplete manifest method/topology/seed: {path}")
            try:
                BenchmarkSpec(method, topology, seed)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid incomplete manifest method/topology/seed: {path}: {exc}"
                ) from exc
            error = manifest.get("error")
            incomplete.append(
                IncompleteRun(path, method, topology, seed, status, error)
            )
        else:
            raise ValueError(f"Manifest status must be running, failed, or completed: {path}")
    completed.sort(key=_run_key)
    incomplete.sort(key=_run_key)
    _require_one_hash(completed, "dataset_hash", "dataset SHA-256")
    _require_one_hash(completed, "split_hash", "split SHA-256")
    fingerprints_by_topology: dict[str, set[str]] = {}
    for run in completed:
        fingerprints_by_topology.setdefault(run.topology, set()).add(
            run.topology_policy_fingerprint
        )
    mismatched = [
        topology
        for topology, fingerprints in fingerprints_by_topology.items()
        if len(fingerprints) != 1
    ]
    if mismatched:
        raise ValueError(
            "Completed runs have incompatible topology-policy fingerprints for: "
            + ", ".join(sorted(mismatched, key=_TOPOLOGY_ORDER.get))
        )
    identities = [(run.method, run.topology, run.seed) for run in [*completed, *incomplete]]
    if len(identities) != len(set(identities)):
        raise ValueError("Run manifests contain duplicate method/topology/seed cells.")
    return RunCollection(completed, incomplete)


def _per_run_row(run: ValidatedRun) -> dict[str, Any]:
    row: dict[str, Any] = {
        "category": run.category,
        "method": run.method,
        "topology": run.topology,
        "seed": run.seed,
        "status": "completed",
        "error": "",
        "parameter_count": run.parameter_count,
    }
    row.update(run.metrics)
    return row


def _incomplete_run_row(run: IncompleteRun) -> dict[str, Any]:
    return {
        "category": run.category,
        "method": run.method,
        "topology": run.topology,
        "seed": run.seed,
        "status": run.status,
        "error": run.error or "",
    }


def summarize_runs(runs: Sequence[ValidatedRun] | RunCollection) -> SummaryRows:
    """Return deterministic per-run and complete three-seed aggregate rows."""
    completed = sorted(list(runs), key=_run_key)
    incomplete = tuple(runs.incomplete) if isinstance(runs, RunCollection) else ()
    groups: dict[tuple[str, str], list[ValidatedRun]] = {}
    for run in completed:
        groups.setdefault((run.method, run.topology), []).append(run)

    aggregate_rows = []
    incomplete_cells = []
    observed_primary = {
        (run.method, run.topology)
        for run in [*completed, *incomplete]
        if run.topology in PRIMARY_STRUCTURES
    }
    for method, topology in sorted(
        observed_primary,
        key=lambda cell: (_METHOD_ORDER[cell[0]], _TOPOLOGY_ORDER[cell[1]]),
    ):
        group = groups.get((method, topology), [])
        completed_seeds = {run.seed for run in group}
        missing_seeds = set(ALLOWED_SEEDS) - completed_seeds
        recorded = [
            run
            for run in incomplete
            if (run.method, run.topology) == (method, topology)
        ]
        if completed_seeds == set(ALLOWED_SEEDS) and len(group) == len(ALLOWED_SEEDS):
            counts = {run.parameter_count for run in group}
            if len(counts) != 1:
                raise ValueError(
                    f"Completed cell has incompatible parameter_count values: {method}/{topology}"
                )
            row: dict[str, Any] = {
                "method": method,
                "topology": topology,
                "seed_count": len(ALLOWED_SEEDS),
                "parameter_count": counts.pop(),
            }
            for metric in STANDARDIZED_METRICS:
                values = [run.metrics[metric] for run in group]
                row[f"{metric}_mean"] = statistics.mean(values)
                row[f"{metric}_sd"] = statistics.stdev(values)
            aggregate_rows.append(row)
        else:
            incomplete_cells.append(
                {
                    "method": method,
                    "topology": topology,
                    "completed_seeds": ";".join(str(seed) for seed in sorted(completed_seeds)),
                    "missing_seeds": ";".join(str(seed) for seed in sorted(missing_seeds)),
                    "recorded_incomplete": ";".join(
                        f"{run.seed}:{run.status}" for run in sorted(recorded, key=_run_key)
                    ),
                }
            )

    completed_rows = tuple(_per_run_row(run) for run in completed)
    incomplete_run_rows = tuple(_incomplete_run_row(run) for run in incomplete)
    per_run_rows = tuple(sorted(
        (*completed_rows, *incomplete_run_rows),
        key=lambda row: (
            _METHOD_ORDER[row["method"]],
            _TOPOLOGY_ORDER[row["topology"]],
            row["seed"],
        ),
    ))
    return SummaryRows(
        per_run_rows=per_run_rows,
        aggregate_rows=tuple(aggregate_rows),
        incomplete_cells=tuple(incomplete_cells),
        incomplete_run_rows=incomplete_run_rows,
        common_secondary_rows=tuple(
            row for row in completed_rows if row["category"] == "common_secondary"
        ),
        method_specific_rows=tuple(
            row for row in completed_rows if row["category"] == "method_specific"
        ),
    )


_PER_RUN_FIELDS = (
    "category",
    "method",
    "topology",
    "seed",
    "status",
    "error",
    "parameter_count",
    *STANDARDIZED_METRICS,
)
_AGGREGATE_FIELDS = (
    "method",
    "topology",
    "seed_count",
    "parameter_count",
    *(field for metric in STANDARDIZED_METRICS for field in (f"{metric}_mean", f"{metric}_sd")),
)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: (
                    format(row.get(field, ""), ".12g")
                    if isinstance(row.get(field, ""), float)
                    else row.get(field, "")
                )
                for field in fields
            })


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return lines


def _write_markdown(path: Path, summary: SummaryRows) -> None:
    lines = ["# Standardized Benchmark Summary", "", "## Primary comparison", ""]
    primary_headers = ["Method", "Topology", "Parameters", *STANDARDIZED_METRICS]
    primary_rows = []
    for row in summary.aggregate_rows:
        primary_rows.append([
            row["method"],
            row["topology"],
            row["parameter_count"],
            *(
                f'{row[f"{metric}_mean"]:.6f} ± {row[f"{metric}_sd"]:.6f}'
                for metric in STANDARDIZED_METRICS
            ),
        ])
    lines.extend(_markdown_table(primary_headers, primary_rows))
    lines.extend(["", "## Incomplete primary cells", ""])
    lines.extend(_markdown_table(
        ["Method", "Topology", "Completed seeds", "Missing seeds", "Recorded incomplete"],
        [
            [
                row["method"],
                row["topology"],
                row["completed_seeds"],
                row["missing_seeds"],
                row["recorded_incomplete"],
            ]
            for row in summary.incomplete_cells
        ],
    ))
    lines.extend(["", "## Common secondary runs", ""])
    lines.extend(_markdown_table(
        ["Method", "Topology", "Seed", "Parameters", *STANDARDIZED_METRICS],
        [
            [
                row["method"],
                row["topology"],
                row["seed"],
                row["parameter_count"],
                *(f'{row[metric]:.6f}' for metric in STANDARDIZED_METRICS),
            ]
            for row in summary.common_secondary_rows
        ],
    ))
    lines.extend(["", "## Method-specific runs", ""])
    lines.extend(_markdown_table(
        ["Method", "Topology", "Seed", "Parameters", *STANDARDIZED_METRICS],
        [
            [
                row["method"],
                row["topology"],
                row["seed"],
                row["parameter_count"],
                *(f'{row[metric]:.6f}' for metric in STANDARDIZED_METRICS),
            ]
            for row in summary.method_specific_rows
        ],
    ))
    lines.extend(["", "## Incomplete runs", ""])
    lines.extend(_markdown_table(
        ["Category", "Method", "Topology", "Seed", "Status", "Error"],
        [
            [
                row["category"],
                row["method"],
                row["topology"],
                row["seed"],
                row["status"],
                row["error"],
            ]
            for row in summary.incomplete_run_rows
        ],
    ))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--summary-aggregate-csv", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    output_paths = (args.summary_csv, args.summary_aggregate_csv, args.summary_md)
    if len({Path(path).absolute() for path in output_paths}) != len(output_paths):
        raise ValueError("Summary output paths must be distinct.")
    summary = summarize_runs(load_compatible_runs(args.results_dir))
    _write_csv(args.summary_csv, summary.per_run_rows, _PER_RUN_FIELDS)
    _write_csv(
        args.summary_aggregate_csv,
        summary.aggregate_rows,
        _AGGREGATE_FIELDS,
    )
    _write_markdown(args.summary_md, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
