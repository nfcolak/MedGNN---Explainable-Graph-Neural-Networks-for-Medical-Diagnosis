"""Fixed-cohort and metric contract for standardized node explanations.

The scientific node-selection policy is fixed at 20% of graph nodes, using
``floor(0.2 * n)`` with a minimum of one node.  Every standardized method uses
this module to resolve that policy and to compute prediction-targeted fidelity+
(mask selected nodes), fidelity- (retain selected nodes), and 90%-mass sparsity.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from shared.lib.benchmark_contract import file_sha256, load_canonical_split
from shared.lib.fidelity import (
    NODE_TOP_K_FRACTION,
    NODE_TOP_K_POLICY,
    fidelity_minus,
    fidelity_plus,
    resolve_k,
    sparsity,
)


COHORT_SCHEMA = "medgnn.explanation_cohort"
COHORT_SCHEMA_VERSION = 1
COHORT_SIZE = 500
TEST_FOLD = 2
STANDARDIZED_METHODS = ("protgnn", "gsat", "graphcare")
EXPLANATION_SCHEMA = "medgnn.standardized_node_explanation"
EXPLANATION_SCHEMA_VERSION = 1
OUTPUT_MANIFEST_SCHEMA = "medgnn.standardized_explanation_manifest"
OUTPUT_MANIFEST_VERSION = 1
TOP_K_POLICY = NODE_TOP_K_POLICY
TOP_K_FRACTION = NODE_TOP_K_FRACTION
EXPLANATION_METRIC_KEYS = (
    "target",
    "node_importance",
    "top_nodes",
    "top_k",
    "fidelity_plus",
    "fidelity_minus",
    "sparsity",
)
RECORD_KEYS = (
    "schema",
    "schema_version",
    "method",
    "subject_id",
    "topology",
    "seed",
    "true_class_id",
    "prediction_class_id",
    "attribution_method",
    "node_explanation",
)


@dataclass(frozen=True)
class ExplanationCohort:
    path: Path
    subject_ids: tuple[str, ...]
    classes: tuple[str, ...]
    split_path: Path
    split_sha256: str
    dataset_path: Path
    dataset_sha256: str
    artifact_sha256: str


@dataclass(frozen=True)
class ResolvedTopK:
    policy: str
    requested_fraction: float
    resolved_k: int
    num_nodes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "requested_fraction": self.requested_fraction,
            "resolved_k": self.resolved_k,
            "num_nodes": self.num_nodes,
        }


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} keys mismatch; missing={missing}, extra={extra}.")


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be readable valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def _dataset_labels(dataset_path: Path, wanted: set[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    try:
        with dataset_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"subject_id", "disease_1"}.issubset(reader.fieldnames or ()):
                raise ValueError("dataset must contain subject_id and disease_1 columns.")
            for row in reader:
                subject = row.get("subject_id", "")
                if subject not in wanted:
                    continue
                if subject in labels:
                    raise ValueError(
                        f"dataset contains duplicate explanation subject {subject!r}."
                    )
                labels[subject] = row.get("disease_1", "")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"dataset must be a readable UTF-8 CSV: {dataset_path}") from exc
    missing = wanted - set(labels)
    if missing:
        raise ValueError(
            "dataset is missing explanation subjects; "
            f"sample={sorted(missing)[:3]}."
        )
    return labels


def validate_cohort_size(count: Any) -> int:
    """Reject non-integral or empty cohort requests before any I/O."""
    if type(count) is not int or count < 1:
        raise ValueError("cohort size must be a positive exact integer.")
    return count


def load_explanation_cohort(
    cohort_path: Any, *, split_path: Any, dataset_path: Any, expected_count: int = COHORT_SIZE
) -> ExplanationCohort:
    """Load exactly the requested count, bound to the current split and data."""
    validate_cohort_size(expected_count)
    cohort_file = Path(cohort_path)
    split_file = Path(split_path)
    dataset_file = Path(dataset_path)
    payload = _load_json_object(cohort_file, "explanation cohort")
    _require_exact_keys(
        payload,
        {
            "schema",
            "schema_version",
            "seed",
            "count",
            "canonical_split",
            "dataset",
            "class_counts",
            "subject_ids",
        },
        "explanation cohort",
    )
    if payload["schema"] != COHORT_SCHEMA or payload["schema_version"] != COHORT_SCHEMA_VERSION:
        raise ValueError(
            f"explanation cohort must use {COHORT_SCHEMA!r} version "
            f"{COHORT_SCHEMA_VERSION}."
        )
    if type(payload["count"]) is not int or payload["count"] != expected_count:
        raise ValueError(f"explanation cohort must declare exactly {expected_count} subjects.")
    subjects = payload["subject_ids"]
    if (
        not isinstance(subjects, list)
        or len(subjects) != expected_count
        or any(type(subject) is not str or not subject for subject in subjects)
        or len(set(subjects)) != expected_count
    ):
        raise ValueError(f"explanation cohort must contain exactly {expected_count} unique IDs.")

    split = load_canonical_split(split_file)
    non_test = [subject for subject in subjects if split["fold"].get(subject) != TEST_FOLD]
    if non_test:
        raise ValueError(
            "every explanation subject must belong to the canonical test fold; "
            f"sample={non_test[:3]}."
        )

    split_ref = payload["canonical_split"]
    dataset_ref = payload["dataset"]
    if not isinstance(split_ref, dict) or not isinstance(dataset_ref, dict):
        raise ValueError("cohort split and dataset provenance must be JSON objects.")
    _require_exact_keys(split_ref, {"path", "sha256"}, "cohort split provenance")
    _require_exact_keys(dataset_ref, {"path", "sha256"}, "cohort dataset provenance")
    current_split_hash = file_sha256(split_file)
    current_dataset_hash = file_sha256(dataset_file)
    if split_ref["sha256"] != current_split_hash:
        raise ValueError("explanation cohort canonical split hash is stale or mismatched.")
    if dataset_ref["sha256"] != current_dataset_hash:
        raise ValueError("explanation cohort dataset hash is stale or mismatched.")

    class_counts = payload["class_counts"]
    if not isinstance(class_counts, dict) or list(class_counts) != list(split["classes"]):
        raise ValueError("explanation cohort class order must equal canonical class order.")
    labels = _dataset_labels(dataset_file, set(subjects))
    observed = {class_name: 0 for class_name in split["classes"]}
    for subject in subjects:
        label = labels[subject]
        if label not in observed:
            raise ValueError(
                f"explanation subject {subject!r} has non-canonical class {label!r}."
            )
        observed[label] += 1
    if class_counts != observed:
        raise ValueError("explanation cohort class counts do not match the current dataset.")

    return ExplanationCohort(
        path=cohort_file,
        subject_ids=tuple(subjects),
        classes=tuple(split["classes"]),
        split_path=split_file,
        split_sha256=current_split_hash,
        dataset_path=dataset_file,
        dataset_sha256=current_dataset_hash,
        artifact_sha256=file_sha256(cohort_file),
    )


def resolve_node_top_k(num_nodes: int) -> ResolvedTopK:
    """Resolve the fixed 20%-floor/minimum-one policy for one graph."""
    if type(num_nodes) is not int or num_nodes < 1:
        raise ValueError("num_nodes must be a positive exact integer.")
    resolved = resolve_k(num_nodes, fraction=TOP_K_FRACTION)
    return ResolvedTopK(TOP_K_POLICY, TOP_K_FRACTION, resolved, num_nodes)


def _importance_array(node_importance: Any, num_nodes: int) -> np.ndarray:
    if torch.is_tensor(node_importance):
        node_importance = node_importance.detach().cpu().numpy()
    importance = np.asarray(node_importance, dtype=float)
    if importance.ndim > 1:
        importance = np.abs(importance).sum(axis=tuple(range(1, importance.ndim)))
    if importance.ndim != 1 or len(importance) != num_nodes:
        raise ValueError("node importance must have exactly one value per node.")
    if not np.isfinite(importance).all():
        raise ValueError("node importance must contain only finite values.")
    return importance


def deterministic_input_gradient(
    wrapper: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    batch: torch.Tensor,
    target_class: int | None = None,
) -> np.ndarray:
    """Return deterministic absolute input×gradient node attribution."""
    wrapper.eval()
    features = x.detach().clone().requires_grad_(True)
    wrapper.zero_grad(set_to_none=True)
    logits = wrapper(features, edge_index, batch=batch)
    if logits.ndim != 2 or logits.size(0) != 1:
        raise ValueError("standardized explanations score exactly one graph at a time.")
    predicted = int(logits.argmax(dim=-1).item())
    target = predicted if target_class is None else int(target_class)
    logits[0, target].backward()
    if features.grad is None:
        raise RuntimeError("model forward produced no gradient for node features.")
    return (features.grad * features).abs().sum(dim=1).detach().cpu().numpy()


def build_node_explanation(
    wrapper: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    batch: torch.Tensor,
    node_importance: Any | None = None,
) -> dict[str, Any]:
    """Compute the one shared standardized node-explanation metric schema."""
    if x.ndim != 2 or x.size(0) < 1:
        raise ValueError("x must contain at least one node-feature row.")
    with torch.no_grad():
        logits = wrapper(x, edge_index, batch=batch)
    if logits.ndim != 2 or logits.size(0) != 1:
        raise ValueError("standardized explanations score exactly one graph at a time.")
    target = int(logits.argmax(dim=-1).item())
    if node_importance is None:
        node_importance = deterministic_input_gradient(
            wrapper, x, edge_index, batch=batch, target_class=target
        )
    importance = _importance_array(node_importance, int(x.size(0)))
    top_k = resolve_node_top_k(int(x.size(0)))
    order = sorted(range(len(importance)), key=lambda index: (-abs(importance[index]), index))
    selected = order[: top_k.resolved_k]
    plus = fidelity_plus(
        wrapper, x, edge_index, importance, target, batch, k=top_k.resolved_k
    )
    minus = fidelity_minus(
        wrapper, x, edge_index, importance, target, batch, k=top_k.resolved_k
    )
    return {
        "target": {"provenance": "model_prediction", "class_id": target},
        "node_importance": [float(value) for value in importance],
        "top_nodes": [
            {"index": int(index), "importance": float(importance[index])}
            for index in selected
        ],
        "top_k": top_k.as_dict(),
        "fidelity_plus": plus,
        "fidelity_minus": minus,
        "sparsity": sparsity(importance),
    }


def subject_id_of(record: Any) -> str:
    value = record.get("subject_id") if isinstance(record, Mapping) else getattr(record, "subject_id", None)
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("subject_id tensor must contain exactly one value.")
        value = value.item()
    if isinstance(value, bool) or value is None:
        raise ValueError("every standardized graph must carry one subject_id.")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if type(value) is str and value:
        return value
    raise ValueError("every standardized graph must carry one string/integer subject_id.")


def select_exact_subject_records(
    records: Iterable[Any], requested_subject_ids: Sequence[str]
) -> list[Any]:
    """Select by ID in cohort order; reject missing or duplicate source records."""
    requested = tuple(requested_subject_ids)
    if len(requested) != len(set(requested)):
        raise ValueError("requested subject IDs must be unique.")
    wanted = set(requested)
    found: dict[str, Any] = {}
    for record in records:
        subject = subject_id_of(record)
        if subject not in wanted:
            continue
        if subject in found:
            raise ValueError(f"duplicate subject {subject!r} in standardized records.")
        found[subject] = record
    missing = [subject for subject in requested if subject not in found]
    if missing:
        raise ValueError(
            f"missing requested subject records ({len(missing)}); sample={missing[:3]}."
        )
    return [found[subject] for subject in requested]


def validate_explanation_record(
    record: Mapping[str, Any], *, method: str, topology: str, seed: int
) -> str:
    if not isinstance(record, Mapping):
        raise ValueError("explanation record must be a JSON object.")
    version = record.get("schema_version")
    _require_exact_keys(record, set(RECORD_KEYS) | ({"graphxai"} if version == 2 else set()), "explanation record")
    if record["schema"] != EXPLANATION_SCHEMA or version not in (1, 2):
        raise ValueError("explanation record schema/version mismatch.")
    if version == 2:
        algorithms = record['graphxai']
        names = {'GradExplainer', 'IntegratedGradExplainer', 'GNNExplainer'}
        if not isinstance(algorithms, Mapping) or set(algorithms) != names:
            raise ValueError('version 2 requires all three GraphXAI algorithms')
        for name, value in algorithms.items():
            if value.get('status') != 'success' or not value.get('provenance', {}).get('source_sha256'):
                raise ValueError('GraphXAI algorithm missing success/source provenance')
            if name == 'GNNExplainer' and not value['provenance'].get('edge_gradient_verified'):
                p = value['provenance']
                if not (p.get('edge_gradient_status') == 'not_applicable_edgeless'
                        and p.get('edge_count') == 0
                        and value['node_explanation']['top_k']['num_nodes'] == 1):
                    raise ValueError('GNNExplainer predictive edge gradient not verified')
            nested = {key: record[key] for key in RECORD_KEYS}
            nested['schema_version'] = 1
            nested['node_explanation'] = value['node_explanation']
            validate_explanation_record(nested, method=method, topology=topology, seed=seed)
            if value['node_explanation']['top_k'] != record['node_explanation']['top_k']:
                raise ValueError('GraphXAI top-k does not match baseline graph')
    if record["method"] != method or record["topology"] != topology or record["seed"] != seed:
        raise ValueError("explanation record method/topology/seed mismatch.")
    subject = record["subject_id"]
    if type(subject) is not str or not subject:
        raise ValueError("explanation record subject_id must be a non-empty string.")
    explanation = record["node_explanation"]
    if not isinstance(explanation, Mapping) or set(explanation) != set(EXPLANATION_METRIC_KEYS):
        raise ValueError("node explanation metric schema mismatch.")
    target = explanation.get("target")
    if target != {
        "provenance": "model_prediction",
        "class_id": record["prediction_class_id"],
    }:
        raise ValueError("node explanation target must be the model prediction.")
    top_k = explanation.get("top_k")
    if not isinstance(top_k, Mapping) or top_k != resolve_node_top_k(
        len(explanation.get("node_importance", []))
    ).as_dict():
        raise ValueError("node explanation top-k policy/resolution mismatch.")
    if explanation["fidelity_plus"].get("k") != top_k["resolved_k"] or explanation[
        "fidelity_minus"
    ].get("k") != top_k["resolved_k"]:
        raise ValueError("fidelity metrics must use the resolved shared top-k.")
    return subject


def _provenance(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return {"path": str(path), "sha256": file_sha256(path)}


def write_standardized_explanations(
    *,
    output_dir: Any,
    records: Sequence[Mapping[str, Any]],
    method: str,
    topology: str,
    seed: int,
    cohort_path: Any,
    split_path: Any,
    dataset_path: Any,
    checkpoint_path: Any | None,
    allow_missing_checkpoint: bool = False,
    expected_count: int = COHORT_SIZE,
) -> dict[str, Any]:
    """Validate then write every and only the fixed cohort, plus provenance."""
    if method not in STANDARDIZED_METHODS:
        raise ValueError(f"unsupported standardized explanation method {method!r}.")
    cohort = load_explanation_cohort(
        cohort_path, split_path=split_path, dataset_path=dataset_path, expected_count=expected_count
    )
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"standardized explanation output must be empty: {output}")
    checkpoint = None if checkpoint_path is None else Path(checkpoint_path)
    if checkpoint is None or not checkpoint.is_file():
        if not allow_missing_checkpoint:
            raise FileNotFoundError(f"standardized explanation checkpoint missing: {checkpoint}")
        checkpoint = None

    by_subject: dict[str, Mapping[str, Any]] = {}
    for record in records:
        subject = validate_explanation_record(
            record, method=method, topology=topology, seed=seed
        )
        if subject in by_subject:
            raise ValueError(f"duplicate subject explanation record {subject!r}.")
        by_subject[subject] = record
    expected = set(cohort.subject_ids)
    actual = set(by_subject)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            f"explanation subject mismatch; missing={missing[:3]}, extra={extra[:3]}."
        )

    output.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    for subject in cohort.subject_ids:
        filename = f"subject_{subject}.json"
        (output / filename).write_text(
            json.dumps(by_subject[subject], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        files.append(filename)
    manifest = {
        "schema": OUTPUT_MANIFEST_SCHEMA,
        "schema_version": OUTPUT_MANIFEST_VERSION,
        "method": method,
        "topology": topology,
        "seed": seed,
        "subject_count": len(cohort.subject_ids),
        "subject_ids": list(cohort.subject_ids),
        "metric_keys": list(EXPLANATION_METRIC_KEYS),
        "top_k_policy": {
            "policy": TOP_K_POLICY,
            "requested_fraction": TOP_K_FRACTION,
        },
        "cohort": _provenance(Path(cohort_path)),
        "canonical_split": _provenance(Path(split_path)),
        "dataset": _provenance(Path(dataset_path)),
        "checkpoint": _provenance(checkpoint),
        "record_files": files,
    }
    (output / "explanation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def load_and_validate_method_output(
    output_dir: Any,
    *,
    method: str,
    topology: str,
    seed: int,
    cohort: ExplanationCohort,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reject missing, duplicate, extra, or malformed persisted records."""
    output = Path(output_dir)
    manifest_path = output / "explanation_manifest.json"
    manifest = _load_json_object(manifest_path, "explanation manifest")
    if (
        manifest.get("schema") != OUTPUT_MANIFEST_SCHEMA
        or manifest.get("schema_version") != OUTPUT_MANIFEST_VERSION
        or manifest.get("method") != method
        or manifest.get("topology") != topology
        or manifest.get("seed") != seed
        or manifest.get("subject_count") != len(cohort.subject_ids)
        or manifest.get("subject_ids") != list(cohort.subject_ids)
        or manifest.get("metric_keys") != list(EXPLANATION_METRIC_KEYS)
        or manifest.get("top_k_policy")
        != {"policy": TOP_K_POLICY, "requested_fraction": TOP_K_FRACTION}
        or manifest.get("cohort", {}).get("sha256") != cohort.artifact_sha256
        or manifest.get("canonical_split", {}).get("sha256") != cohort.split_sha256
        or manifest.get("dataset", {}).get("sha256") != cohort.dataset_sha256
    ):
        raise ValueError(f"{method} explanation manifest contract mismatch.")

    declared_files = manifest.get("record_files")
    if not isinstance(declared_files, list) or len(declared_files) != len(cohort.subject_ids):
        raise ValueError(f"{method} explanation manifest must declare exactly {len(cohort.subject_ids)} records.")
    json_files = sorted(path.name for path in output.glob("*.json") if path.name != manifest_path.name)
    if sorted(declared_files) != json_files:
        declared_counts = {name: declared_files.count(name) for name in set(declared_files)}
        duplicates = sorted(name for name, count in declared_counts.items() if count != 1)
        undeclared = sorted(set(json_files) - set(declared_files))
        absent = sorted(set(declared_files) - set(json_files))
        raise ValueError(
            f"{method} record files missing/duplicate/extra; missing={absent[:3]}, "
            f"duplicate={duplicates[:3]}, extra={undeclared[:3]}."
        )

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for filename in declared_files:
        record = _load_json_object(output / filename, f"{method} explanation record")
        subject = validate_explanation_record(
            record, method=method, topology=topology, seed=seed
        )
        if subject in seen:
            raise ValueError(f"duplicate {method} subject record {subject!r}.")
        seen.add(subject)
        records.append(record)
    missing = set(cohort.subject_ids) - seen
    extra = seen - set(cohort.subject_ids)
    if missing or extra:
        raise ValueError(
            f"{method} subject records missing/extra; missing={sorted(missing)[:3]}, "
            f"extra={sorted(extra)[:3]}."
        )
    return manifest, records
