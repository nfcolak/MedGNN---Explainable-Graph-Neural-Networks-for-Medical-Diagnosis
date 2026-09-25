"""Leakage-safe graph-level features for the matched XGBoost baseline.

Features are deterministic projections of the same event-graph JSONL consumed by
the GCHM runner. Clinical token/unit vocabularies and relation columns are fitted
from labelled training graphs only; identifiers and provenance never enter X.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from comparison.standardized.event_graph_gchm_xgb_v1.labels import (
    DEFAULT_GRAPH_ROOT, sha256,
)
from event_graph_analysis.tensorize import EventGraphTensorizer


KIND_ORDER = ("patient", "visit", "event", "concept", "knowledge")
EVENT_STAT_ORDER = ("count", "value_mean", "value_std", "value_min", "value_max",
                    "time_mean", "available_mean", "observed_count")
EDGE_STAT_ORDER = ("count", "delta_mean", "delta_min", "delta_max", "has_delta_count")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _json_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    path.chmod(0o600)


def load_binding(binding_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    binding_root = Path(binding_root).resolve()
    manifest = _load_json(binding_root / "binding_manifest.json")
    if manifest.get("schema_version") != "event_graph_target_binding_v1" or manifest.get("status") != "completed":
        raise ValueError("Only a completed event-graph target binding may be consumed")
    targets = binding_root / "targets.csv"
    if manifest.get("artifact_files", {}).get("targets.csv") != sha256(targets):
        raise ValueError("Target binding checksum mismatch")
    import csv
    rows: dict[str, dict[str, Any]] = {}
    with targets.open(newline="") as stream:
        for row in csv.DictReader(stream):
            sample_id = row.get("sample_id", "")
            if not sample_id or sample_id in rows:
                raise ValueError("Empty or duplicate target sample identity")
            target = int(row["target"])
            rows[sample_id] = {
                "sample_id": sample_id,
                "subject_id": row["subject_id"],
                "stay_id": row["stay_id"],
                "split": row["split"],
                "target": target,
                "label": row["label"],
            }
    return rows, manifest


def _event_key(node: dict[str, Any]) -> tuple[str, str | None]:
    unit = node.get("unit")
    if unit is not None and not isinstance(unit, str):
        raise ValueError("Event unit must be a string or null")
    return str(node["token"]), unit


def _key_text(key: tuple[str, str | None]) -> str:
    return f"{key[0]}|unit={key[1] if key[1] is not None else '<null>'}"


def iter_graphs(graph_root: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    path = Path(graph_root) / "graphs.jsonl"
    with path.open() as stream:
        for ordinal, line in enumerate(stream):
            graph = json.loads(line)
            if not isinstance(graph, dict):
                raise ValueError("Graph JSONL record must be an object")
            yield ordinal, graph


def _fit_vocab(graph_root: Path, train_ids: set[str]) -> dict[str, Any]:
    concepts: set[str] = set()
    events: set[tuple[str, str | None]] = set()
    knowledge: set[str] = set()
    relations: set[str] = set()
    train_seen: set[str] = set()
    for _, graph in iter_graphs(graph_root):
        if graph.get("sample_id") not in train_ids:
            continue
        train_seen.add(graph["sample_id"])
        for node in graph["nodes"]:
            kind = node["kind"]
            if kind == "concept":
                concepts.add(node["token"])
            elif kind == "event":
                events.add(_event_key(node))
            elif kind == "knowledge":
                knowledge.add(node["token"])
        for edge in graph["edges"]:
            relation = edge["relation"]
            relations.add(relation)
            relations.add("inv:" + relation)
    if train_seen != train_ids:
        missing = sorted(train_ids - train_seen)[:5]
        raise ValueError(f"Target-bound train graphs missing from JSONL: {missing}")
    return {
        "concepts": sorted(concepts),
        "events": sorted(events, key=lambda key: (key[0], key[1] is not None, key[1] or "")),
        "knowledge": sorted(knowledge),
        "relations": sorted(relations),
    }


def _feature_names(vocab: dict[str, Any]) -> list[str]:
    names = [f"concept_present::{token}" for token in vocab["concepts"]]
    for key in vocab["events"]:
        text = _key_text(tuple(key))
        names.extend(f"event::{text}::{stat}" for stat in EVENT_STAT_ORDER)
    names.extend(f"knowledge_present::{token}" for token in vocab["knowledge"])
    names.extend(f"node_count::{kind}" for kind in KIND_ORDER)
    for relation in vocab["relations"]:
        names.extend(f"edge::{relation}::{stat}" for stat in EDGE_STAT_ORDER)
    names.extend(("coverage::events", "coverage::prior_visits", "coverage::knowledge_edges",
                  "coverage::index_event_count", "coverage::prior_event_count"))
    return names


def _encode_graph(graph: dict[str, Any], vocab: dict[str, Any],
                  adapter: EventGraphTensorizer) -> np.ndarray:
    """Project the same transformed PyG node/edge channels used by EventGCHM."""
    values: dict[str, float] = {}
    nodes = graph["nodes"]
    data = adapter.transform(graph)
    by_id = {node["id"]: node for node in nodes}
    event_visit: dict[str, str] = {}
    for edge in graph["edges"]:
        if edge["relation"] == "contains_event":
            source = by_id[edge["source"]]
            target = by_id[edge["target"]]
            if source["kind"] == "visit" and target["kind"] == "event":
                event_visit[target["id"]] = source["token"]

    for token in vocab["concepts"]:
        values[f"concept_present::{token}"] = 0.0
    for key in vocab["events"]:
        prefix = f"event::{_key_text(tuple(key))}"
        for stat in EVENT_STAT_ORDER:
            values[f"{prefix}::{stat}"] = 0.0
    for token in vocab["knowledge"]:
        values[f"knowledge_present::{token}"] = 0.0
    for kind in KIND_ORDER:
        values[f"node_count::{kind}"] = 0.0
    for relation in vocab["relations"]:
        for stat in EDGE_STAT_ORDER:
            values[f"edge::{relation}::{stat}"] = 0.0

    event_counts: dict[tuple[str, str | None], int] = defaultdict(int)
    event_values: dict[tuple[str, str | None], list[float]] = defaultdict(list)
    event_times: dict[tuple[str, str | None], list[float]] = defaultdict(list)
    event_available: dict[tuple[str, str | None], list[float]] = defaultdict(list)
    index_event_count = 0
    prior_event_count = 0
    for index, node in enumerate(nodes):
        kind = node["kind"]
        values[f"node_count::{kind}"] += 1.0
        if kind == "concept" and node["token"] in vocab["concepts"]:
            values[f"concept_present::{node['token']}"] = 1.0
        elif kind == "knowledge" and node["token"] in vocab["knowledge"]:
            values[f"knowledge_present::{node['token']}"] = 1.0
        elif kind == "event":
            key = _event_key(node)
            if key in vocab["events"]:
                event_counts[key] += 1
                tensor_row = data.x[index]
                if bool(tensor_row[1]):
                    event_values[key].append(float(tensor_row[0]))
                if bool(tensor_row[3]):
                    event_times[key].append(float(tensor_row[2]))
                if bool(tensor_row[5]):
                    event_available[key].append(float(tensor_row[4]))
            visit_token = event_visit.get(node["id"], "")
            if visit_token == "visit:index":
                index_event_count += 1
            elif visit_token == "visit:prior":
                prior_event_count += 1

    relation_by_id = {identifier: relation for relation, identifier in adapter.relation_to_id.items()}
    edge_deltas: dict[str, list[float]] = defaultdict(list)
    edge_has_delta: dict[str, int] = defaultdict(int)
    for index, relation_id in enumerate(data.edge_type.tolist()):
        relation = relation_by_id.get(relation_id)
        if relation is None or relation not in vocab["relations"]:
            continue
        prefix = f"edge::{relation}"
        values[f"{prefix}::count"] += 1.0
        edge_attr = data.edge_attr[index]
        if bool(edge_attr[1]):
            edge_deltas[relation].append(float(edge_attr[0]))
            edge_has_delta[relation] += 1
    for key in vocab["events"]:
        key = tuple(key)
        prefix = f"event::{_key_text(key)}"
        observations = event_values.get(key, [])
        times = event_times.get(key, [])
        available = event_available.get(key, [])
        values[f"{prefix}::count"] = float(event_counts.get(key, 0))
        values[f"{prefix}::observed_count"] = float(len(observations))
        if observations:
            arr = np.asarray(observations, dtype=np.float64)
            values[f"{prefix}::value_mean"] = float(arr.mean())
            values[f"{prefix}::value_std"] = float(arr.std())
            values[f"{prefix}::value_min"] = float(arr.min())
            values[f"{prefix}::value_max"] = float(arr.max())
        if times:
            values[f"{prefix}::time_mean"] = float(np.mean(times))
        if available:
            values[f"{prefix}::available_mean"] = float(np.mean(available))
    for relation in vocab["relations"]:
        prefix = f"edge::{relation}"
        deltas = edge_deltas.get(relation, [])
        values[f"{prefix}::has_delta_count"] = float(edge_has_delta.get(relation, 0))
        if deltas:
            arr = np.asarray(deltas, dtype=np.float64)
            values[f"{prefix}::delta_mean"] = float(arr.mean())
            values[f"{prefix}::delta_min"] = float(arr.min())
            values[f"{prefix}::delta_max"] = float(arr.max())
    coverage = graph.get("coverage", {})
    values["coverage::events"] = float(coverage.get("events", 0))
    values["coverage::prior_visits"] = float(coverage.get("prior_visits", 0))
    values["coverage::knowledge_edges"] = float(coverage.get("knowledge_edges", 0))
    values["coverage::index_event_count"] = float(index_event_count)
    values["coverage::prior_event_count"] = float(prior_event_count)
    names = _feature_names(vocab)
    row = np.asarray([values[name] for name in names], dtype=np.float32)
    if not np.isfinite(row).all():
        raise ValueError("Nonfinite graph-level feature")
    return row


def build_feature_matrix(
    graph_root: Path = DEFAULT_GRAPH_ROOT,
    binding_root: Path | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    """Materialize XGBoost features using only target-bound graphs."""
    graph_root = Path(graph_root).resolve()
    if binding_root is None:
        binding_root = graph_root.parent / "first_recorded_lab_all_visits_v2_targets_v1"
    binding_root = Path(binding_root).resolve()
    if output is None:
        output = graph_root.parent / "first_recorded_lab_all_visits_v2_features_v1"
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite feature artifact: {output}")
    binding, binding_manifest = load_binding(binding_root)
    graph_manifest = _load_json(graph_root / "manifest.json")
    train_ids = {sample for sample, row in binding.items()
                 if row["split"] == "train" and row["target"] >= 0}
    vocab = _fit_vocab(graph_root, train_ids)
    adapter = EventGraphTensorizer(add_reverse_edges=True)
    adapter.fit(
        graph for _, graph in iter_graphs(graph_root)
        if graph.get("sample_id") in train_ids
    )
    names = _feature_names(vocab)
    rows: list[np.ndarray] = []
    targets: list[int] = []
    folds: list[int] = []
    ordinals: list[int] = []
    sample_ids: list[str] = []
    subjects: list[str] = []
    fold_id = {"train": 0, "validation": 1, "test": 2}
    for ordinal, graph in iter_graphs(graph_root):
        sample_id = graph["sample_id"]
        row = binding.get(sample_id)
        if row is None or row["target"] < 0:
            continue
        if row["split"] != graph["split"]:
            raise ValueError("Graph split differs from target binding")
        rows.append(_encode_graph(graph, vocab, adapter))
        targets.append(row["target"])
        folds.append(fold_id[row["split"]])
        ordinals.append(ordinal)
        sample_ids.append(sample_id)
        subjects.append(row["subject_id"])
    if not rows:
        raise ValueError("No labelled graph features were materialized")
    X = np.stack(rows).astype(np.float32, copy=False)
    y = np.asarray(targets, dtype=np.int64)
    fold_array = np.asarray(folds, dtype=np.int8)
    if len(names) != X.shape[1] or not np.isfinite(X).all():
        raise ValueError("Feature matrix shape or finiteness mismatch")
    for fold in (0, 1, 2):
        selected_subjects = {subjects[i] for i in range(len(subjects)) if fold_array[i] == fold}
        other = {subjects[i] for i in range(len(subjects)) if fold_array[i] != fold}
        if selected_subjects & other:
            raise ValueError("A labelled subject crosses folds")

    output.mkdir(parents=True)
    np.savez_compressed(output / "features.npz", X=X, y=y, folds=fold_array, ordinals=np.asarray(ordinals, dtype=np.int64))
    (output / "feature_names.json").write_text(json.dumps(names, indent=2) + "\n")
    (output / "sample_metadata.json").write_text(json.dumps({"sample_ids": sample_ids, "subjects": subjects}, indent=2) + "\n")
    (output / "feature_names.json").chmod(0o600)
    (output / "sample_metadata.json").chmod(0o600)
    metadata = {
        "schema_version": "event_graph_xgb_features_v1",
        "status": "completed",
        "graph_root": str(graph_root),
        "graph_sha256": graph_manifest["graphs_sha256"],
        "binding_root": str(binding_root),
        "binding_manifest_sha256": sha256(binding_root / "binding_manifest.json"),
        "binding_targets_sha256": sha256(binding_root / "targets.csv"),
        "fit_scope": "labelled training graphs only",
        "tensorizer": adapter.to_dict(),
        "counts": {
            "rows": int(len(y)),
            "features": int(X.shape[1]),
            "train": int((fold_array == 0).sum()),
            "validation": int((fold_array == 1).sum()),
            "test": int((fold_array == 2).sum()),
        },
        "feature_layout": {
            "concept_presence": len(vocab["concepts"]),
            "event_keys": len(vocab["events"]),
            "event_stats_per_key": len(EVENT_STAT_ORDER),
            "knowledge_presence": len(vocab["knowledge"]),
            "node_counts": len(KIND_ORDER),
            "edge_stats_per_relation": len(EDGE_STAT_ORDER),
            "edge_relations": len(vocab["relations"]),
            "coverage": 5,
        },
        "vocabulary": vocab,
        "artifact_files": {},
        "test_evaluated": False,
        "temporal_clean": False,
    }
    for name in ("features.npz", "feature_names.json", "sample_metadata.json"):
        metadata["artifact_files"][name] = sha256(output / name)
    _json_save(output / "features_manifest.json", metadata)
    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument("--binding-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_feature_matrix(args.graph_root, args.binding_root, args.output), indent=2))
