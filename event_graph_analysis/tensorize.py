"""Train-only preprocessing for the ``event_graph_v1`` JSON contract.

Identifiers are used solely to resolve graph-local edges and are never retained
in model inputs or fitted state. Tokens must be clinical/category tokens supplied
by the builder, not patient or encounter identifiers. Upstream construction owns
prediction-time censoring and provenance validation; this adapter cannot infer a
prediction cutoff from relative timestamps alone.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from numbers import Real
from typing import Any

import torch
from torch_geometric.data import Data

SCHEMA_VERSION = "event_graph_v1"
NODE_KINDS = ("patient", "visit", "event", "concept", "knowledge")
KIND_TO_ID = {kind: index for index, kind in enumerate(NODE_KINDS)}
NODE_FEATURES = (
    "scaled_value", "has_value", "time_signed_log1p", "has_time",
    "available_signed_log1p", "has_available",
)
EDGE_FEATURES = ("delta_signed_log1p", "has_delta")


def _number(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number or null")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field} is outside supported numeric range") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _signed_log(value: float | None) -> float:
    return 0.0 if value is None else math.copysign(math.log1p(abs(value)), value)


def _key(node: Mapping[str, Any]) -> tuple[str, str | None]:
    return node["token"], node["unit"]


def _key_order(key: tuple[str, str | None]) -> tuple[str, bool, str]:
    return key[0], key[1] is not None, key[1] or ""


def _validate_graph(graph: Mapping[str, Any]) -> None:
    """Validate structure and covariates without consulting the target."""
    if not isinstance(graph, Mapping) or graph.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version={SCHEMA_VERSION!r}")
    _text(graph.get("sample_id"), "sample_id")
    if graph.get("split") not in ("train", "validation", "test"):
        raise ValueError("split must be train, validation or test")
    nodes, edges = graph.get("nodes"), graph.get("edges")
    if not isinstance(nodes, list) or not nodes or not isinstance(edges, list):
        raise ValueError("nodes must be a nonempty list and edges must be a list")
    ids: set[str] = set()
    patients = 0
    for node in nodes:
        if not isinstance(node, Mapping):
            raise ValueError("Each node must be an object")
        node_id = _text(node.get("id"), "node.id")
        if node_id in ids:
            raise ValueError("Duplicate graph-local node id")
        ids.add(node_id)
        if node.get("kind") not in KIND_TO_ID:
            raise ValueError("Unknown node kind")
        patients += node["kind"] == "patient"
        _text(node.get("token"), "node.token")
        if "unit" not in node or (node["unit"] is not None and not isinstance(node["unit"], str)):
            raise ValueError("node.unit must be a string or null")
        for field in ("value", "time_hours", "available_hours"):
            if field not in node:
                raise ValueError(f"Missing node.{field}")
            _number(node[field], f"node.{field}")
    if patients != 1:
        raise ValueError("Each graph must contain exactly one patient node")
    for edge in edges:
        if not isinstance(edge, Mapping):
            raise ValueError("Each edge must be an object")
        source = _text(edge.get("source"), "edge.source")
        target = _text(edge.get("target"), "edge.target")
        if source not in ids or target not in ids:
            raise ValueError("Edge endpoint missing from nodes")
        relation = _text(edge.get("relation"), "edge.relation")
        if relation.startswith("inv:"):
            raise ValueError("inv: is reserved for adapter-generated reverse relations")
        if "delta_hours" not in edge:
            raise ValueError("Missing edge.delta_hours")
        _number(edge["delta_hours"], "edge.delta_hours")


class EventGraphTensorizer:
    """Fit on training graphs, then transform any split into standard PyG Data.

    ``fit(train_graphs)`` consumes even a one-shot iterable exactly once and
    returns self. It rejects non-training graphs, ignores targets entirely, and
    replaces state atomically only after successful completion. No input graph
    is retained. To transform the same one-shot source later, reopen the source.

    Token vocabulary keys are exact ``(token, unit)`` pairs, preserving null,
    empty and distinct units without conversions or string normalization. ID 0
    is reserved for unknown pairs/relations; known IDs are sorted deterministically.
    Numeric statistics are population mean/std for each exact pair, using online
    Welford accumulation. Constant/singleton scales use 1. Unseen numeric scales
    yield scaled_value=0 (never borrow another unit's scale), while has_value
    still reports observedness. Known values are z-scored and clipped to
    ``numeric_clip``. Times/deltas use signed log1p without fitted statistics.

    ``to_dict()`` produces JSON-safe fitted state; persist with ``json.dump`` and
    restore via ``EventGraphTensorizer.from_dict(json.load(...))``. Persist it
    alongside model weights: embedding dimensions and vocabulary must match.
    """

    def __init__(self, *, add_reverse_edges: bool = True, numeric_clip: float = 10.0):
        if not isinstance(add_reverse_edges, bool):
            raise ValueError("add_reverse_edges must be bool")
        clip = _number(numeric_clip, "numeric_clip")
        if clip is None or not 0 < clip <= 1e6:
            raise ValueError("numeric_clip must be in (0, 1e6]")
        self.add_reverse_edges = add_reverse_edges
        self.numeric_clip = clip
        self.token_to_id: dict[tuple[str, str | None], int] = {}
        self.relation_to_id: dict[str, int] = {}
        self.numeric_stats: dict[tuple[str, str | None], dict[str, Any]] = {}
        self._fitted = False

    @property
    def num_tokens(self) -> int:
        """Embedding size, including unknown ID 0."""
        self._require_fitted()
        return len(self.token_to_id) + 1

    @property
    def num_relations(self) -> int:
        """Embedding size, including unknown ID 0 and fitted reverse relations."""
        self._require_fitted()
        return len(self.relation_to_id) + 1

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit(training_graphs) or from_dict(state) first")

    def fit(self, graphs: Iterable[Mapping[str, Any]]) -> EventGraphTensorizer:
        """Learn covariate-only state in one pass; reject empty/mixed-split inputs."""
        tokens: set[tuple[str, str | None]] = set()
        relations: set[str] = set()
        accum: dict[tuple[str, str | None], tuple[int, float, float]] = {}
        graph_count = 0
        for graph in graphs:
            _validate_graph(graph)
            if graph["split"] != "train":
                raise ValueError("fit accepts training graphs only")
            graph_count += 1
            for node in graph["nodes"]:
                key = _key(node)
                tokens.add(key)
                value = _number(node["value"], "node.value")
                if value is None:
                    continue
                count, mean, m2 = accum.get(key, (0, 0.0, 0.0))
                count += 1
                delta = value - mean
                mean += delta / count
                m2 += delta * (value - mean)
                if not math.isfinite(mean) or not math.isfinite(m2):
                    raise ValueError("Numeric values overflowed scaler accumulation")
                accum[key] = count, mean, m2
            for edge in graph["edges"]:
                relation = edge["relation"]
                relations.add(relation)
                if self.add_reverse_edges:
                    relations.add("inv:" + relation)
        if graph_count == 0:
            raise ValueError("Cannot fit an empty training iterable")
        stats = {}
        for key in sorted(accum, key=_key_order):
            count, mean, m2 = accum[key]
            std = math.sqrt(max(m2 / count, 0.0))
            stats[key] = {"count": count, "mean": mean, "scale": std if std > 0 else 1.0}
        self.token_to_id = {key: i + 1 for i, key in enumerate(sorted(tokens, key=_key_order))}
        self.relation_to_id = {key: i + 1 for i, key in enumerate(sorted(relations))}
        self.numeric_stats = stats
        self._fitted = True
        return self

    def transform(self, graph: Mapping[str, Any]) -> Data:
        """Tensorize one graph without updating state; optional y is a long [1].

        All covariates are float32, all indices are int64. Reverse edge deltas
        negate the original source-to-target elapsed time; missing stays missing.
        Original edges (including duplicates/self-loops) are preserved, with one
        additional reversed edge per original when enabled. IDs and provenance
        are not attached to Data. Batch labelled and unlabelled graphs separately
        because standard PyG collation requires matching attribute sets.
        """
        self._require_fitted()
        _validate_graph(graph)
        nodes = graph["nodes"]
        indices = {node["id"]: i for i, node in enumerate(nodes)}
        features, token_ids, kinds, patient_mask = [], [], [], []
        for node in nodes:
            key = _key(node)
            value = _number(node["value"], "node.value")
            stats = self.numeric_stats.get(key)
            scaled = 0.0
            if value is not None and stats is not None:
                scaled = (value - stats["mean"]) / stats["scale"]
                scaled = max(-self.numeric_clip, min(self.numeric_clip, scaled))
            time = _number(node["time_hours"], "node.time_hours")
            available = _number(node["available_hours"], "node.available_hours")
            features.append([scaled, float(value is not None), _signed_log(time),
                             float(time is not None), _signed_log(available),
                             float(available is not None)])
            token_ids.append(self.token_to_id.get(key, 0))
            kinds.append(KIND_TO_ID[node["kind"]])
            patient_mask.append(node["kind"] == "patient")
        endpoints, types, attrs = [], [], []
        for edge in graph["edges"]:
            source, target = indices[edge["source"]], indices[edge["target"]]
            relation = edge["relation"]
            delta = _number(edge["delta_hours"], "edge.delta_hours")
            endpoints.append([source, target])
            types.append(self.relation_to_id.get(relation, 0))
            attrs.append([_signed_log(delta), float(delta is not None)])
            if self.add_reverse_edges:
                endpoints.append([target, source])
                types.append(self.relation_to_id.get("inv:" + relation, 0))
                attrs.append([-_signed_log(delta), float(delta is not None)])
        data = Data(
            x=torch.tensor(features, dtype=torch.float32),
            token_id=torch.tensor(token_ids, dtype=torch.long),
            kind_id=torch.tensor(kinds, dtype=torch.long),
            edge_index=torch.tensor(endpoints, dtype=torch.long).reshape(-1, 2).t().contiguous(),
            edge_type=torch.tensor(types, dtype=torch.long),
            edge_attr=torch.tensor(attrs, dtype=torch.float32).reshape(-1, 2),
            patient_mask=torch.tensor(patient_mask, dtype=torch.bool),
            num_nodes=len(nodes),
        )
        if "target" in graph:
            target = graph["target"]
            if type(target) is not int or not 0 <= target <= torch.iinfo(torch.long).max:
                raise ValueError("target must be a nonnegative int64 class index")
            data.y = torch.tensor([target], dtype=torch.long)
        return data

    def transform_many(self, graphs: Iterable[Mapping[str, Any]]) -> Iterable[Data]:
        """Yield transformed graphs lazily; this never fits or caches graphs."""
        self._require_fitted()
        for graph in graphs:
            yield self.transform(graph)

    def to_dict(self) -> dict[str, Any]:
        """Return independent, JSON-safe state (no graphs, IDs or labels)."""
        self._require_fitted()
        return {
            "state_version": 1, "schema_version": SCHEMA_VERSION,
            "node_kinds": list(NODE_KINDS), "node_features": list(NODE_FEATURES),
            "edge_features": list(EDGE_FEATURES),
            "add_reverse_edges": self.add_reverse_edges, "numeric_clip": self.numeric_clip,
            "tokens": [{"token": token, "unit": unit} for token, unit in
                       sorted(self.token_to_id, key=self.token_to_id.__getitem__)],
            "relations": sorted(self.relation_to_id, key=self.relation_to_id.__getitem__),
            "numeric_stats": [{"token": token, "unit": unit, **dict(stats)}
                              for (token, unit), stats in sorted(
                                  self.numeric_stats.items(), key=lambda item: _key_order(item[0]))],
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> EventGraphTensorizer:
        """Restore a versioned JSON state; reject inconsistent vocabularies/scales."""
        if not isinstance(state, Mapping) or state.get("state_version") != 1:
            raise ValueError("Unsupported tensorizer state version")
        expected = {"schema_version": SCHEMA_VERSION, "node_kinds": list(NODE_KINDS),
                    "node_features": list(NODE_FEATURES), "edge_features": list(EDGE_FEATURES)}
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("Tensorizer schema or feature layout mismatch")
        required = ("add_reverse_edges", "numeric_clip", "tokens", "relations", "numeric_stats")
        if any(key not in state for key in required):
            raise ValueError("Incomplete tensorizer state")
        obj = cls(add_reverse_edges=state["add_reverse_edges"], numeric_clip=state["numeric_clip"])
        if any(not isinstance(state[key], list) for key in ("tokens", "relations", "numeric_stats")):
            raise ValueError("State vocabularies and statistics must be lists")

        def pair(record: Any) -> tuple[str, str | None]:
            if not isinstance(record, Mapping) or "unit" not in record:
                raise ValueError("Invalid token/unit state record")
            token = _text(record.get("token"), "state.token")
            unit = record["unit"]
            if unit is not None and not isinstance(unit, str):
                raise ValueError("Invalid unit in state")
            return token, unit

        keys = [pair(record) for record in state["tokens"]]
        if not keys or keys != sorted(set(keys), key=_key_order):
            raise ValueError("Token vocabulary must be nonempty, unique and sorted")
        relations = [_text(r, "state.relation") for r in state["relations"]]
        if relations != sorted(set(relations)):
            raise ValueError("Relation vocabulary must be unique and sorted")
        base = {r for r in relations if not r.startswith("inv:")}
        expected_relations = base | ({"inv:" + r for r in base} if obj.add_reverse_edges else set())
        if set(relations) != expected_relations:
            raise ValueError("Reverse relation vocabulary does not match configuration")
        obj.token_to_id = {key: i + 1 for i, key in enumerate(keys)}
        obj.relation_to_id = {r: i + 1 for i, r in enumerate(relations)}
        for record in state["numeric_stats"]:
            key = pair(record)
            if key not in obj.token_to_id or key in obj.numeric_stats:
                raise ValueError("Unknown or duplicate numeric scale key")
            count = record.get("count")
            mean = _number(record.get("mean"), "state.mean")
            scale = _number(record.get("scale"), "state.scale")
            if type(count) is not int or count < 1 or mean is None or scale is None or scale <= 0:
                raise ValueError("Invalid numeric scale statistics")
            obj.numeric_stats[key] = {"count": count, "mean": mean, "scale": scale}
        obj._fitted = True
        return obj
