"""Method-independent graph identities and topology primitives."""
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from typing import Any, Iterable, Tuple


STANDARDIZED_GRAPH_MEMBERSHIP_CONTRACT = {
    "version": "patient-medication-chiefcomplaint-v1",
    "included_node_types": ["patient", "medication", "chiefcomplaint"],
    "excluded_node_types": ["vital", "diagnosis", "symptom"],
    "vital_policy": "exclude_nodes_no_categorical_proxy",
    "medication_column_prefixes": ["med_", "pyx_"],
    "chiefcomplaint_policy": "training_vocabulary",
}


def _require_nonempty_string(value: object, field_name: str) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"{field_name} must be a non-empty string.")


def _materialize_relation_pairs(
    pairs: Iterable[Tuple[str, str]], field_name: str
) -> Tuple[Tuple[str, str], ...]:
    normalized = []
    for pair in tuple(pairs):
        if isinstance(pair, (str, bytes)):
            raise ValueError(f"{field_name} entries must be pairs of node IDs.")
        try:
            endpoints = tuple(pair)
        except TypeError as error:
            raise ValueError(
                f"{field_name} entries must be pairs of node IDs."
            ) from error
        if len(endpoints) != 2:
            raise ValueError(f"{field_name} entries must be pairs of node IDs.")
        left, right = endpoints
        _require_nonempty_string(left, f"{field_name} endpoint")
        _require_nonempty_string(right, f"{field_name} endpoint")
        normalized.append((left, right))
    return tuple(normalized)


@dataclass(frozen=True, order=True)
class CanonicalNode:
    """One method-independent node identity and semantic type."""

    node_id: str
    node_type: str

    def __post_init__(self) -> None:
        _require_nonempty_string(self.node_id, "node_id")
        _require_nonempty_string(self.node_type, "node_type")


@dataclass(frozen=True, order=True)
class CanonicalEdge:
    """One typed directed edge between canonical node identities."""

    source: str
    target: str
    edge_type: str

    def __post_init__(self) -> None:
        _require_nonempty_string(self.source, "source")
        _require_nonempty_string(self.target, "target")
        _require_nonempty_string(self.edge_type, "edge_type")
        if self.source == self.target:
            raise ValueError(f"Self-edges are not canonical: {self.source!r}.")


@dataclass(frozen=True)
class CanonicalGraph:
    """Canonical subject graph independent of model tensor representation."""

    subject_id: str
    class_id: int
    nodes: Iterable[CanonicalNode]
    edges: Iterable[CanonicalEdge]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.subject_id, "subject_id")
        if type(self.class_id) is not int or not 0 <= self.class_id <= 29:
            raise ValueError("class_id must be an exact int in 0..29.")
        nodes = tuple(self.nodes)
        edges = tuple(self.edges)
        node_types = {}
        for node in nodes:
            prior_type = node_types.setdefault(node.node_id, node.node_type)
            if prior_type != node.node_type:
                raise ValueError(
                    f"Canonical node {node.node_id!r} has multiple types: "
                    f"{prior_type!r} and {node.node_type!r}."
                )
        unknown_ids = {
            endpoint
            for edge in edges
            for endpoint in (edge.source, edge.target)
            if endpoint not in node_types
        }
        if unknown_ids:
            raise ValueError(
                "Canonical edges reference unknown node identities: "
                + ", ".join(repr(node_id) for node_id in sorted(unknown_ids))
                + "."
            )
        object.__setattr__(self, "nodes", tuple(sorted(set(nodes))))
        object.__setattr__(self, "edges", tuple(sorted(set(edges))))


def canonical_graph_from_indexed(
    subject_id: str,
    class_id: int,
    node_ids: Iterable[str],
    node_types: Iterable[str],
    edge_index: Any,
    edge_types: Iterable[str],
) -> CanonicalGraph:
    """Normalize an indexed PyG/GraphCare record to canonical identities.

    Callers translate model-local node and relation IDs to the shared canonical
    names before calling. This keeps tensor layout method-specific while making
    node membership and typed topology directly comparable.
    """
    ids = tuple(node_ids)
    types = tuple(node_types)
    if len(ids) != len(types) or len(set(ids)) != len(ids):
        raise ValueError("Indexed graph requires one unique type for every node ID.")
    raw_index = edge_index.tolist() if hasattr(edge_index, "tolist") else edge_index
    try:
        rows = tuple(tuple(row) for row in raw_index)
    except TypeError as exc:
        raise ValueError("edge_index must be an iterable with shape [2, E].") from exc
    if len(rows) != 2 or len(rows[0]) != len(rows[1]):
        raise ValueError("edge_index must have shape [2, E].")
    canonical_edge_types = tuple(edge_types)
    if len(canonical_edge_types) != len(rows[0]):
        raise ValueError("edge_types must contain one canonical type per edge.")

    edges = []
    for source_index, target_index, edge_type in zip(
        rows[0], rows[1], canonical_edge_types
    ):
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, Integral)
            or isinstance(target_index, bool)
            or not isinstance(target_index, Integral)
            or not 0 <= int(source_index) < len(ids)
            or not 0 <= int(target_index) < len(ids)
        ):
            raise ValueError("edge_index contains an invalid local node index.")
        edges.append(
            CanonicalEdge(ids[int(source_index)], ids[int(target_index)], edge_type)
        )
    return CanonicalGraph(
        subject_id,
        class_id,
        (CanonicalNode(node_id, node_type) for node_id, node_type in zip(ids, types)),
        edges,
    )


def _record_value(record: Any, name: str) -> Any:
    if isinstance(record, Mapping):
        if name not in record:
            raise ValueError(f"Canonical record metadata is missing {name!r}.")
        return record[name]
    if not hasattr(record, name):
        raise ValueError(f"Canonical record metadata is missing {name!r}.")
    return getattr(record, name)


def _one_scalar(value: Any, name: str) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numel"):
        if int(value.numel()) != 1:
            raise ValueError(f"Canonical record metadata {name!r} must be scalar.")
        return value.reshape(-1)[0].item()
    return value


def canonical_graph_from_pyg_record(record: Any) -> CanonicalGraph:
    """Extract canonical identities and typed edges from a cached PyG record."""
    subject_id = _one_scalar(_record_value(record, "subject_id"), "subject_id")
    class_id = _one_scalar(_record_value(record, "y"), "y")
    if isinstance(subject_id, bool) or not isinstance(subject_id, (str, Integral)):
        raise ValueError("Canonical record metadata 'subject_id' is invalid.")
    if isinstance(class_id, bool) or not isinstance(class_id, Integral):
        raise ValueError("Canonical record metadata 'y' is invalid.")
    return canonical_graph_from_indexed(
        subject_id=str(subject_id),
        class_id=int(class_id),
        node_ids=_record_value(record, "canonical_node_ids"),
        node_types=_record_value(record, "canonical_node_types"),
        edge_index=_record_value(record, "edge_index"),
        edge_types=_record_value(record, "canonical_edge_types"),
    )


def canonical_graph_from_graphcare_record(
    record: Mapping[str, Any], kg: Mapping[str, Any]
) -> CanonicalGraph:
    """Extract a canonical graph from one production GraphCare dataset record."""
    if not isinstance(record, Mapping) or not isinstance(kg, Mapping):
        raise ValueError("Canonical GraphCare record and KG metadata must be mappings.")
    try:
        global_names = kg["canonical_node_ids_by_global_id"]
        global_types = kg["canonical_node_types_by_global_id"]
        relation_types = kg["canonical_edge_types_by_relation_id"]
        num_nodes = kg["num_nodes"]
        num_rels = kg["num_rels"]
    except KeyError as exc:
        raise ValueError(
            f"Canonical GraphCare KG metadata is missing {exc.args[0]!r}."
        ) from exc
    if (
        not isinstance(global_names, list)
        or not isinstance(global_types, list)
        or len(global_names) != num_nodes
        or len(global_types) != num_nodes
        or not isinstance(relation_types, list)
        or len(relation_types) != num_rels
    ):
        raise ValueError("Canonical GraphCare KG metadata has invalid indexed domains.")
    raw_global_ids = _record_value(record, "node_ids")
    global_ids = raw_global_ids.tolist() if hasattr(raw_global_ids, "tolist") else list(raw_global_ids)
    if any(
        isinstance(node_id, bool)
        or not isinstance(node_id, Integral)
        or not 0 <= int(node_id) < num_nodes
        for node_id in global_ids
    ):
        raise ValueError("Canonical GraphCare record contains an invalid global node ID.")
    raw_relation_ids = _record_value(record, "rel_ids")
    relation_ids = (
        raw_relation_ids.tolist()
        if hasattr(raw_relation_ids, "tolist")
        else list(raw_relation_ids)
    )
    if any(
        isinstance(relation_id, bool)
        or not isinstance(relation_id, Integral)
        or not 0 <= int(relation_id) < num_rels
        for relation_id in relation_ids
    ):
        raise ValueError("Canonical GraphCare record contains an invalid relation ID.")
    subject_id = _one_scalar(_record_value(record, "subject_id"), "subject_id")
    class_id = _one_scalar(_record_value(record, "y"), "y")
    if isinstance(subject_id, bool) or not isinstance(subject_id, (str, Integral)):
        raise ValueError("Canonical record metadata 'subject_id' is invalid.")
    if isinstance(class_id, bool) or not isinstance(class_id, Integral):
        raise ValueError("Canonical record metadata 'y' is invalid.")
    return canonical_graph_from_indexed(
        subject_id=str(subject_id),
        class_id=int(class_id),
        node_ids=[global_names[int(node_id)] for node_id in global_ids],
        node_types=[global_types[int(node_id)] for node_id in global_ids],
        edge_index=_record_value(record, "edge_index"),
        edge_types=[relation_types[int(relation_id)] for relation_id in relation_ids],
    )


def validate_cross_method_fingerprints(
    graphs_by_method: Mapping[str, CanonicalGraph],
) -> str:
    """Require one subject/topology record to match across at least two methods."""
    if not isinstance(graphs_by_method, Mapping) or len(graphs_by_method) < 2:
        raise ValueError("Cross-method validation requires at least two method graphs.")
    fingerprints = {}
    for method, graph in graphs_by_method.items():
        _require_nonempty_string(method, "method")
        if not isinstance(graph, CanonicalGraph):
            raise ValueError(f"Method {method!r} did not provide a CanonicalGraph.")
        fingerprints[method] = graph_fingerprint(graph)
    if len(set(fingerprints.values())) != 1:
        raise ValueError(
            "Methods produced different canonical node/edge fingerprints for the "
            "same subject/topology."
        )
    return next(iter(fingerprints.values()))


def build_edges(
    node_ids: Iterable[str],
    cooccur_pairs: Iterable[Tuple[str, str]],
    ontology_pairs: Iterable[Tuple[str, str]],
    structure: str,
    patient_id: str,
) -> Tuple[CanonicalEdge, ...]:
    """Build a deterministic record-local policy around an explicit patient hub."""
    if structure not in {"star", "cooccur", "ontology", "full"}:
        raise ValueError(
            f"Unknown record-local structure {structure!r}; "
            "choose star, cooccur, ontology, or full."
        )
    _require_nonempty_string(patient_id, "patient_id")
    raw_ids = tuple(node_ids)
    if not raw_ids:
        raise ValueError("node_ids must contain the explicit patient_id.")
    for node_id in raw_ids:
        _require_nonempty_string(node_id, "node_id")
    ordered_ids = tuple(dict.fromkeys(raw_ids))
    if patient_id not in ordered_ids:
        raise ValueError("patient_id must identify a node in node_ids.")
    concept_ids = set(ordered_ids) - {patient_id}
    edges = {
        CanonicalEdge(source, target, "patient_concept")
        for concept_id in concept_ids
        for source, target in ((patient_id, concept_id), (concept_id, patient_id))
    }
    if structure == "cooccur":
        pair_groups = (
            (_materialize_relation_pairs(cooccur_pairs, "cooccur_pairs"), "cooccur"),
        )
    elif structure == "ontology":
        pair_groups = (
            (_materialize_relation_pairs(ontology_pairs, "ontology_pairs"), "ontology"),
        )
    elif structure == "full":
        pair_groups = (
            (_materialize_relation_pairs(cooccur_pairs, "cooccur_pairs"), "cooccur"),
            (_materialize_relation_pairs(ontology_pairs, "ontology_pairs"), "ontology"),
        )
    else:
        pair_groups = ()
    for pairs, edge_type in pair_groups:
        edges.update(
            CanonicalEdge(source, target, edge_type)
            for left, right in pairs
            if left != right and left in concept_ids and right in concept_ids
            for source, target in ((left, right), (right, left))
        )
    return tuple(sorted(edges))


def graph_fingerprint(graph: CanonicalGraph) -> str:
    """Hash canonical graph metadata, identities, types, and typed topology."""
    payload = {
        "subject_id": graph.subject_id,
        "class_id": graph.class_id,
        "nodes": [
            {"node_id": node.node_id, "node_type": node.node_type}
            for node in sorted(graph.nodes)
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "edge_type": edge.edge_type,
            }
            for edge in sorted(graph.edges)
        ],
    }
    canonical_json = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
