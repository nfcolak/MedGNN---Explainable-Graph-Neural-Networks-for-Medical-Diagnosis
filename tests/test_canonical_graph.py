"""Focused tests for canonical benchmark graph topology primitives."""
from dataclasses import FrozenInstanceError
import hashlib
import json

import pytest

from shared.lib.canonical_graph import (
    CanonicalEdge,
    CanonicalGraph,
    CanonicalNode,
    build_edges,
    graph_fingerprint,
)


PATIENT = "p"
CONCEPTS = ("a", "b", "c")
NODE_IDS = (PATIENT, *CONCEPTS)


def test_canonical_node_is_a_frozen_identity_and_type_pair():
    node = CanonicalNode("a", "medication")

    assert node.node_id == "a"
    assert node.node_type == "medication"
    with pytest.raises(FrozenInstanceError):
        node.node_type = "diagnosis"


@pytest.mark.parametrize(
    ("node_id", "node_type"),
    [("", "medication"), ("a", ""), (1, "medication"), ("a", None)],
)
def test_canonical_node_requires_nonempty_string_fields(node_id, node_type):
    with pytest.raises(ValueError, match="non-empty string"):
        CanonicalNode(node_id, node_type)


def test_canonical_graph_normalizes_node_and_edge_order_and_duplicates():
    nodes = (
        CanonicalNode("p", "patient"),
        CanonicalNode("b", "lab"),
        CanonicalNode("a", "medication"),
        CanonicalNode("a", "medication"),
    )
    edges = (
        CanonicalEdge("p", "b", "patient_concept"),
        CanonicalEdge("a", "p", "patient_concept"),
        CanonicalEdge("p", "a", "patient_concept"),
        CanonicalEdge("p", "b", "patient_concept"),
        CanonicalEdge("b", "p", "patient_concept"),
    )

    graph = CanonicalGraph("subject-1", 7, nodes, edges)

    assert graph.nodes == tuple(sorted(set(nodes)))
    assert graph.edges == tuple(sorted(set(edges)))
    with pytest.raises(FrozenInstanceError):
        graph.class_id = 8


def test_canonical_graph_materializes_one_shot_node_and_edge_iterables():
    nodes = (
        CanonicalNode("p", "patient"),
        CanonicalNode("a", "medication"),
    )
    edges = (
        CanonicalEdge("p", "a", "patient_concept"),
        CanonicalEdge("a", "p", "patient_concept"),
    )

    graph = CanonicalGraph(
        "subject-1",
        7,
        (node for node in reversed(nodes)),
        (edge for edge in reversed(edges)),
    )

    assert graph.nodes == tuple(sorted(nodes))
    assert graph.edges == tuple(sorted(edges))


@pytest.mark.parametrize("subject_id", ["", None, 1])
def test_canonical_graph_requires_nonempty_string_subject_id(subject_id):
    with pytest.raises(ValueError, match="subject_id must be a non-empty string"):
        CanonicalGraph(subject_id, 7, (), ())


@pytest.mark.parametrize("class_id", [-1, 30, True, False, 7.0, "7", None])
def test_canonical_graph_requires_exact_class_integer_in_fixed_range(class_id):
    with pytest.raises(ValueError, match=r"class_id must be an exact int in 0\.\.29"):
        CanonicalGraph("subject-1", class_id, (), ())


@pytest.mark.parametrize("class_id", [0, 29])
def test_canonical_graph_accepts_class_range_boundaries(class_id):
    assert CanonicalGraph("subject-1", class_id, (), ()).class_id == class_id


def test_canonical_graph_rejects_conflicting_types_for_one_node_identity():
    with pytest.raises(ValueError, match="multiple types"):
        CanonicalGraph(
            "subject-1",
            7,
            (CanonicalNode("a", "lab"), CanonicalNode("a", "medication")),
            (),
        )


def test_canonical_graph_rejects_edges_to_nodes_outside_the_graph():
    with pytest.raises(ValueError, match="unknown node"):
        CanonicalGraph(
            "subject-1",
            7,
            (CanonicalNode("p", "patient"),),
            (CanonicalEdge("p", "external", "patient_concept"),),
        )


def test_canonical_edge_forbids_self_edges():
    with pytest.raises(ValueError, match="Self-edges"):
        CanonicalEdge("a", "a", "cooccur")


@pytest.mark.parametrize(
    ("source", "target", "edge_type"),
    [
        ("", "a", "cooccur"),
        ("a", "", "cooccur"),
        (1, "a", "cooccur"),
        ("a", "b", ""),
    ],
)
def test_canonical_edge_requires_nonempty_string_fields(source, target, edge_type):
    with pytest.raises(ValueError, match="non-empty string"):
        CanonicalEdge(source, target, edge_type)


def test_build_edges_rejects_unknown_or_expanded_structures():
    for structure in ("unknown", "full_kg_expanded"):
        with pytest.raises(ValueError, match="record-local structure"):
            build_edges(NODE_IDS, (), (), structure, patient_id=PATIENT)


def test_build_edges_requires_explicit_patient_id_argument():
    with pytest.raises(TypeError, match="patient_id"):
        build_edges(NODE_IDS, (), (), "star")  # pyright: ignore[reportCallIssue]


@pytest.mark.parametrize("patient_id", ["", None, 1])
def test_build_edges_rejects_malformed_explicit_patient_hub(patient_id):
    with pytest.raises(ValueError, match="patient_id must be a non-empty string"):
        build_edges(NODE_IDS, (), (), "star", patient_id=patient_id)


def test_build_edges_rejects_patient_hub_absent_from_nodes():
    with pytest.raises(ValueError, match="patient_id must identify a node"):
        build_edges(CONCEPTS, (), (), "star", patient_id=PATIENT)


@pytest.mark.parametrize("node_ids", [(1,), ("p", 1), (["nested"],)])
def test_build_edges_requires_nonempty_string_node_ids(node_ids):
    with pytest.raises(ValueError, match="non-empty string"):
        build_edges(node_ids, (), (), "star", patient_id=PATIENT)


def test_star_uses_explicit_hub_independent_of_concept_order():
    expected = build_edges(
        (PATIENT, *CONCEPTS), (), (), "star", patient_id=PATIENT
    )

    assert build_edges(
        ("c", PATIENT, "a", "b"), (), (), "star", patient_id=PATIENT
    ) == expected
    assert build_edges(
        set(NODE_IDS), (), (), "star", patient_id=PATIENT
    ) == expected


def test_star_contains_exactly_bidirectional_patient_concept_edges():
    assert build_edges(NODE_IDS, (), (), "star", patient_id=PATIENT) == (
        CanonicalEdge("a", "p", "patient_concept"),
        CanonicalEdge("b", "p", "patient_concept"),
        CanonicalEdge("c", "p", "patient_concept"),
        CanonicalEdge("p", "a", "patient_concept"),
        CanonicalEdge("p", "b", "patient_concept"),
        CanonicalEdge("p", "c", "patient_concept"),
    )


def test_cooccur_adds_only_selected_bidirectional_cooccurrence_pairs():
    edges = build_edges(
        NODE_IDS,
        cooccur_pairs=(("b", "a"),),
        ontology_pairs=(("b", "c"),),
        structure="cooccur",
        patient_id=PATIENT,
    )

    assert edges == tuple(
        sorted(
            {
                *build_edges(NODE_IDS, (), (), "star", patient_id=PATIENT),
                CanonicalEdge("a", "b", "cooccur"),
                CanonicalEdge("b", "a", "cooccur"),
            }
        )
    )


def test_ontology_adds_only_selected_bidirectional_ontology_pairs():
    edges = build_edges(
        NODE_IDS,
        cooccur_pairs=(("a", "b"),),
        ontology_pairs=(("c", "b"),),
        structure="ontology",
        patient_id=PATIENT,
    )

    assert edges == tuple(
        sorted(
            {
                *build_edges(NODE_IDS, (), (), "star", patient_id=PATIENT),
                CanonicalEdge("b", "c", "ontology"),
                CanonicalEdge("c", "b", "ontology"),
            }
        )
    )


def test_full_is_the_deduplicated_union_of_star_and_selected_pairs():
    edges = build_edges(
        NODE_IDS,
        cooccur_pairs=(("a", "b"), ("b", "a")),
        ontology_pairs=(("b", "c"), ("b", "c")),
        structure="full",
        patient_id=PATIENT,
    )

    assert edges == tuple(
        sorted(
            {
                *build_edges(NODE_IDS, (), (), "star", patient_id=PATIENT),
                CanonicalEdge("a", "b", "cooccur"),
                CanonicalEdge("b", "a", "cooccur"),
                CanonicalEdge("b", "c", "ontology"),
                CanonicalEdge("c", "b", "ontology"),
            }
        )
    )


def test_build_edges_rejects_string_as_a_malformed_relation_pair():
    with pytest.raises(ValueError, match="cooccur_pairs entries must be pairs"):
        build_edges(
            NODE_IDS,
            cooccur_pairs=("ab",),  # pyright: ignore[reportArgumentType]
            ontology_pairs=(),
            structure="cooccur",
            patient_id=PATIENT,
        )


@pytest.mark.parametrize(
    ("cooccur_pairs", "ontology_pairs", "field_name"),
    [
        ((("a",),), (), "cooccur_pairs"),
        ((), (("a", "b", "c"),), "ontology_pairs"),
        ((("a", 1),), (), "cooccur_pairs"),
        ((), ((None, "b"),), "ontology_pairs"),
    ],
)
def test_build_edges_rejects_malformed_relation_pairs(
    cooccur_pairs, ontology_pairs, field_name
):
    with pytest.raises(ValueError, match=field_name):
        build_edges(
            NODE_IDS,
            cooccur_pairs=cooccur_pairs,
            ontology_pairs=ontology_pairs,
            structure="full",
            patient_id=PATIENT,
        )


def test_build_edges_omits_record_external_relation_pairs():
    assert build_edges(
        NODE_IDS,
        cooccur_pairs=(("a", "external"),),
        ontology_pairs=(("external", "b"),),
        structure="full",
        patient_id=PATIENT,
    ) == build_edges(NODE_IDS, (), (), "star", patient_id=PATIENT)


def test_build_edges_omits_patient_containing_relation_pairs():
    assert build_edges(
        NODE_IDS,
        cooccur_pairs=((PATIENT, "a"),),
        ontology_pairs=(("b", PATIENT),),
        structure="full",
        patient_id=PATIENT,
    ) == build_edges(NODE_IDS, (), (), "star", patient_id=PATIENT)


def test_build_edges_omits_self_relation_pairs():
    assert build_edges(
        NODE_IDS,
        cooccur_pairs=(("a", "a"),),
        ontology_pairs=(("b", "b"),),
        structure="full",
        patient_id=PATIENT,
    ) == build_edges(NODE_IDS, (), (), "star", patient_id=PATIENT)


def test_build_edges_materializes_one_shot_node_and_relation_iterables():
    expected = build_edges(
        NODE_IDS,
        cooccur_pairs=(("a", "b"),),
        ontology_pairs=(("b", "c"),),
        structure="full",
        patient_id=PATIENT,
    )

    assert build_edges(
        (node_id for node_id in ("c", PATIENT, "b", "a")),
        cooccur_pairs=(pair for pair in (("a", "b"),)),
        ontology_pairs=(pair for pair in (("b", "c"),)),
        structure="full",
        patient_id=PATIENT,
    ) == expected


def test_fingerprint_is_order_independent_sha256_of_canonical_json():
    nodes = (
        CanonicalNode("p", "patient"),
        CanonicalNode("a", "medication"),
        CanonicalNode("b", "lab"),
    )
    edges = build_edges(("p", "a", "b"), (), (), "star", patient_id="p")
    reordered = CanonicalGraph(
        "subject-1", 7, tuple(reversed(nodes)), tuple(reversed(edges))
    )
    canonical = CanonicalGraph("subject-1", 7, nodes, edges)
    payload = {
        "class_id": 7,
        "edges": [
            {
                "edge_type": edge.edge_type,
                "source": edge.source,
                "target": edge.target,
            }
            for edge in sorted(edges)
        ],
        "nodes": [
            {"node_id": node.node_id, "node_type": node.node_type}
            for node in sorted(nodes)
        ],
        "subject_id": "subject-1",
    }
    expected = hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()

    assert graph_fingerprint(reordered) == expected
    assert graph_fingerprint(canonical) == expected


@pytest.mark.parametrize(
    ("field_name", "variant"),
    [
        (
            "subject_id",
            CanonicalGraph(
                "subject-2",
                7,
                (
                    CanonicalNode("p", "patient"),
                    CanonicalNode("a", "medication"),
                    CanonicalNode("b", "lab"),
                ),
                (CanonicalEdge("p", "a", "patient_concept"),),
            ),
        ),
        (
            "class_id",
            CanonicalGraph(
                "subject-1",
                8,
                (
                    CanonicalNode("p", "patient"),
                    CanonicalNode("a", "medication"),
                    CanonicalNode("b", "lab"),
                ),
                (CanonicalEdge("p", "a", "patient_concept"),),
            ),
        ),
        (
            "node_id",
            CanonicalGraph(
                "subject-1",
                7,
                (
                    CanonicalNode("p", "patient"),
                    CanonicalNode("a", "medication"),
                    CanonicalNode("c", "lab"),
                ),
                (CanonicalEdge("p", "a", "patient_concept"),),
            ),
        ),
        (
            "node_type",
            CanonicalGraph(
                "subject-1",
                7,
                (
                    CanonicalNode("p", "patient"),
                    CanonicalNode("a", "medication"),
                    CanonicalNode("b", "diagnosis"),
                ),
                (CanonicalEdge("p", "a", "patient_concept"),),
            ),
        ),
        (
            "edge_source",
            CanonicalGraph(
                "subject-1",
                7,
                (
                    CanonicalNode("p", "patient"),
                    CanonicalNode("a", "medication"),
                    CanonicalNode("b", "lab"),
                ),
                (CanonicalEdge("b", "a", "patient_concept"),),
            ),
        ),
        (
            "edge_target",
            CanonicalGraph(
                "subject-1",
                7,
                (
                    CanonicalNode("p", "patient"),
                    CanonicalNode("a", "medication"),
                    CanonicalNode("b", "lab"),
                ),
                (CanonicalEdge("p", "b", "patient_concept"),),
            ),
        ),
        (
            "edge_type",
            CanonicalGraph(
                "subject-1",
                7,
                (
                    CanonicalNode("p", "patient"),
                    CanonicalNode("a", "medication"),
                    CanonicalNode("b", "lab"),
                ),
                (CanonicalEdge("p", "a", "ontology"),),
            ),
        ),
    ],
)
def test_fingerprint_changes_when_any_canonical_field_changes(field_name, variant):
    baseline = CanonicalGraph(
        "subject-1",
        7,
        (
            CanonicalNode("p", "patient"),
            CanonicalNode("a", "medication"),
            CanonicalNode("b", "lab"),
        ),
        (CanonicalEdge("p", "a", "patient_concept"),),
    )

    assert graph_fingerprint(variant) != graph_fingerprint(baseline), field_name
