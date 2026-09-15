"""Contract tests for the standardized GNN benchmark."""
from collections import Counter
from dataclasses import FrozenInstanceError
import json
from itertools import product
from pathlib import Path

import pytest

from shared.lib.benchmark_contract import (
    ALLOWED_SEEDS,
    EXPECTED_CLASSES,
    EXPECTED_FOLD_COUNTS,
    PRIMARY_STRUCTURES,
    BenchmarkSpec,
    file_sha256,
    load_canonical_split,
    validate_primary_matrix,
)
from shared.lib.graph_structures import (
    GRAPH_STRUCTURES,
    SUPPORTED_METHODS,
    all_methods,
    supported_structures,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_SPLIT = REPO_ROOT / "comparison" / "canonical_split.json"
CANONICAL_CLASSES = (
    "Acute kidney failure",
    "Acute pharyngitis",
    "Acute upper respiratory infection",
    "Alcohol abuse with intoxication",
    "Anemia",
    "Anxiety disorder",
    "Back or spine pain",
    "Cardiovascular risk factor",
    "Dehydration",
    "Diabetes mellitus",
    "End stage renal disease",
    "Epilepsy",
    "GI bleed",
    "Head injury",
    "Heart failure",
    "Hypokalemia",
    "Hypotension",
    "Hypothyroidism",
    "Laceration w/o fb of l idx fngr w/o damage to nail",
    "Limb injury or pain",
    "Lower respiratory disease",
    "Major depressive disorder",
    "Multiple fractures of ribs",
    "Non-ST elevation (NSTEMI) myocardial infarction",
    "Sepsis",
    "Skin or soft-tissue infection",
    "UTI or pyelonephritis",
    "Unsp intestnl obst",
    "Unspecified asthma with (acute) exacerbation",
    "Unspecified atrial fibrillation",
)


def test_registry_exposes_all_three_benchmark_methods():
    assert SUPPORTED_METHODS == ("protgnn", "gsat", "graphcare")
    assert all_methods() == list(SUPPORTED_METHODS)
    for method in SUPPORTED_METHODS:
        assert {"star", "cooccur", "ontology", "full"} <= set(
            supported_structures(method)
        )


def test_full_is_record_local_for_every_method():
    metadata = GRAPH_STRUCTURES["full"]
    assert metadata["record_local"] is True
    assert metadata["kg_expanded"] is False
    assert all(metadata[method] for method in SUPPORTED_METHODS)


def test_full_kg_expanded_is_graphcare_only():
    metadata = GRAPH_STRUCTURES["full_kg_expanded"]
    assert metadata["record_local"] is False
    assert metadata["kg_expanded"] is True
    assert metadata["protgnn"] is False
    assert metadata["gsat"] is False
    assert metadata["graphcare"] is True
    assert "full_kg_expanded" not in supported_structures("protgnn")
    assert "full_kg_expanded" not in supported_structures("gsat")
    assert "full_kg_expanded" in supported_structures("graphcare")


def test_benchmark_spec_is_frozen_and_accepts_allowed_values():
    spec = BenchmarkSpec("protgnn", "star", 1234)
    assert spec.method == "protgnn"
    assert spec.structure == "star"
    assert spec.seed == 1234
    with pytest.raises(FrozenInstanceError):
        spec.seed = 1235


@pytest.mark.parametrize(
    ("method", "structure", "seed", "message"),
    [
        ("unknown", "star", 1234, "method"),
        ("protgnn", "full_kg_expanded", 1234, "not supported"),
        ("gsat", "star", 1, "seed"),
    ],
)
def test_benchmark_spec_rejects_values_outside_the_contract(
    method, structure, seed, message
):
    with pytest.raises(ValueError, match=message):
        BenchmarkSpec(method, structure, seed)


def test_contract_uses_exact_primary_structures():
    assert PRIMARY_STRUCTURES == ("star", "cooccur")


def test_contract_uses_exact_allowed_seeds():
    assert ALLOWED_SEEDS == (1234, 1235, 1236)
    for method, structure, seed in product(
        SUPPORTED_METHODS, PRIMARY_STRUCTURES, ALLOWED_SEEDS
    ):
        assert BenchmarkSpec(method, structure, seed).seed == seed


def test_load_canonical_split_preserves_expected_fold_counts_and_classes():
    split = load_canonical_split(CANONICAL_SPLIT)
    assert Counter(split["fold"].values()) == EXPECTED_FOLD_COUNTS
    assert EXPECTED_CLASSES == CANONICAL_CLASSES
    assert tuple(split["classes"]) == CANONICAL_CLASSES


@pytest.mark.parametrize("mutation", ["reorder", "substitute"])
def test_load_canonical_split_rejects_changed_class_mapping(tmp_path, mutation):
    split = json.loads(CANONICAL_SPLIT.read_text(encoding="utf-8"))
    if mutation == "reorder":
        split["classes"][0], split["classes"][1] = (
            split["classes"][1],
            split["classes"][0],
        )
    else:
        split["classes"][0] = "Substituted diagnosis"

    path = tmp_path / "canonical_split.json"
    path.write_text(json.dumps(split), encoding="utf-8")

    with pytest.raises(ValueError, match="class identity and order"):
        load_canonical_split(path)


@pytest.mark.parametrize("invalid_fold", [False, 0.0, "0", None])
def test_load_canonical_split_rejects_noninteger_fold_ids(tmp_path, invalid_fold):
    split = json.loads(CANONICAL_SPLIT.read_text(encoding="utf-8"))
    subject = next(subject for subject, fold in split["fold"].items() if fold == 0)
    split["fold"][subject] = invalid_fold
    path = tmp_path / "canonical_split.json"
    path.write_text(json.dumps(split), encoding="utf-8")

    with pytest.raises(ValueError, match="integer identifiers"):
        load_canonical_split(path)


@pytest.mark.parametrize("invalid_class", ["", 7, False, None, ["nested"], {"x": 1}])
def test_load_canonical_split_rejects_nonstring_or_empty_classes(
    tmp_path, invalid_class
):
    split = json.loads(CANONICAL_SPLIT.read_text(encoding="utf-8"))
    split["classes"][0] = invalid_class
    path = tmp_path / "canonical_split.json"
    path.write_text(json.dumps(split), encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty string"):
        load_canonical_split(path)


def test_file_sha256_hashes_file_bytes(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"benchmark\n")
    assert file_sha256(path) == (
        "8f8dbecfd77ab2386b49d723c6b2474f2c22c246805fa0f677bbaf6e4f7bbbfe"
    )


def _primary_specs():
    return [
        BenchmarkSpec(method, structure, seed)
        for method, structure, seed in product(
            SUPPORTED_METHODS, PRIMARY_STRUCTURES, ALLOWED_SEEDS
        )
    ]


def test_validate_primary_matrix_accepts_exact_complete_matrix():
    specs = _primary_specs()
    assert len(specs) == 18
    assert validate_primary_matrix(specs) is None


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "secondary"])
def test_validate_primary_matrix_rejects_incomplete_or_nonprimary_matrix(mutation):
    specs = _primary_specs()
    if mutation == "missing":
        specs.pop()
    elif mutation == "duplicate":
        specs[-1] = specs[0]
    else:
        specs[-1] = BenchmarkSpec("graphcare", "ontology", 1236)

    with pytest.raises(ValueError, match="primary benchmark matrix"):
        validate_primary_matrix(specs)
