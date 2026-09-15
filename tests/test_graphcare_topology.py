"""Focused tests for GraphCare topology and preprocessing provenance."""
from copy import deepcopy
import json

import pandas as pd
import pytest
import torch

import graphcare_analysis.build_kg as graphcare_build_kg
from comparison.standardized import run_benchmark as benchmark_runner
from graphcare_analysis.adapter import GraphCareDataset, _subgraph, build_loaders
from graphcare_analysis.build_kg import (
    build_global_kg,
    validate_kg_schema,
    validate_training_provenance,
    vocab_and_presence,
)
from graphcare_analysis.run import _load_kg
from shared.lib import benchmark_contract
from shared.lib.benchmark_contract import file_sha256


def _write_tiny_canonical_split(path, monkeypatch, folds, classes):
    counts = {fold_id: list(folds.values()).count(fold_id) for fold_id in (0, 1, 2)}
    counts = {fold_id: count for fold_id, count in counts.items() if count}
    monkeypatch.setattr(benchmark_contract, "EXPECTED_FOLD_COUNTS", counts)
    monkeypatch.setattr(benchmark_contract, "EXPECTED_CLASSES", tuple(classes))
    monkeypatch.setattr(benchmark_contract, "EXPECTED_CLASS_COUNT", len(classes))
    path.write_text(
        json.dumps({"fold": folds, "classes": classes}), encoding="utf-8"
    )
    return path


def _global_edges(graph):
    node_ids = graph["node_ids"].tolist()
    return {
        (node_ids[source], node_ids[target], relation)
        for (source, target), relation in zip(
            graph["edge_index"].t().tolist(), graph["rel_ids"].tolist()
        )
    }


def test_star_adds_explicit_patient_hub_and_bidirectional_spokes():
    graph = _subgraph(
        code_ids=[2, 0],
        neighbours={0: [(0, 1)], 1: [(0, 0)], 2: []},
        num_nodes=4,
        structure="star",
        patient_hub_id=3,
        patient_relation_id=2,
    )

    assert graph["node_ids"].tolist() == [0, 2, 3]
    assert _global_edges(graph) == {
        (3, 0, 2),
        (0, 3, 2),
        (3, 2, 2),
        (2, 3, 2),
    }
    assert torch.equal(graph["visit_node"], torch.tensor([1.0, 0.0, 1.0, 1.0]))
    assert torch.equal(graph["ehr_nodes"], torch.tensor([1.0, 0.0, 1.0, 0.0]))


def test_cooccur_adds_only_record_local_pmi_edges_on_top_of_star():
    graph = _subgraph(
        code_ids=[0, 1],
        neighbours={0: [(0, 1), (1, 1)], 1: [(0, 0), (1, 0)]},
        num_nodes=3,
        structure="cooccur",
        patient_hub_id=2,
        patient_relation_id=2,
    )

    assert _global_edges(graph) == {
        (2, 0, 2),
        (0, 2, 2),
        (2, 1, 2),
        (1, 2, 2),
        (0, 1, 0),
        (1, 0, 0),
    }


def test_full_is_record_local_union_without_external_kg_nodes():
    graph = _subgraph(
        code_ids=[0, 2],
        neighbours={
            0: [(0, 2), (1, 2), (1, 3)],
            1: [],
            2: [(0, 0), (1, 0)],
            3: [(1, 0)],
        },
        num_nodes=5,
        structure="full",
        patient_hub_id=4,
        patient_relation_id=2,
    )

    assert graph["node_ids"].tolist() == [0, 2, 4]
    assert _global_edges(graph) == {
        (4, 0, 2),
        (0, 4, 2),
        (4, 2, 2),
        (2, 4, 2),
        (0, 2, 0),
        (2, 0, 0),
        (0, 2, 1),
        (2, 0, 1),
    }


def test_full_kg_expanded_preserves_record_external_one_hop_nodes():
    graph = _subgraph(
        code_ids=[0],
        neighbours={0: [(0, 1), (1, 2)], 1: [(0, 0)], 2: [(1, 0)]},
        num_nodes=4,
        structure="full_kg_expanded",
        patient_hub_id=3,
        patient_relation_id=2,
    )

    assert graph["node_ids"].tolist() == [0, 1, 2, 3]
    assert _global_edges(graph) == {
        (3, 0, 2),
        (0, 3, 2),
        (0, 1, 0),
        (0, 2, 1),
    }
    assert graph["ehr_nodes"].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_global_kg_reserves_patient_hub_entity_and_relation():
    df = pd.DataFrame(
        {
            "med_alpha": [1, 1],
            "symptom_1": ["pain", ""],
            "chiefcomplaint_1": ["", "fatigue"],
        }
    )

    kg = build_global_kg(save=False, df=df)

    hub_id = kg["ent2id"]["patient:hub"]
    relation_id = kg["rel2id"]["patient_has_concept"]
    assert hub_id == kg["num_nodes"] - 1
    assert relation_id == kg["num_rels"] - 1
    assert kg["neighbours"][hub_id] == []


def test_dataset_uses_reserved_hub_metadata_in_model_tensors():
    dataset = GraphCareDataset(
        code_lists=[[0]],
        labels=[7],
        neighbours={0: [], 1: []},
        num_nodes=2,
        structure="star",
        patient_hub_id=1,
        patient_relation_id=2,
    )

    graph = dataset[0]
    assert graph["node_ids"].tolist() == [0, 1]
    assert _global_edges(graph) == {(1, 0, 2), (0, 1, 2)}
    assert graph["visit_node"].shape == (2,)
    assert graph["ehr_nodes"].shape == (2,)
    assert graph["y"] == 7


@pytest.mark.parametrize(
    "fit_indices",
    [
        [0.5],
        [True],
        [0, 0],
        [-1],
        [2],
        [[0]],
    ],
)
def test_vocab_rejects_invalid_fit_indices(fit_indices):
    df = pd.DataFrame({"med_alpha": [1, 0]})

    with pytest.raises(ValueError, match="fit_indices"):
        vocab_and_presence(df, fit_indices=fit_indices)


def test_hub_only_subgraph_has_safe_empty_edge_tensor_shapes():
    graph = _subgraph(
        code_ids=[],
        neighbours={0: []},
        num_nodes=1,
        structure="star",
        patient_hub_id=0,
        patient_relation_id=2,
    )

    assert graph["node_ids"].tolist() == [0]
    assert graph["edge_index"].shape == (2, 0)
    assert graph["rel_ids"].shape == (0,)
    assert graph["visit_node"].tolist() == [1.0]
    assert graph["ehr_nodes"].tolist() == [0.0]


@pytest.mark.parametrize("structure", ["star", "cooccur"])
def test_standardized_loader_retains_zero_concept_row_as_hub_only(
    tmp_path, monkeypatch, structure
):
    df = pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "disease_1": ["class-a", "class-a", "class-a"],
            "med_alpha": [1, 0, 1],
            "med_validation_only": [0, 1, 0],
        }
    )
    dataset_path = tmp_path / "merged_ed.csv"
    df.to_csv(dataset_path, index=False)
    split_path = _write_tiny_canonical_split(
        tmp_path / "split.json",
        monkeypatch,
        {"1": 0, "2": 1, "3": 2},
        ["class-a"],
    )
    monkeypatch.setattr("graphcare_analysis.build_kg.cfg.data_dir", tmp_path)

    kg = build_global_kg(save=False, split_json=split_path)

    tr, va, te, *_ = build_loaders(kg, split_json=split_path, structure=structure)
    assert [len(loader.dataset) for loader in (tr, va, te)] == [1, 1, 1]
    graph = va.dataset[0]
    assert graph["subject_id"] == "2"
    assert graph["node_ids"].tolist() == [kg["patient_hub_id"]]
    assert graph["edge_index"].shape == (2, 0)
    assert graph["rel_ids"].numel() == 0
    assert graph["ehr_nodes"].sum() == 0
    assert graph["y"] == 0


def test_standardized_loader_applies_limit_independently_to_canonical_folds(
    tmp_path, monkeypatch
):
    df = pd.DataFrame(
        {
            "subject_id": list(range(1, 9)),
            "disease_1": ["class-a"] * 8,
            "med_alpha": [1] * 8,
        }
    )
    dataset_path = tmp_path / "merged_ed.csv"
    df.to_csv(dataset_path, index=False)
    split_path = _write_tiny_canonical_split(
        tmp_path / "split.json",
        monkeypatch,
        {"1": 0, "2": 0, "3": 0, "4": 1, "5": 1, "6": 2, "7": 2, "8": 2},
        ["class-a"],
    )
    monkeypatch.setattr("graphcare_analysis.build_kg.cfg.data_dir", tmp_path)
    monkeypatch.setattr("graphcare_analysis.adapter.cfg.data_dir", tmp_path)
    kg = build_global_kg(save=False, split_json=split_path)

    for limit in (2, 4):
        train, validation, test, _, _ = build_loaders(
            kg,
            limit=limit,
            batch_size=8,
            split_json=split_path,
            structure="star",
        )

        observed = {
            "train": len(train.dataset),
            "validation": len(validation.dataset),
            "test": len(test.dataset),
        }
        assert observed == benchmark_runner._effective_fold_counts(
            {"train": 3, "validation": 2, "test": 3}, limit
        )


def test_split_aware_vocab_excludes_validation_and_test_only_concepts():
    df = pd.DataFrame(
        {
            "med_train": [1, 0, 0, 0],
            "med_validation_only": [0, 0, 1, 0],
            "med_test_only": [0, 0, 0, 1],
            "symptom_1": [
                "train-a",
                "train-b",
                "validation-only",
                "test-only",
            ],
        }
    )

    ent2id, _, presence, med_cols = vocab_and_presence(df, fit_indices=[0, 1])

    assert med_cols == ["med_train"]
    assert "med:validation_only" not in ent2id
    assert "med:test_only" not in ent2id
    assert "sym:validation-only" not in ent2id
    assert "sym:test-only" not in ent2id
    assert presence.shape == (4, 3)
    assert presence[2:].sum() == 0


def test_standardized_kg_excludes_diagnosis_symptoms_but_keeps_chiefcomplaints(
    tmp_path, monkeypatch
):
    df = pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "disease_1": ["class-a"] * 3,
            "med_alpha": [1, 1, 1],
            "symptom_1": ["diagnosis-r-code", "diagnosis-r-code", "diagnosis-r-code"],
            "chiefcomplaint_1": ["chest pain", "chest pain", "chest pain"],
        }
    )
    dataset_path = tmp_path / "merged_ed.csv"
    df.to_csv(dataset_path, index=False)
    split_path = _write_tiny_canonical_split(
        tmp_path / "split.json",
        monkeypatch,
        {"1": 0, "2": 1, "3": 2},
        ["class-a"],
    )

    standardized = build_global_kg(
        save=False, df=df, split_json=split_path, dataset_path=dataset_path
    )
    legacy = build_global_kg(save=False, df=df)

    assert not any(name.startswith("sym:") for name in standardized["ent2id"])
    assert "cc:chest pain" in standardized["ent2id"]
    assert "sym:diagnosis-r-code" in legacy["ent2id"]
    assert "cc:chest pain" in legacy["ent2id"]
    assert standardized["preprocessing"]["recipe"]["symptom_input_policy"] == (
        "exclude_diagnosis_derived"
    )


def test_split_aware_kg_rejects_noncanonical_split(tmp_path):
    df = pd.DataFrame({"subject_id": [1], "med_alpha": [1]})
    dataset_path = tmp_path / "merged_ed.csv"
    df.to_csv(dataset_path, index=False)
    split_path = tmp_path / "split.json"
    split_path.write_text(
        '{"fold":{"1":0},"classes":[]}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="fold counts"):
        build_global_kg(
            save=False,
            df=df,
            split_json=split_path,
            dataset_path=dataset_path,
        )


def test_split_aware_kg_rejects_missing_canonical_subject(tmp_path, monkeypatch):
    df = pd.DataFrame({"subject_id": [1], "med_alpha": [1]})
    dataset_path = tmp_path / "merged_ed.csv"
    df.to_csv(dataset_path, index=False)
    split_path = _write_tiny_canonical_split(
        tmp_path / "split.json", monkeypatch, {"1": 0, "2": 1}, []
    )

    with pytest.raises(ValueError, match="subject/row identity"):
        build_global_kg(
            save=False,
            df=df,
            split_json=split_path,
            dataset_path=dataset_path,
        )


def test_standardized_kg_rejects_dataframe_not_loaded_from_bound_dataset(
    tmp_path, monkeypatch
):
    source_df = pd.DataFrame({"subject_id": [1], "med_alpha": [1]})
    dataset_path = tmp_path / "merged_ed.csv"
    source_df.to_csv(dataset_path, index=False)
    split_path = _write_tiny_canonical_split(
        tmp_path / "split.json", monkeypatch, {"1": 0}, []
    )
    changed_df = source_df.assign(med_alpha=0)

    with pytest.raises(ValueError, match="exact source dataset"):
        build_global_kg(
            save=False,
            df=changed_df,
            split_json=split_path,
            dataset_path=dataset_path,
        )


def test_standardized_provenance_binds_dataset_recipe_training_rows_and_adjacency(
    tmp_path, monkeypatch
):
    df = pd.DataFrame(
        {
            "subject_id": [10, 20, 30],
            "med_alpha": [1, 1, 0],
            "med_beta": [1, 0, 1],
        }
    )
    dataset_path = tmp_path / "merged_ed.csv"
    df.to_csv(dataset_path, index=False)
    split_path = _write_tiny_canonical_split(
        tmp_path / "split.json",
        monkeypatch,
        {"10": 0, "20": 0, "30": 1},
        [],
    )
    monkeypatch.setattr("graphcare_analysis.build_kg.cfg.med_min_prev", 0.25)
    monkeypatch.setattr("graphcare_analysis.build_kg.cfg.pmi_threshold", 0.5)

    kg = build_global_kg(
        save=False,
        df=df,
        split_json=split_path,
        dataset_path=dataset_path,
    )

    provenance = kg["preprocessing"]
    assert provenance["schema_version"] == 4
    assert provenance["dataset_sha256"] == file_sha256(dataset_path)
    assert provenance["recipe"]["med_min_prev"] == 0.25
    assert provenance["recipe"]["pmi_threshold"] == 0.5
    assert provenance["recipe"]["med_class_mapping_version"] == "med-classes-v1"
    assert len(provenance["recipe"]["med_class_mapping_sha256"]) == 64
    assert len(provenance["fit_identity_sha256"]) == 64
    assert len(provenance["typed_adjacency_sha256"]) == 64
    assert provenance["recipe"]["graph_membership_contract"]["vital_policy"] == (
        "exclude_nodes_no_categorical_proxy"
    )
    assert validate_training_provenance(
        kg, split_path, dataset_path=dataset_path
    ) == provenance


def test_split_aware_kg_fits_pmi_only_on_training_rows(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4, 5, 6],
            "med_alpha": [1, 0, 0, 0, 1, 1],
            "med_beta": [0, 1, 0, 0, 1, 1],
        }
    )
    dataset_path = tmp_path / "merged_ed.csv"
    df.to_csv(dataset_path, index=False)
    split_path = _write_tiny_canonical_split(
        tmp_path / "split.json",
        monkeypatch,
        {"1": 0, "2": 0, "3": 0, "4": 0, "5": 1, "6": 2},
        [],
    )
    monkeypatch.setattr("graphcare_analysis.build_kg.cfg.pmi_threshold", 0.1)

    kg = build_global_kg(
        save=False, df=df, split_json=split_path, dataset_path=dataset_path
    )

    alpha = kg["ent2id"]["med:alpha"]
    beta = kg["ent2id"]["med:beta"]
    assert (kg["rel2id"]["co_occurs"], beta) not in kg["neighbours"][alpha]
    assert kg["preprocessing"]["fit_scope"] == "training_fold"
    assert kg["preprocessing"]["fit_fold"] == 0
    assert kg["preprocessing"]["fit_row_count"] == 4
    assert kg["preprocessing"]["split_sha256"] == file_sha256(split_path)


@pytest.mark.parametrize(
    "preprocessing",
    [
        None,
        {
            "schema_version": 1,
            "fit_scope": "all_rows_legacy",
            "fit_fold": None,
            "fit_row_count": 5,
            "split_sha256": None,
        },
        {
            "schema_version": 1,
            "fit_scope": "training_fold",
            "fit_fold": 0,
            "fit_row_count": 4,
            "split_sha256": "wrong-split",
        },
    ],
)
def test_standardized_preprocessing_fails_closed_without_matching_training_provenance(
    tmp_path, monkeypatch, preprocessing
):
    df = pd.DataFrame({"subject_id": [1], "med_alpha": [1]})
    dataset_path = tmp_path / "merged_ed.csv"
    df.to_csv(dataset_path, index=False)
    split_path = _write_tiny_canonical_split(
        tmp_path / "split.json", monkeypatch, {"1": 0}, []
    )
    kg = _single_concept_kg(preprocessing)

    with pytest.raises(ValueError, match="training-fold provenance"):
        validate_training_provenance(
            kg, split_path, dataset_path=dataset_path
        )


def _single_concept_kg(preprocessing=None):
    kg = {
        "num_nodes": 2,
        "num_rels": 3,
        "ent2id": {"med:alpha": 0, "patient:hub": 1},
        "rel2id": {
            "co_occurs": 0,
            "same_class": 1,
            "patient_has_concept": 2,
        },
        "neighbours": {0: [], 1: []},
        "patient_hub_id": 1,
        "patient_relation_id": 2,
    }
    if preprocessing is not None:
        kg["preprocessing"] = preprocessing
    return kg


def _build_provenance_fixture(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "subject_id": [10, 20, 30],
            "disease_1": ["class-a", "class-a", "class-a"],
            "med_alpha": [1, 1, 0],
            "med_beta": [1, 0, 1],
        }
    )
    dataset_path = tmp_path / "merged_ed.csv"
    df.to_csv(dataset_path, index=False)
    split_path = _write_tiny_canonical_split(
        tmp_path / "split.json",
        monkeypatch,
        {"10": 0, "20": 0, "30": 1},
        ["class-a"],
    )
    kg = build_global_kg(
        save=False, df=df, split_json=split_path, dataset_path=dataset_path
    )
    return df, dataset_path, split_path, kg


@pytest.mark.parametrize(
    "mutation",
    ["dataset", "recipe", "med-class", "training-identity", "adjacency"],
)
def test_standardized_provenance_rejects_every_bound_identity(
    tmp_path, monkeypatch, mutation
):
    _, dataset_path, split_path, kg = _build_provenance_fixture(
        tmp_path, monkeypatch
    )
    if mutation == "dataset":
        dataset_path.write_bytes(dataset_path.read_bytes() + b"\n")
    elif mutation == "recipe":
        monkeypatch.setattr(
            "graphcare_analysis.build_kg.cfg.pmi_threshold",
            graphcare_build_kg.cfg.pmi_threshold + 1.0,
        )
    elif mutation == "med-class":
        monkeypatch.setitem(graphcare_build_kg.MED_CLASS, "test_drug", "test_class")
    elif mutation == "training-identity":
        kg["preprocessing"]["fit_identity_sha256"] = "wrong-training-identity"
    else:
        kg["neighbours"][0].append((kg["rel2id"]["co_occurs"], 0))

    with pytest.raises(ValueError, match="provenance"):
        validate_training_provenance(
            kg,
            split_path,
            dataset_path=dataset_path,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda kg: kg.update(rel2id={
            "co_occurs": 1,
            "same_class": 0,
            "patient_has_concept": 2,
        }),
        lambda kg: kg.update(rel2id={
            "co_occurs": False,
            "same_class": 1,
            "patient_has_concept": 2,
        }),
        lambda kg: kg.update(num_rels=2),
        lambda kg: kg.update(patient_relation_id=3),
        lambda kg: kg["neighbours"][0].append((3, 0)),
        lambda kg: kg["neighbours"][0].append((0, 2)),
        lambda kg: kg["neighbours"][1].append((0, 0)),
    ],
)
def test_kg_schema_rejects_swapped_out_of_range_or_corrupt_adjacency(mutation):
    kg = deepcopy(_single_concept_kg())
    mutation(kg)

    with pytest.raises(ValueError, match="GraphCare KG schema"):
        validate_kg_schema(kg)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda kg: kg.update(rel2id={
            "co_occurs": 1,
            "same_class": 0,
            "patient_has_concept": 2,
        }),
        lambda kg: kg["neighbours"][0].append((3, 0)),
        lambda kg: kg["neighbours"][0].append((0, 2)),
        lambda kg: kg["neighbours"][1].append((0, 0)),
    ],
)
def test_cached_kg_rejects_corrupt_relation_or_adjacency_before_use(
    tmp_path, mutation
):
    kg = deepcopy(_single_concept_kg())
    mutation(kg)
    kg_path = tmp_path / "kg.pt"
    torch.save(kg, kg_path)

    with pytest.raises(ValueError, match="GraphCare KG schema"):
        _load_kg(kg_path)


def test_standardized_loader_rejects_noncanonical_split(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {"subject_id": [1], "disease_1": ["class-a"], "med_alpha": [1]}
    )
    split_path = tmp_path / "split.json"
    split_path.write_text(
        '{"fold":{"1":0},"classes":["class-a"]}', encoding="utf-8"
    )
    monkeypatch.setattr("graphcare_analysis.adapter.pd.read_csv", lambda *a, **k: df)
    kg = _single_concept_kg(
        {
            "schema_version": 1,
            "fit_scope": "training_fold",
            "fit_fold": 0,
            "fit_row_count": 1,
            "split_sha256": file_sha256(split_path),
        }
    )

    with pytest.raises(ValueError, match="fold counts"):
        build_loaders(kg, split_json=split_path, structure="star")


def test_standardized_loader_rejects_kg_without_training_provenance(
    tmp_path, monkeypatch
):
    df = pd.DataFrame(
        {
            "subject_id": [1],
            "disease_1": ["class-a"],
            "med_alpha": [1],
        }
    )
    df.to_csv(tmp_path / "merged_ed.csv", index=False)
    split_path = _write_tiny_canonical_split(
        tmp_path / "split.json", monkeypatch, {"1": 0}, ["class-a"]
    )
    monkeypatch.setattr("graphcare_analysis.adapter.cfg.data_dir", tmp_path)
    legacy_kg = _single_concept_kg()

    with pytest.raises(ValueError, match="training-fold provenance"):
        build_loaders(legacy_kg, split_json=split_path, structure="star")


def test_standardized_loader_rejects_wrong_training_row_count(
    tmp_path, monkeypatch
):
    df = pd.DataFrame(
        {"subject_id": [1], "disease_1": ["class-a"], "med_alpha": [1]}
    )
    df.to_csv(tmp_path / "merged_ed.csv", index=False)
    split_path = _write_tiny_canonical_split(
        tmp_path / "split.json", monkeypatch, {"1": 0}, ["class-a"]
    )
    monkeypatch.setattr("graphcare_analysis.adapter.cfg.data_dir", tmp_path)
    kg = build_global_kg(save=False, split_json=split_path)
    kg["preprocessing"]["fit_row_count"] = 2

    with pytest.raises(ValueError, match="training-fold provenance"):
        build_loaders(kg, split_json=split_path, structure="star")


def test_standardized_kg_cache_is_validated_before_use(tmp_path, monkeypatch):
    df = pd.DataFrame({"subject_id": [1], "med_alpha": [1]})
    df.to_csv(tmp_path / "merged_ed.csv", index=False)
    split_path = _write_tiny_canonical_split(
        tmp_path / "split.json", monkeypatch, {"1": 0}, []
    )
    monkeypatch.setattr("graphcare_analysis.build_kg.cfg.data_dir", tmp_path)
    kg_path = tmp_path / "kg.pt"
    torch.save(
        _single_concept_kg({"fit_scope": "all_rows_legacy"}), kg_path
    )

    with pytest.raises(ValueError, match="training-fold provenance"):
        _load_kg(kg_path, split_json=split_path)
