"""Regression tests for standardized ProtGNN/GSAT preprocessing."""

import json
from pathlib import Path

import pandas as pd
import pytest

from protgnn_analysis.load_dataset import (
    IntraPatientHeteroDataset,
    get_dataloader,
)
from shared.lib import benchmark_contract
from shared.lib.benchmark_contract import file_sha256


def _write_split(path, monkeypatch, folds):
    monkeypatch.setattr(
        benchmark_contract,
        "EXPECTED_FOLD_COUNTS",
        {fold_id: list(folds.values()).count(fold_id) for fold_id in (0, 1, 2)},
    )
    monkeypatch.setattr(benchmark_contract, "EXPECTED_CLASSES", ("class-a",))
    monkeypatch.setattr(benchmark_contract, "EXPECTED_CLASS_COUNT", 1)
    path.write_text(
        json.dumps({"fold": folds, "classes": ["class-a"], "n_kept": len(folds)}),
        encoding="utf-8",
    )
    return path


def _write_dataset(path):
    frame = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4, 5, 6],
            "disease_1": ["class-a"] * 6,
            "heartrate": [60.0, 70.0, 80.0, 90.0, 190.0, 200.0],
            "age": [10.0, 20.0, 30.0, 40.0, 100.0, 110.0],
            "med_alpha": [1, 0, 0, 0, 1, 1],
            "med_beta": [0, 1, 0, 0, 1, 1],
            "med_heldout": [0, 0, 0, 0, 1, 1],
            "symptom_1": ["", "", "train-sym", "", "heldout-sym", "heldout-sym"],
            "chiefcomplaint_1": ["", "", "", "train-cc", "heldout-cc", "heldout-cc"],
        }
    )
    frame.to_csv(path, index=False)
    return frame


def test_standardized_preprocessing_fits_every_learned_artifact_on_fold_zero(
    tmp_path, monkeypatch
):
    dataset_path = tmp_path / "merged_ed.csv"
    _write_dataset(dataset_path)
    split_path = _write_split(
        tmp_path / "canonical_split.json",
        monkeypatch,
        {"1": 0, "2": 0, "3": 0, "4": 0, "5": 1, "6": 2},
    )

    dataset = IntraPatientHeteroDataset(
        root=tmp_path,
        name="tiny",
        csv_filename=dataset_path.name,
        med_min_prev=0.2,
        pmi_threshold=0.1,
        target="disease",
        graph_structure="cooccur",
        canonical_split=split_path,
    )
    metadata = json.loads(Path(dataset.metadata_path).read_text(encoding="utf-8"))

    assert len(dataset) == 6
    assert metadata["med_vocab"] == ["med_alpha", "med_beta"]
    assert metadata["symptom_vocab"] == []
    assert metadata["cc_vocab"] == ["train-cc"]
    assert not any("symptom" in name or "sym_id[" in name for name in dataset.feature_cols)
    assert all(int(dataset[index].num_sym_nodes) == 0 for index in range(len(dataset)))
    assert int(dataset[3].num_cc_nodes) == 1
    assert metadata["vital_vocab"] == []
    assert metadata["vital_stats"] == {}
    assert not any("vital" in name for name in dataset.feature_cols)
    assert all(
        "vital" not in dataset[index].canonical_node_types
        for index in range(len(dataset))
    )
    assert metadata["patient_num_stats"]["age"]["mean"] == pytest.approx(25.0)
    assert metadata["n_med_med_edges"] == 0
    assert metadata["n_concept_cooccur_edges"] == 0
    assert int(dataset[4].num_med_nodes) == 2

    provenance = metadata["preprocessing"]
    assert provenance["fit_scope"] == "training_fold"
    assert provenance["fit_fold"] == 0
    assert provenance["fit_subject_count"] == 4
    assert provenance["split_sha256"] == file_sha256(split_path)
    assert provenance["dataset_sha256"] == file_sha256(dataset_path)
    assert len(provenance["fit_subjects_sha256"]) == 64
    assert provenance["recipe"]["symptom_input_policy"] == "exclude_diagnosis_derived"
    assert provenance["schema_version"] == 4
    assert provenance["recipe"]["graph_membership_contract"]["vital_policy"] == (
        "exclude_nodes_no_categorical_proxy"
    )


@pytest.mark.parametrize("structure", ["star", "cooccur"])
def test_standardized_hub_only_has_no_fabricated_self_edge(tmp_path, monkeypatch, structure):
    source = tmp_path / "merged_ed.csv"
    _write_dataset(source)
    split = _write_split(tmp_path / "split.json", monkeypatch,
                         {"1": 0, "2": 0, "3": 0, "4": 0, "5": 1, "6": 2})
    dataset = IntraPatientHeteroDataset(root=tmp_path, name="tiny",
        csv_filename=source.name, target="disease", graph_structure=structure,
        canonical_split=split)
    graph = dataset[2]  # diagnosis-derived symptom excluded: no fitted clinical content
    assert graph.canonical_node_ids == ["patient:hub"]
    assert graph.edge_index.shape == (2, 0)
    assert graph.canonical_edge_types == []
    assert len(dataset) == 6
    assert dataset._preprocessing_recipe()["clinical_empty_graph_policy"] == (
        "retain_patient_hub_no_edges_v1"
    )


def test_legacy_preprocessing_preserves_symptoms_and_chiefcomplaints(tmp_path):
    dataset_path = tmp_path / "merged_ed.csv"
    _write_dataset(dataset_path)

    dataset = IntraPatientHeteroDataset(
        root=tmp_path,
        name="tiny",
        csv_filename=dataset_path.name,
        med_min_prev=0.2,
        pmi_threshold=0.1,
        target="disease",
        graph_structure="cooccur",
    )
    metadata = json.loads(Path(dataset.metadata_path).read_text(encoding="utf-8"))

    assert metadata["symptom_vocab"] == ["heldout-sym", "train-sym"]
    assert metadata["cc_vocab"] == ["heldout-cc", "train-cc"]
    assert any("sym_id[" in name for name in dataset.feature_cols)
    assert any(int(dataset[index].num_sym_nodes) > 0 for index in range(len(dataset)))


def test_standardized_cache_is_isolated_and_fails_closed_on_legacy_provenance(
    tmp_path, monkeypatch
):
    dataset_path = tmp_path / "merged_ed.csv"
    _write_dataset(dataset_path)
    split_path = _write_split(
        tmp_path / "canonical_split.json",
        monkeypatch,
        {"1": 0, "2": 0, "3": 0, "4": 0, "5": 1, "6": 2},
    )
    kwargs = dict(
        root=tmp_path,
        name="tiny",
        csv_filename=dataset_path.name,
        med_min_prev=0.2,
        pmi_threshold=0.1,
        target="disease",
        graph_structure="cooccur",
    )
    legacy = IntraPatientHeteroDataset(**kwargs)
    standardized = IntraPatientHeteroDataset(
        **kwargs, canonical_split=split_path
    )

    assert legacy.processed_dir != standardized.processed_dir
    legacy_metadata = json.loads(Path(legacy.metadata_path).read_text(encoding="utf-8"))
    assert legacy_metadata["preprocessing"]["fit_scope"] == "all_rows_legacy"
    with pytest.raises(ValueError, match="built with the same validated canonical split"):
        get_dataloader(legacy, batch_size=1, canonical_split=split_path)

    standardized_metadata = json.loads(
        Path(standardized.metadata_path).read_text(encoding="utf-8")
    )
    standardized_metadata["preprocessing"] = legacy_metadata["preprocessing"]
    Path(standardized.metadata_path).write_text(
        json.dumps(standardized_metadata), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="matching training-fold provenance"):
        IntraPatientHeteroDataset(**kwargs, canonical_split=split_path)
