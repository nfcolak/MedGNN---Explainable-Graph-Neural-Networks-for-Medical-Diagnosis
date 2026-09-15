"""Production-cache cross-method canonical topology invariant tests."""

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from comparison.standardized.audit_caches import audit_standardized_caches
from graphcare_analysis.adapter import build_standardized_dataset
from graphcare_analysis.build_kg import build_global_kg
from protgnn_analysis.load_dataset import IntraPatientHeteroDataset
from shared.lib import benchmark_contract
from shared.lib.canonical_graph import (
    CanonicalGraph,
    canonical_graph_from_graphcare_record,
    canonical_graph_from_pyg_record,
    validate_cross_method_fingerprints,
)


def _write_fixture(project_root, monkeypatch):
    data_dir = project_root / "data"
    comparison_dir = project_root / "comparison"
    data_dir.mkdir(parents=True)
    comparison_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4, 5, 6],
            "disease_1": ["class-a"] * 6,
            # Vital-bearing input must not become categorical vital-name-only nodes.
            "heartrate": [70.0, 75.0, 80.0, 85.0, 90.0, 95.0],
            "med_alpha": [1, 0, 0, 0, 1, 1],
            "med_beta": [1, 0, 0, 0, 1, 1],
            "med_gamma": [0, 1, 1, 1, 0, 0],
            "chiefcomplaint_1": [
                "chest pain", "fatigue", "fatigue", "fatigue", "chest pain", "chest pain"
            ],
            "chiefcomplaint_2": ["chest pain", "", "", "", "", ""],
        }
    )
    dataset_path = data_dir / "merged_ed.csv"
    frame.to_csv(dataset_path, index=False)
    split_path = comparison_dir / "canonical_split.json"
    folds = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 1, "6": 2}
    monkeypatch.setattr(
        benchmark_contract,
        "EXPECTED_FOLD_COUNTS",
        {0: 4, 1: 1, 2: 1},
    )
    monkeypatch.setattr(benchmark_contract, "EXPECTED_CLASSES", ("class-a",))
    monkeypatch.setattr(benchmark_contract, "EXPECTED_CLASS_COUNT", 1)
    split_path.write_text(
        json.dumps({"fold": folds, "classes": ["class-a"], "n_kept": 6}),
        encoding="utf-8",
    )
    monkeypatch.setattr("graphcare_analysis.build_kg.cfg.data_dir", data_dir)
    monkeypatch.setattr("graphcare_analysis.adapter.cfg.data_dir", data_dir)
    monkeypatch.setattr("graphcare_analysis.build_kg.cfg.pmi_threshold", 0.1)
    return data_dir, dataset_path, split_path


def _production_graphs(tmp_path, monkeypatch, structure):
    data_dir, dataset_path, split_path = _write_fixture(tmp_path, monkeypatch)
    pyg_dataset = IntraPatientHeteroDataset(
        root=data_dir,
        name="mimic_intra_patient_disease",
        csv_filename="merged_ed.csv",
        med_min_prev=0.01,
        pmi_threshold=0.1,
        icd_min_prev=0.0,
        target="disease",
        graph_structure=structure,
        canonical_split=split_path,
    )
    kg_path = data_dir / "graphs" / structure / "graphcare" / "kg.pt"
    kg_path.parent.mkdir(parents=True)
    kg = build_global_kg(
        save=True,
        save_path=kg_path,
        split_json=split_path,
        dataset_path=dataset_path,
    )
    graphcare_dataset, *_ = build_standardized_dataset(
        kg,
        split_json=split_path,
        structure=structure,
        dataset_path=dataset_path,
    )
    pyg_record = next(
        pyg_dataset[index]
        for index in range(len(pyg_dataset))
        if int(pyg_dataset[index].subject_id) == 1
    )
    graphcare_record = next(
        graphcare_dataset[index]
        for index in range(len(graphcare_dataset))
        if graphcare_dataset[index]["subject_id"] == "1"
    )
    return (
        canonical_graph_from_pyg_record(pyg_record),
        canonical_graph_from_graphcare_record(graphcare_record, kg),
        pyg_record,
        graphcare_record,
        kg,
        split_path,
    )


@pytest.mark.parametrize("structure", ["star", "cooccur"])
def test_actual_standardized_builders_match_on_vital_bearing_input(
    tmp_path, monkeypatch, structure
):
    pyg_graph, graphcare_graph, pyg_record, _, kg, _ = _production_graphs(
        tmp_path, monkeypatch, structure
    )

    fingerprint = validate_cross_method_fingerprints(
        {"protgnn": pyg_graph, "gsat": pyg_graph, "graphcare": graphcare_graph}
    )

    assert len(fingerprint) == 64
    assert {node.node_id for node in pyg_graph.nodes} == {
        "patient:hub", "med:alpha", "med:beta", "cc:chest pain"
    }
    assert not any(node.node_type == "vital" for node in pyg_graph.nodes)
    assert not any("vital" in node_id for node_id in pyg_record.canonical_node_ids)
    assert kg["canonical_node_ids_by_global_id"][kg["patient_hub_id"]] == "patient:hub"
    assert kg["canonical_edge_types_by_relation_id"] == [
        "cooccur", "ontology", "patient_concept"
    ]


def test_cross_method_validator_rejects_node_mismatch_from_production_adapters(
    tmp_path, monkeypatch
):
    pyg_graph, graphcare_graph, *_ = _production_graphs(tmp_path, monkeypatch, "star")
    removed = next(node for node in graphcare_graph.nodes if node.node_id == "med:beta")
    mismatched = CanonicalGraph(
        graphcare_graph.subject_id,
        graphcare_graph.class_id,
        [node for node in graphcare_graph.nodes if node != removed],
        [
            edge for edge in graphcare_graph.edges
            if removed.node_id not in (edge.source, edge.target)
        ],
    )

    with pytest.raises(ValueError, match="different canonical node/edge fingerprints"):
        validate_cross_method_fingerprints({"protgnn": pyg_graph, "graphcare": mismatched})


def test_cross_method_validator_rejects_edge_mismatch_from_production_adapters(
    tmp_path, monkeypatch
):
    pyg_graph, graphcare_graph, *_ = _production_graphs(tmp_path, monkeypatch, "cooccur")
    mismatched = replace(graphcare_graph, edges=graphcare_graph.edges[:-1])

    with pytest.raises(ValueError, match="different canonical node/edge fingerprints"):
        validate_cross_method_fingerprints({"protgnn": pyg_graph, "graphcare": mismatched})


@pytest.mark.parametrize("adapter", ["pyg", "graphcare"])
def test_production_adapters_fail_closed_when_canonical_metadata_is_missing(adapter):
    if adapter == "pyg":
        record = Data(
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            y=torch.tensor([0]),
            subject_id=torch.tensor([1]),
        )
        with pytest.raises(ValueError, match="(?i)canonical.*metadata"):
            canonical_graph_from_pyg_record(record)
    else:
        record = {
            "node_ids": torch.tensor([0]),
            "edge_index": torch.zeros((2, 0), dtype=torch.long),
            "rel_ids": torch.zeros(0, dtype=torch.long),
            "y": 0,
            "subject_id": "1",
        }
        with pytest.raises(ValueError, match="(?i)canonical.*metadata"):
            canonical_graph_from_graphcare_record(record, {"num_nodes": 1})


@pytest.mark.parametrize("structure", ["star", "cooccur"])
def test_cache_audit_reads_real_cache_records_and_reports_full_parity(
    tmp_path, monkeypatch, structure
):
    *_, split_path = _production_graphs(tmp_path, monkeypatch, structure)

    report = audit_standardized_caches(
        project_root=tmp_path,
        dataset_path=tmp_path / "data" / "merged_ed.csv",
        split_path=split_path,
        structures=[structure],
        protgnn_parameters={"med_min_prev": 0.01, "pmi_threshold": 0.1},
    )

    assert report[structure]["subject_count"] == 6
    assert len(report[structure]["parity_sha256"]) == 64


def test_cache_audit_fails_closed_without_creating_missing_caches(tmp_path, monkeypatch):
    data_dir, _, split_path = _write_fixture(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="Missing production ProtGNN/GSAT star cache"):
        audit_standardized_caches(
            project_root=tmp_path,
            dataset_path=data_dir / "merged_ed.csv",
            split_path=split_path,
            structures=["star"],
        )

    assert not (data_dir / "graphs").exists()
