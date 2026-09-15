"""End-to-end contract tests for standardized explanations."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from comparison.standardized.run_explanations import (
    build_commands,
    validate_cross_method_outputs,
)
from graphcare_analysis.explainability.explain_graphcare import (
    GraphCareGraphXAIWrapper,
    explain_graphcare_record,
)
from gsat_analysis.explainability.explain_gsat import select_standardized_records as select_gsat_records
from protgnn_analysis.explainability.explain_standardized import (
    select_standardized_records as select_protgnn_records,
)
from shared.lib.explanation_contract import (
    EXPLANATION_METRIC_KEYS,
    STANDARDIZED_METHODS,
    build_node_explanation,
    load_explanation_cohort,
    resolve_node_top_k,
    select_exact_subject_records,
    write_standardized_explanations,
)


ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "comparison" / "standardized" / "explanation_subjects.json"
SPLIT = ROOT / "comparison" / "canonical_split.json"
DATASET = ROOT / "data" / "merged_ed.csv"


class _SumClassifier(nn.Module):
    def forward(self, x, edge_index, batch=None):
        score = x[:, 0].sum().reshape(1) - 1.0
        return torch.stack((torch.zeros_like(score), score), dim=1)


class _ToyGraphCare(nn.Module):
    """Small model exercising both embedding lookup paths GraphCare uses."""

    def __init__(self):
        super().__init__()
        self.node_emb = nn.Embedding.from_pretrained(
            torch.tensor([[1.0], [2.0], [3.0], [4.0]]), freeze=False
        )
        self.num_nodes = 4

    def forward(self, node_ids, rel_ids, edge_index, batch, visit_node, ehr_nodes):
        local = self.node_emb(node_ids).sum().reshape(1)
        ehr = (ehr_nodes @ self.node_emb.weight).sum().reshape(1)
        score = local + ehr - 5.0
        return torch.stack((torch.zeros_like(score), score), dim=1)


def _cohort_payload():
    return json.loads(COHORT.read_text(encoding="utf-8"))


def _write_payload(tmp_path, payload):
    path = tmp_path / "cohort.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_explanation_cohort_consumes_valid_exact_50():
    cohort = load_explanation_cohort(COHORT, split_path=SPLIT, dataset_path=DATASET, expected_count=50)

    assert len(cohort.subject_ids) == len(set(cohort.subject_ids)) == 50
    assert cohort.subject_ids == tuple(_cohort_payload()["subject_ids"])
    assert cohort.classes == tuple(_cohort_payload()["class_counts"])


def test_load_explanation_cohort_rejects_missing_test_id(tmp_path):
    payload = _cohort_payload()
    payload["subject_ids"][-1] = "not-a-canonical-test-subject"

    with pytest.raises(ValueError, match="test fold"):
        load_explanation_cohort(
            _write_payload(tmp_path, payload), split_path=SPLIT, dataset_path=DATASET, expected_count=50
        )


def test_load_explanation_cohort_rejects_duplicate_id(tmp_path):
    payload = _cohort_payload()
    payload["subject_ids"][-1] = payload["subject_ids"][0]

    with pytest.raises(ValueError, match="50 unique"):
        load_explanation_cohort(
            _write_payload(tmp_path, payload), split_path=SPLIT, dataset_path=DATASET, expected_count=50
        )


def test_shared_top_k_policy_resolves_identically_on_small_graphs():
    assert [resolve_node_top_k(n).resolved_k for n in (1, 2, 4, 5, 6, 10)] == [
        1,
        1,
        1,
        1,
        1,
        2,
    ]
    assert resolve_node_top_k(10).requested_fraction == pytest.approx(0.2)


def test_shared_node_explanation_has_prediction_target_and_exact_metric_schema():
    model = _SumClassifier().eval()
    x = torch.tensor([[2.0], [1.0], [0.0]], requires_grad=True)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    batch = torch.zeros(3, dtype=torch.long)

    record = build_node_explanation(model, x, edge_index, batch=batch)

    assert set(record) == set(EXPLANATION_METRIC_KEYS)
    assert record["target"] == {"provenance": "model_prediction", "class_id": 1}
    assert record["top_k"] == {
        "policy": "node_fraction_floor_min_1",
        "requested_fraction": 0.2,
        "resolved_k": 1,
        "num_nodes": 3,
    }
    assert len(record["node_importance"]) == 3
    assert record["fidelity_plus"]["k"] == record["fidelity_minus"]["k"] == 1


def test_protgnn_standardized_selection_is_by_subject_not_loader_order():
    requested = ["3", "1", "2"]
    records = [
        {"subject_id": torch.tensor([1])},
        {"subject_id": torch.tensor([2])},
        {"subject_id": torch.tensor([3])},
        {"subject_id": torch.tensor([99])},
    ]

    selected = select_protgnn_records(records, requested)

    assert [str(int(item["subject_id"].item())) for item in selected] == requested


def test_gsat_standardized_selection_never_silently_returns_subset():
    with pytest.raises(ValueError, match="missing requested subject"):
        select_gsat_records([{"subject_id": "1"}], ["1", "2"])


def test_exact_selection_rejects_duplicate_source_subject():
    with pytest.raises(ValueError, match="duplicate subject"):
        select_exact_subject_records(
            [{"subject_id": "1"}, {"subject_id": "1"}], ["1"]
        )


def test_graphcare_wrapper_forward_importance_and_shared_metrics():
    model = _ToyGraphCare().eval()
    graph = {
        "node_ids": torch.tensor([0, 2]),
        "rel_ids": torch.empty(0, dtype=torch.long),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
        "batch": torch.zeros(2, dtype=torch.long),
        "visit_node": torch.ones((1, 1, 4)),
        "ehr_nodes": torch.tensor([[1.0, 0.0, 1.0, 0.0]]),
        "y": torch.tensor([1]),
        "subject_id": "7",
    }
    wrapper = GraphCareGraphXAIWrapper(model)
    x = wrapper.set_context(graph)

    expected = model(
        graph["node_ids"], graph["rel_ids"], graph["edge_index"], graph["batch"],
        graph["visit_node"], graph["ehr_nodes"],
    )
    assert torch.allclose(wrapper(x, graph["edge_index"], graph["batch"]), expected)
    assert wrapper.verify()[0] is True

    record = explain_graphcare_record(model, graph)
    assert record["subject_id"] == "7"
    assert record["attribution_method"] == "deterministic_input_x_gradient"
    assert set(record["node_explanation"]) == set(EXPLANATION_METRIC_KEYS)
    assert len(record["node_explanation"]["node_importance"]) == 2


def _base_record(method, subject):
    explanation = build_node_explanation(
        _SumClassifier().eval(),
        torch.tensor([[2.0], [1.0]]),
        torch.empty((2, 0), dtype=torch.long),
        batch=torch.zeros(2, dtype=torch.long),
    )
    return {
        "schema": "medgnn.standardized_node_explanation",
        "schema_version": 1,
        "method": method,
        "subject_id": subject,
        "topology": "star",
        "seed": 1234,
        "true_class_id": 0,
        "prediction_class_id": explanation["target"]["class_id"],
        "attribution_method": "synthetic",
        "node_explanation": explanation,
    }


def _write_method_output(root, method, subjects):
    records = [_base_record(method, subject) for subject in subjects]
    write_standardized_explanations(
        output_dir=root / method,
        records=records,
        method=method,
        topology="star",
        seed=1234,
        cohort_path=COHORT,
        split_path=SPLIT,
        dataset_path=DATASET,
        checkpoint_path=None,
        allow_missing_checkpoint=True,
        expected_count=50,
    )


def test_cross_method_validator_accepts_exact_subject_and_schema_match(tmp_path):
    subjects = list(_cohort_payload()["subject_ids"])
    for method in STANDARDIZED_METHODS:
        _write_method_output(tmp_path, method, subjects)

    summary = validate_cross_method_outputs(
        tmp_path, cohort_path=COHORT, split_path=SPLIT, dataset_path=DATASET,
        topology="star", seed=1234, cohort_size=50,
    )

    assert summary["subject_count"] == 50
    assert summary["methods"] == list(STANDARDIZED_METHODS)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "extra", "metric_schema"])
def test_cross_method_validator_rejects_missing_duplicate_extra_or_schema_mismatch(
    tmp_path, defect
):
    subjects = list(_cohort_payload()["subject_ids"])
    for method in STANDARDIZED_METHODS:
        _write_method_output(tmp_path, method, subjects)

    target_dir = tmp_path / "gsat"
    files = sorted(target_dir.glob("subject_*.json"))
    if defect == "missing":
        files[0].unlink()
    elif defect == "duplicate":
        duplicate = json.loads(files[0].read_text(encoding="utf-8"))
        (target_dir / "duplicate.json").write_text(json.dumps(duplicate), encoding="utf-8")
    elif defect == "extra":
        extra = _base_record("gsat", "999999")
        (target_dir / "subject_999999.json").write_text(json.dumps(extra), encoding="utf-8")
    else:
        record = json.loads(files[0].read_text(encoding="utf-8"))
        del record["node_explanation"]["sparsity"]
        files[0].write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="missing|duplicate|extra|metric schema"):
        validate_cross_method_outputs(
            tmp_path, cohort_path=COHORT, split_path=SPLIT, dataset_path=DATASET,
            topology="star", seed=1234, cohort_size=50,
        )


def test_explanation_dry_run_builds_three_commands_without_outputs(tmp_path):
    commands = build_commands(
        topology="star", seed=1234, cohort_size=50, output_root=tmp_path / "not-created",
        cohort_path=COHORT, split_path=SPLIT, dataset_path=DATASET,
        require_checkpoints=False,
    )

    assert len(commands) == len({tuple(command) for command in commands}) == 3
    assert not (tmp_path / "not-created").exists()
    assert all("--cohort" in command for command in commands)
