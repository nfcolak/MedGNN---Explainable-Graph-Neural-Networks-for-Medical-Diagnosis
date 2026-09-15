"""Behavior tests for the shared explanation metrics."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from shared.lib.fidelity import fidelity_minus, fidelity_plus, sparsity
from gsat_analysis.explainability.explain_gsat import _record_importance
from gsat_analysis.explainability.summarize_gsat import build_rows


class _SumClassifier(nn.Module):
    def forward(self, x, edge_index, batch=None):
        score = x[:, 0].sum().reshape(1) - 2.0
        return torch.stack((torch.zeros_like(score), score), dim=1)


def test_fidelity_plus_removes_important_nodes_and_minus_keeps_them():
    model = _SumClassifier().eval()
    x = torch.tensor([[3.0], [1.0], [0.0]])
    edge_index = torch.empty((2, 0), dtype=torch.long)
    batch = torch.zeros(3, dtype=torch.long)
    importance = np.array([3.0, 1.0, 0.0])

    plus = fidelity_plus(model, x, edge_index, importance, 1, batch, k=1)
    minus = fidelity_minus(model, x, edge_index, importance, 1, batch, k=1)

    assert plus["acc"] == 1
    assert plus["prob"] > 0
    assert plus["k"] == 1
    assert minus["acc"] == 0
    assert 0 < minus["prob"] < plus["prob"]


def test_fidelity_scores_the_requested_class_not_the_ground_truth():
    model = _SumClassifier().eval()
    x = torch.tensor([[3.0], [1.0], [0.0]])
    edge_index = torch.empty((2, 0), dtype=torch.long)
    batch = torch.zeros(3, dtype=torch.long)
    importance = np.array([3.0, 1.0, 0.0])

    predicted_class = fidelity_plus(
        model, x, edge_index, importance, target_class=1, batch=batch, k=1
    )
    other_class = fidelity_plus(
        model, x, edge_index, importance, target_class=0, batch=batch, k=1
    )

    assert predicted_class["prob"] == pytest.approx(-other_class["prob"])
    assert predicted_class["target_class"] == 1
    assert other_class["target_class"] == 0


def test_gsat_record_binds_fidelity_to_prediction_explanation_target():
    model = _SumClassifier().eval()
    x = torch.tensor([[3.0], [1.0], [0.0]])
    edge_index = torch.empty((2, 0), dtype=torch.long)
    batch = torch.zeros(3, dtype=torch.long)
    record = {}
    aggregate = {
        "sp": {"BuiltinAttention": []},
        "fp": {"BuiltinAttention": []},
        "fm": {"BuiltinAttention": []},
    }

    _record_importance(
        record,
        "BuiltinAttention",
        np.array([3.0, 1.0, 0.0]),
        model,
        x,
        edge_index,
        target_class=1,
        target_provenance="model_prediction",
        batch=batch,
        top_k=1,
        agg=aggregate,
    )

    explanation = record["BuiltinAttention"]
    assert explanation["target"] == {
        "provenance": "model_prediction",
        "class_id": 1,
    }
    assert explanation["fidelity_plus"]["target_class"] == 1
    assert explanation["fidelity_minus"]["target_class"] == 1


def test_gsat_summary_preserves_explanation_target_provenance(tmp_path):
    explanation_path = tmp_path / "graph_0.json"
    explanation_path.write_text(
        __import__("json").dumps(
            {
                "graph_idx": 0,
                "pred_label": 2,
                "true_label": 1,
                "correct": False,
                "num_nodes": 1,
                "node_tokens": ["patient"],
                "explanation_target": {
                    "provenance": "model_prediction",
                    "class_id": 2,
                    "class_name": "class-c",
                },
                "explanations": {},
            }
        ),
        encoding="utf-8",
    )

    rows, *_ = build_rows([str(explanation_path)])

    assert rows[0]["target_provenance"] == "model_prediction"
    assert rows[0]["target_class"] == "class-c"
    assert rows[0]["target_class_id"] == 2


def test_sparsity_reports_fraction_excludable_at_requested_importance_mass():
    assert sparsity([9.0, 1.0, 0.0], mass=0.9) == 0.6667
    assert sparsity([], mass=0.9) == 0.0
