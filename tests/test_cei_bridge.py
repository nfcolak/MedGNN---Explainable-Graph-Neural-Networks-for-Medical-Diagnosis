"""CEI-GNN v3 under the clinical_graph_v2 explanation runner (the bridge), plus the
dev-fold and CLI-default changes the bridge needed.

The model under test is his real EvidenceAdapterV3 on his synthetic graph (fixtures from
tests/test_cei_gnn_v3_core.py), not a stand-in.
"""
import copy
import inspect
import json
import sys

import pytest
import torch
from torch_geometric.data import Batch

from core.explain import graphxai_standardized as gx
from core.explain import ig_clinical, runner
from core.explain.cei_bridge import CEIGraphXAIWrapper
from tests.test_cei_gnn_v3_core import (
    CLASSES, EDGE_DIM, HIDDEN, LAYOUT, NODE_DIM, NUM_TOKENS, RELATIONS, TOKEN_DIM, TRIPLES,
    _adapter, _graph,
)
from tests.test_explain_runner import train_one


@pytest.fixture
def bridge(tmp_path):
    torch.manual_seed(5)
    adapter = _adapter(tmp_path).eval()
    wrapper = CEIGraphXAIWrapper(adapter)
    batch = Batch.from_data_list([_graph()])
    return adapter, wrapper, batch, wrapper.set_context(batch)


def test_wrapper_reproduces_the_adapter_and_explains_his_continuous_row(bridge):
    adapter, wrapper, batch, x = bridge
    ok, diff = wrapper.verify(x, batch.edge_index)
    assert ok and diff < 1e-6
    assert x.shape == (8, NODE_DIM + TOKEN_DIM + HIDDEN)      # numeric | token vectors | type vectors


def test_forward_declares_no_edge_mask_so_the_explainer_has_one_delivery_path(bridge):
    _, wrapper, _, _ = bridge
    assert "edge_mask" not in inspect.signature(wrapper.forward).parameters
    assert gx._model_declares_edge_mask(wrapper) is False


@pytest.mark.parametrize("value", [0.5, 0.0])
def test_a_removal_mask_matches_his_own_public_hook_exactly_once(bridge, value):
    """Independent reference: his network's public set_edge_mask. If our PyG-hook delivery
    applied the mask twice, 0.5 would match a 0.25 reference instead."""
    adapter, wrapper, batch, x = bridge
    edges = batch.edge_index
    mask = torch.full((edges.size(1),), value)
    got = wrapper(x, edges, edge_mask=mask).detach()

    reference = copy.deepcopy(adapter)
    reference.network.set_edge_mask(mask)
    expected = reference.forward_continuous(x, edges, batch).detach()
    # his wrapper holds the PLE term fixed at its original value; the adapter recomputes
    # it from the (unchanged) features, so the two agree
    assert torch.allclose(got, expected, atol=1e-6), (got, expected)
    assert not torch.allclose(got, wrapper(x, edges).detach(), atol=1e-6)    # the mask does something


def test_the_builtin_is_his_exact_additive_accounting(bridge):
    adapter, wrapper, batch, x = bridge
    importance = wrapper.builtin_node_importance(x, batch.edge_index)
    detail = wrapper.builtin_detail()
    prediction = int(wrapper(x, batch.edge_index).argmax(-1))
    assert importance.shape == (8,) and (importance >= 0).all()
    assert detail["explained_class"] == prediction
    assert detail["accounting_logit_abs_diff"] < 1e-4          # contributions sum to the logit
    other = (prediction + 1) % CLASSES
    redirected = wrapper.builtin_node_importance(x, batch.edge_index, target_class=other)
    assert wrapper.builtin_detail()["explained_class"] == other
    assert not torch.allclose(importance, redirected)


def test_gnnexplainer_reaches_cei_and_leaves_no_mask_state_behind(bridge):
    _, wrapper, batch, x = bridge
    result = gx.explain_algorithms(wrapper, x, batch.edge_index,
                                   batch=torch.zeros(8, dtype=torch.long), steps=2, epochs=3)
    assert all(result[a]["status"] == "success" for a in gx.ALGORITHMS), result
    assert result["GNNExplainer"]["provenance"]["edge_gradient_verified"] is True
    assert result["GNNExplainer"]["provenance"]["edge_gradient_l1"] > 0
    assert not wrapper.inner.network.edge_aggregator.explain


def test_ig_baseline_keeps_kind_and_type_vectors_zeroes_numeric_and_token_vectors(bridge):
    _, wrapper, batch, x = bridge
    spec = wrapper.ig_feature_spec()
    assert spec == {"numeric_width": NODE_DIM, "zeroed_tail": TOKEN_DIM, "kept_tail": HIDDEN}
    baseline = ig_clinical.not_recorded_baseline(x, batch.node_type, LAYOUT, **spec)
    assert torch.equal(baseline[:, -HIDDEN:], x[:, -HIDDEN:])                      # what the node IS
    assert torch.count_nonzero(baseline[:, :NODE_DIM + TOKEN_DIM]) == 0            # nothing recorded
    with pytest.raises(ValueError, match="columns"):
        ig_clinical.not_recorded_baseline(x, batch.node_type, LAYOUT)             # tails not declared


def test_the_whole_subject_pipeline_runs_on_cei(bridge):
    _, wrapper, batch, x = bridge
    data = _graph()
    data.sample_id, data.subject = "cei-1", "p1"
    ig = {"layout": LAYOUT, "steps": 4, "tolerance": 0.05, "step_selection": {}, "bank": None,
          "reference_draws": 0, "keep_zero_baseline": True, "seed": 1,
          "feature_spec": wrapper.ig_feature_spec()}
    record = runner.explain_subject(wrapper, data, method="cei_gnn_v3", conv=None, steps=2,
                                    epochs=3, num_relation_columns=RELATIONS, ig=ig,
                                    random_repeats=3)
    assert record["builtin_available"] is True
    assert record["builtin_detail"]["accounting_logit_abs_diff"] < 1e-4
    assert record["graphxai"]["IntegratedGradExplainer"]["provenance"]["baseline"] == "not_recorded"
    assert record["edge_attr_contribution"]["status"] == "not_applicable"
    assert record["wrapper_verification"]["max_abs_logit_diff"] <= 1e-5
    assert "vs_random" in record["fidelity_v2"]["builtin"]
    assert record["edge_dependence"]["edge_count"] == 6


def _binding(adapter, **override):
    return {"method": "cei_gnn_v3", "conv": None, "node_dim": NODE_DIM, "edge_dim": EDGE_DIM,
            "num_classes": CLASSES, "vocabulary_size": NUM_TOKENS, "num_relations": RELATIONS,
            "num_meta_relations": TRIPLES, "hidden": HIDDEN, "layers": 1,
            "edge_direction": "forward", "seed": 1234,
            "method_config": adapter.run_config()} | override


def test_build_model_rebuilds_cei_from_its_recorded_options(bridge):
    adapter, _, batch, _ = bridge
    model, is_adapter = runner.build_model(_binding(adapter), adapter.state_dict())
    assert is_adapter is True
    with torch.no_grad():
        assert torch.equal(model.eval()(batch, epoch=0).logits, adapter(batch, epoch=0).logits)


def test_build_model_refuses_a_cei_whose_recorded_configuration_does_not_match(bridge):
    adapter, *_ = bridge
    tampered = adapter.run_config()
    tampered["parameter_count"] += 1
    with pytest.raises(ValueError, match="configuration differs"):
        runner.build_model(_binding(adapter, method_config=tampered), adapter.state_dict())


# ------------------------------------------------------------------ fold + CLI defaults
def test_a_run_that_keeps_validation_closed_is_explained_on_its_dev_fold(tmp_path):
    run_dir = train_one(tmp_path, "run_dev_only", method="gsat", conv=None, final_eval="none")
    with pytest.raises(ValueError, match="selected_validation"):
        runner.run_explanations(run_dir, tmp_path / "ex_val", max_subjects=1, steps=2, epochs=3)
    out = tmp_path / "ex_dev"
    manifest = runner.run_explanations(run_dir, out, fold="dev", max_subjects=2, steps=2, epochs=3,
                                       ig_step_candidates=(4,), random_repeats=0)
    assert manifest["replay_verified"] and manifest["fold"] == "dev" and manifest["failed_count"] == 0
    cohort = json.loads((out / "cohort.json").read_text())
    assert cohort["fold"] == "dev"
    assert cohort["fold_sample_ids_sha256"] == manifest["fold_sample_ids_sha256"]


def test_the_dev_fold_replay_really_checks_the_dev_hash(tmp_path):
    run_dir = train_one(tmp_path, "run_dev_tamper", method="gsat", conv=None)
    binding_path = run_dir / "binding.json"
    binding = json.loads(binding_path.read_text())
    binding["selected_dev"]["prediction_sha256"] = "0" * 64
    binding_path.write_text(json.dumps(binding))
    with pytest.raises(ValueError, match="REPLAY CHECK FAILED"):
        runner.run_explanations(run_dir, tmp_path / "ex_tamper", fold="dev", max_subjects=1,
                                steps=2, epochs=3)


def test_cli_explains_500_patients_by_default_and_zero_means_everyone(monkeypatch, tmp_path):
    seen = {}

    def fake(run_dir, output, **kwargs):
        seen.update(kwargs)
        return {"method": "m", "conv": None, "explained_count": 0, "failed_count": 0}

    monkeypatch.setattr(runner, "run_explanations", fake)
    base = ["runner", "--run-dir", str(tmp_path), "--output", str(tmp_path / "o")]
    monkeypatch.setattr(sys, "argv", base)
    runner._cli()
    assert seen["sample_patients"] == runner.DEFAULT_EXPLAINED_PATIENTS == 500
    monkeypatch.setattr(sys, "argv", base + ["--sample-patients", "0", "--fold", "dev"])
    runner._cli()
    assert seen["sample_patients"] is None and seen["fold"] == "dev"
