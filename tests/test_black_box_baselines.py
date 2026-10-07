"""Black-box baselines (clinical_gnn with --conv edge_conditioned | hgt | gchm) under
the GraphXAI runner: post-hoc explainers only, through PyG's own mask hook.

The point of these tests is the plan's "applied exactly once per edge": the mask
must reach every message, and must reach it once, not zero times and not twice.
"""
import inspect
import json
import types

import pytest
import torch

from core.explain import graphxai_standardized as gx
from core.explain import runner
from tests.test_explain_runner import train_one

CONVS = ("edge_conditioned", "hgt", "gchm")


def overrides(conv, **extra):
    values = {"method": "clinical_gnn", "conv": conv}
    if conv == "hgt":
        values["heads"] = 2
    values.update(extra)
    return values


@pytest.fixture(params=CONVS)
def run(request, tmp_path):
    conv = request.param
    run_dir = train_one(tmp_path, f"run_{conv}", **overrides(conv))
    bundle = runner.load_run_bundle(run_dir)
    binding = bundle.binding
    splits = runner.rebuild_validation_split(binding)
    model, is_adapter = runner.build_model(binding, bundle.state_dict)
    assert is_adapter is False
    return conv, run_dir, binding, splits, model, runner.build_wrapper(binding, model)


def test_wrapper_does_not_declare_edge_mask_so_the_explainer_uses_one_path_only(run):
    *_, wrapper = run
    assert "edge_mask" not in inspect.signature(wrapper.forward).parameters
    assert gx._model_declares_edge_mask(wrapper) is False


def test_the_view_carries_edge_payload_and_follows_an_edge_attr_override(run):
    _, _, _, splits, _, wrapper = run
    x = wrapper.set_context(runner._single_graph_batch(splits["validation"][0]))
    ei = wrapper._context.edge_index
    view = wrapper._batch_with(x, ei)
    assert torch.equal(view.edge_payload, view.edge_attr[:, -7:])
    zeroed = torch.zeros_like(view.edge_attr)
    assert torch.count_nonzero(wrapper._batch_with(x, ei, edge_attr=zeroed).edge_payload) == 0


def test_runner_explains_the_baseline_end_to_end(run, tmp_path):
    conv, run_dir, binding, *_ = run
    out = tmp_path / f"explain_{conv}"
    manifest = runner.run_explanations(run_dir, out, max_subjects=3, steps=2, epochs=3)
    assert manifest["replay_verified"] is True
    assert manifest["failed_count"] == 0, manifest["failures"]
    assert manifest["explained_count"] >= 1
    for name in manifest["record_files"]:
        record = json.loads((out / name).read_text())
        assert record["builtin_available"] is False
        assert record["wrapper_verification"]["max_abs_logit_diff"] <= 1e-5
        for algo in ("GradExplainer", "IntegratedGradExplainer"):
            assert record["graphxai"][algo]["status"] == "success"
        gnn = record["graphxai"]["GNNExplainer"]
        assert gnn["status"] == "success", gnn
        assert gnn["provenance"]["edge_gradient_verified"] is True
        assert gnn["provenance"]["edge_gradient_l1"] > 0
        assert "fidelity_plus" in record["fidelity_v2"]["GNNExplainer"]
        contribution = record["edge_attr_contribution"]
        if conv == "hgt":
            assert contribution["status"] == "not_applicable"
        else:
            assert contribution["edge_count"] == record["graphxai"]["GNNExplainer"]["provenance"]["edge_count"]


@pytest.mark.parametrize("run", ["edge_conditioned", "gchm"], indirect=True)
def test_an_explicit_removal_mask_is_applied_exactly_once(run):
    """Independent reference: scale every message by 0.5 by hand. If the wrapper's
    mask were applied twice the result would match a 0.25 reference instead."""
    _, _, _, splits, model, wrapper = run
    x = wrapper.set_context(runner._single_graph_batch(splits["validation"][0]))
    ei = wrapper._context.edge_index
    got = wrapper(x, ei, edge_mask=torch.full((ei.size(1),), 0.5)).detach()

    view = wrapper._batch_with(x, ei)
    originals = {}
    for layer in model.layers:
        originals[layer] = layer.message
        layer.message = (lambda orig: lambda *a, **k: 0.5 * orig(*a, **k))(layer.message)
    try:
        model.eval()
        reference = model(view).detach()
    finally:
        for layer, original in originals.items():
            layer.message = original
    assert torch.allclose(got, reference, atol=1e-5), (got, reference)
    # and it really is the 0.5 intervention, not a no-op
    assert not torch.allclose(got, wrapper(x, ei).detach(), atol=1e-6)


def test_hgt_mask_scales_messages_after_the_attention_softmax(tmp_path):
    run_dir = train_one(tmp_path, "run_hgt_mask", **overrides("hgt"))
    bundle = runner.load_run_bundle(run_dir)
    binding = bundle.binding
    splits = runner.rebuild_validation_split(binding)
    model, _ = runner.build_model(binding, bundle.state_dict)
    wrapper = runner.build_wrapper(binding, model)
    x = wrapper.set_context(runner._single_graph_batch(splits["validation"][0]))
    ei = wrapper._context.edge_index
    got = wrapper(x, ei, edge_mask=torch.full((ei.size(1),), 0.5)).detach()
    view = wrapper._batch_with(x, ei)
    for layer in model.layers:
        layer.message = (lambda orig: lambda *a, **k: 0.5 * orig(*a, **k))(layer.message)
    model.eval()
    assert torch.allclose(got, model(view).detach(), atol=1e-5)


def test_removing_all_message_passing_makes_gnnexplainer_unsupported_not_a_result(tmp_path):
    run_dir = train_one(tmp_path, "run_nomp", **overrides("edge_conditioned", no_message_passing=True))
    out = tmp_path / "explain_nomp"
    manifest = runner.run_explanations(run_dir, out, max_subjects=2, steps=2, epochs=3)
    assert manifest["failed_count"] == 0, manifest["failures"]
    for name in manifest["record_files"]:
        record = json.loads((out / name).read_text())
        gnn = record["graphxai"]["GNNExplainer"]
        assert gnn["status"] == "unsupported" and "node_explanation" not in gnn
        assert "GNNExplainer" not in record["fidelity_v2"]
