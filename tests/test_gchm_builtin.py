"""GCHM-PNA v3's built-in node-level explanation (core/explain/gchm_builtin.py).

A freshly built v3 has zero-initialised label-wise and wide weights, which would make every
contribution zero and every test vacuous, so each test randomises them first. Where
possible the reference is Necati's own code (his `labelwise_logits`, `wide_logits`,
`forward`), not a re-derivation of ours.
"""
import json

import pytest
import torch

from core.explain import gchm_builtin, runner
from core.explain.graphxai_wrapper import ClinicalGraphXAIWrapper
from gchm_pna.gchm_v3 import GCHMv3
from tests.test_explain_runner import train_one


def v3_overrides(**extra):
    return {"method": "clinical_gnn", "conv": "gchm_v3", **extra}


@pytest.fixture
def v3(tmp_path):
    run_dir = train_one(tmp_path, "run_v3_builtin", **v3_overrides())
    bundle = runner.load_run_bundle(run_dir)
    binding = bundle.binding
    splits = runner.rebuild_validation_split(binding)
    model, is_adapter = runner.build_model(binding, bundle.state_dict)
    assert is_adapter is False and isinstance(model, GCHMv3)
    torch.manual_seed(11)
    with torch.no_grad():
        for parameter in (model.class_weight, model.wide_token.weight, model.wide_value.weight):
            parameter.copy_(torch.randn_like(parameter) * 0.5)
        model.wide_token.weight[0].zero_()          # padding row stays zero, as in training
        model.wide_value.weight[0].zero_()
    data = runner._single_graph_batch(splits["validation"][0])
    return run_dir, binding, splits, model.eval(), data


def test_the_parts_sum_back_to_the_models_logit_and_match_his_own_functions(v3):
    *_, model, data = v3
    importance, detail = gchm_builtin.explain(model, data)
    with torch.no_grad():
        h, batch, _, size = model.node_states(data)
        target = detail["explained_class"]
        his_labelwise = float(model.labelwise_logits(h, batch, size)[0, target])    # includes the bias
        his_wide = float(model.wide_logits(data, batch, size)[0, target])
        his_logit = float(model(data)[0, target])
    assert detail["labelwise_node_total"] == pytest.approx(his_labelwise, abs=1e-5)
    assert detail["wide_node_total"] == pytest.approx(his_wide, abs=1e-5)
    assert detail["logit"] == pytest.approx(his_logit, abs=1e-6)
    assert (detail["graph_level_head_logit"] + detail["labelwise_node_total"]
            + detail["wide_node_total"]) == pytest.approx(his_logit, abs=1e-4)
    assert detail["accounting_logit_abs_diff"] < 1e-4
    # The point of the whole thing: the PER-NODE contributions add up to the node-additive part
    # of the logit. Reference is his two functions, not our own totals (his labelwise_logits
    # already includes the class bias, so the bias is added to our node sum before comparing).
    assert (sum(detail["signed_node_contributions"]) + detail["class_bias"]
            == pytest.approx(his_labelwise + his_wide, abs=1e-5))


def test_importance_is_the_absolute_signed_node_contribution(v3):
    *_, model, data = v3
    importance, detail = gchm_builtin.explain(model, data)
    signed = torch.tensor(detail["signed_node_contributions"])
    assert importance.shape == (data.x.size(0),) and torch.allclose(importance, signed.abs())
    assert importance.sum() > 0 and (signed < 0).any() and (signed > 0).any()   # real, signed, nontrivial
    assert detail["node_additive_part"] == pytest.approx(
        float(signed.sum()) + detail["class_bias"], abs=1e-5)
    assert len(detail["labelwise_attention"]) == importance.numel()
    assert sum(detail["labelwise_attention"]) == pytest.approx(1.0, abs=1e-5)   # a softmax over the nodes


def test_a_chosen_class_is_explained_and_changes_the_contributions(v3):
    *_, model, data = v3
    _, own = gchm_builtin.explain(model, data)
    other = (own["predicted_class"] + 1) % model.num_classes
    _, redirected = gchm_builtin.explain(model, data, target_class=other)
    assert redirected["explained_class"] == other and redirected["predicted_class"] == own["predicted_class"]
    assert redirected["signed_node_contributions"] != own["signed_node_contributions"]
    assert redirected["accounting_logit_abs_diff"] < 1e-4


def test_the_accounting_guard_refuses_when_the_decomposition_stops_matching_the_model(v3):
    *_, model, data = v3
    original = model.wide_logits
    model.wide_logits = lambda *a, **k: original(*a, **k) + 1.0       # the model drifts from the accounting
    with pytest.raises(RuntimeError, match="does not reproduce the model's logits"):
        gchm_builtin.explain(model, data)


def test_a_configuration_with_no_node_additive_path_has_no_builtin(v3):
    _, binding, _, model, data = v3
    pooled_only = GCHMv3(binding["vocabulary_size"], binding["node_dim"], binding["edge_dim"],
                         binding["num_classes"], num_relations=binding["num_relations"],
                         degree_histogram=model.degree_histogram, hidden=model.settings["hidden"],
                         readout="pool", wide=False)
    with pytest.raises(RuntimeError, match="cannot be assigned to nodes"):
        gchm_builtin.explain(pooled_only, data)


def test_explain_restores_the_models_training_mode(v3):
    *_, model, data = v3
    model.train()
    gchm_builtin.explain(model, data)
    assert model.training is True


def test_the_wrapper_exposes_it_for_v3_and_refuses_clearly_for_v2(v3, tmp_path):
    run_dir, binding, splits, model, data = v3
    wrapper = runner.build_wrapper(binding, model)
    assert isinstance(wrapper, ClinicalGraphXAIWrapper)
    x = wrapper.set_context(data)
    importance = wrapper.builtin_node_importance(x, data.edge_index)
    detail = wrapper.builtin_detail()
    assert detail["kind"] == "gchm_v3_node_additive_accounting"
    assert torch.allclose(importance, torch.tensor(detail["signed_node_contributions"]).abs())
    wrapper.set_context(data)                                          # a new graph forgets the old detail
    assert wrapper.builtin_detail() is None

    second = tmp_path / "second_run"
    second.mkdir()
    v2_dir = train_one(second, "run_v2_builtin", method="clinical_gnn", conv="gchm_v2")
    bundle = runner.load_run_bundle(v2_dir)
    v2_model, _ = runner.build_model(bundle.binding, bundle.state_dict)
    v2_wrapper = runner.build_wrapper(bundle.binding, v2_model)
    v2_splits = runner.rebuild_validation_split(bundle.binding)
    v2_x = v2_wrapper.set_context(runner._single_graph_batch(v2_splits["validation"][0]))
    with pytest.raises(NotImplementedError, match="no per-node built-in explanation"):
        v2_wrapper.builtin_node_importance(v2_x, v2_wrapper._context.edge_index)


def test_the_runner_records_it_and_scores_it_on_the_common_scale(v3, tmp_path):
    run_dir, *_ = v3
    out = tmp_path / "ex_v3_builtin"
    manifest = runner.run_explanations(run_dir, out, max_subjects=3, steps=2, epochs=3,
                                       ig_step_candidates=(4,), random_repeats=3)
    assert manifest["failed_count"] == 0, manifest["failures"]
    for name in manifest["record_files"]:
        record = json.loads((out / name).read_text())
        assert record["builtin_available"] is True
        assert record["builtin_detail"]["accounting_logit_abs_diff"] < 1e-4
        scored = record["fidelity_v2"]["builtin"]
        assert "sparsity" in scored and "vs_random" in scored
