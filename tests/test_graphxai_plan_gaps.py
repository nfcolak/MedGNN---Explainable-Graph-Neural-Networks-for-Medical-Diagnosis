"""GraphXAI plan: behaviours added after the first port.

- item 1: cohort contract v2 (validation/dev only, hash-pinned to the run),
  runner writes cohort.json, CEI-GNN is refused clearly
- item 2: the wrapper must reproduce the model on every graph before explaining
- item 3: an all-zero edge-mask derivative is "unsupported", not a result
- item 4: every record carries the versioned intervention definition
- item 5: the degree-normalised GNNExplainer reduction is opt-in, 'sum' stays default
"""
import json

import pytest
import torch

from core.contracts import sample_ids_sha256
from core.explain import fidelity_v2 as fv2
from core.explain import graphxai_standardized as gx
from core.explain import runner
from core.explain.explanation_contract_v2 import build_cohort_v2, validate_cohort_v2
from tests.test_explain_runner import train_one
from tests.test_real_graphxai import Tiny

IDS = ["s1", "s2", "s3", "s4"]


def binding_for(ids):
    return {"split_sample_ids_sha256": {"validation": sample_ids_sha256(ids),
                                        "dev": sample_ids_sha256(ids[:2])},
            "preprocessing_sha256": "pp"}


def test_cohort_v2_is_pinned_to_the_runs_fold_hash():
    binding = binding_for(IDS)
    cohort = build_cohort_v2(binding, fold="validation", fold_sample_ids=IDS,
                             explained_sample_ids=IDS)
    validate_cohort_v2(cohort, binding)
    assert cohort["schema_version"] == 2 and cohort["is_full_fold"] is True
    prefix = build_cohort_v2(binding, fold="validation", fold_sample_ids=IDS,
                             explained_sample_ids=IDS[:2])
    assert prefix["is_full_fold"] is False and prefix["explained_count"] == 2


def test_cohort_v2_never_accepts_the_test_fold():
    with pytest.raises(ValueError, match="test fold"):
        build_cohort_v2(binding_for(IDS), fold="test", fold_sample_ids=IDS,
                        explained_sample_ids=IDS)


def test_cohort_v2_rejects_a_fold_that_differs_from_the_runs_hash():
    with pytest.raises(ValueError, match="do not match"):
        build_cohort_v2(binding_for(IDS), fold="validation",
                        fold_sample_ids=IDS[::-1], explained_sample_ids=IDS[:1])


def test_cohort_v2_validate_detects_tampering():
    binding = binding_for(IDS)
    cohort = build_cohort_v2(binding, fold="validation", fold_sample_ids=IDS,
                             explained_sample_ids=IDS)
    cohort["fold_sample_ids_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fold hash"):
        validate_cohort_v2(cohort, binding)


@pytest.fixture
def trained(tmp_path):
    run_dir = train_one(tmp_path, "run_plan_gaps", method="gsat", conv=None)
    bundle = runner.load_run_bundle(run_dir)
    binding = bundle.binding
    splits = runner.rebuild_validation_split(binding)
    model, _ = runner.build_model(binding, bundle.state_dict)
    return run_dir, bundle, binding, splits, runner.build_wrapper(binding, model)


def test_runner_writes_a_v2_cohort_and_records_its_version(tmp_path, trained):
    run_dir, bundle, binding, _, _ = trained
    out = tmp_path / "explanations_cohort"
    manifest = runner.run_explanations(run_dir, out, max_subjects=2, steps=2, epochs=3)
    cohort = json.loads((out / manifest["cohort_file"]).read_text())
    validate_cohort_v2(cohort, binding)
    assert manifest["cohort_schema_version"] == 2
    assert manifest["cohort_is_full_fold"] == cohort["is_full_fold"]
    assert cohort["fold"] == "validation"


def test_runner_refuses_cei_clearly(trained):
    _, bundle, binding, _, _ = trained
    other = dict(binding, method="cei_gnn_v3")
    with pytest.raises(NotImplementedError, match="not supported by this runner"):
        runner.build_model(other, bundle.state_dict)


def test_a_wrapper_that_does_not_reproduce_the_model_blocks_the_explanation(trained):
    *_, splits, wrapper = trained
    wrapper.verify = lambda *args, **kwargs: (False, 0.5)
    with pytest.raises(ValueError, match="does not reproduce the model"):
        runner.explain_subject(wrapper, splits["validation"][0], method="gsat", conv=None,
                               steps=2, epochs=2, num_relation_columns=1)


def test_zero_edge_gradient_is_unsupported_not_a_result(monkeypatch, trained):
    _, _, binding, splits, wrapper = trained
    real = gx.explain_algorithms

    def zeroed(*args, **kwargs):
        out = real(*args, **kwargs)
        out["GNNExplainer"]["provenance"]["zero_predictive_edge_gradient"] = True
        return out

    monkeypatch.setattr(gx, "explain_algorithms", zeroed)
    record = runner.explain_subject(wrapper, splits["validation"][0], method="gsat", conv=None,
                                    steps=2, epochs=2, num_relation_columns=binding["num_relations"])
    gnn = record["graphxai"]["GNNExplainer"]
    assert gnn["status"] == "unsupported" and "unsupported_reason" in gnn
    assert "node_explanation" not in gnn
    assert "GNNExplainer" not in record["fidelity_v2"]
    assert record["graphxai"]["GradExplainer"]["status"] == "success"


def test_records_stamp_the_wrapper_check_and_the_intervention_version(trained):
    _, _, binding, splits, wrapper = trained
    record = runner.explain_subject(wrapper, splits["validation"][0], method="gsat", conv=None,
                                    steps=2, epochs=2, num_relation_columns=binding["num_relations"])
    assert record["wrapper_verification"]["max_abs_logit_diff"] <= 1e-5
    assert record["fidelity_v2_intervention"]["version"] == fv2.INTERVENTION_CONTRACT_VERSION


def _gnn_importance(**kwargs):
    torch.manual_seed(1234)
    model = Tiny().eval()
    x = torch.tensor([[1., 2.], [3., 1.], [2., 4.]])
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    out = gx.explain_algorithms(model, x, edges, batch=torch.zeros(3, dtype=torch.long),
                                steps=2, epochs=3, **kwargs)["GNNExplainer"]
    return out["provenance"]["node_reduction"], torch.tensor(out["node_explanation"]["node_importance"])


def test_degree_normalised_reduction_is_opt_in_and_sum_stays_default():
    default_name, default_imp = _gnn_importance()
    mean_name, mean_imp = _gnn_importance(node_reduction="mean")
    assert default_name == "sum_incident_continuous_edge_mask"
    assert mean_name == "mean_incident_continuous_edge_mask"
    degree = torch.tensor([2., 4., 2.])   # incident edge endpoints per node in this graph
    assert torch.allclose(mean_imp * degree, default_imp, atol=1e-5)


def test_unknown_node_reduction_is_rejected():
    with pytest.raises(ValueError, match="node_reduction"):
        _gnn_importance(node_reduction="median")


def test_protgnn_runner_reports_the_models_own_subgraph_or_says_why_not(tmp_path):
    run_dir = train_one(tmp_path, "run_protgnn_builtin", method="protgnn", conv=None)
    out = tmp_path / "explanations_protgnn"
    manifest = runner.run_explanations(run_dir, out, max_subjects=4, steps=2, epochs=3)
    assert manifest["method"] == "protgnn" and manifest["failed_count"] == 0, manifest["failures"]
    for name in manifest["record_files"]:
        record = json.loads((out / name).read_text())
        if record["builtin_available"]:
            detail = record["builtin_detail"]
            members = [i for i, v in enumerate(record["builtin_node_importance"]) if v == 1.0]
            assert detail["kind"] == "prototype_mcts_subgraph"
            assert members == detail["subgraph_nodes"]
            assert detail["predicted_class"] == record["prediction_class_id"]
            assert "fidelity_plus" in record["fidelity_v2"]["builtin"]
        else:
            assert record["builtin_unavailable_reason"]
            assert record["builtin_detail"] is None
