"""GraphXAI plan, Phase B: measurement fixes.

B1 eligible evidence nodes, support cap, sparsity   B2 random-removal reference
B3 clinical IG baseline + completeness              B4 weight-randomisation / stability
B5 edge dependence                                  B6 binding.json + dated dirs
B7 AUROC/AUPRC                                      sampling + true-class explanations

Where a check could pass vacuously it uses a case with a known answer (an oracle
ordering, a linear model whose IG is exact, a model whose weights we know changed).
"""
import json
import types

import numpy as np
import pytest
import torch

from core import NODE_KINDS, discrimination
from core.contracts import sample_ids_sha256
from core.explain import audit, fidelity_v2 as fv2, ig_clinical, runner
from core.explain import graphxai_standardized as gx
from core.schema import sha256
from tests.test_explain_runner import train_one
from tests.test_fidelity_v2 import WRAPPERS, seven_node_graph
from tests.test_real_graphxai import Tiny


# ------------------------------------------------------------------ B1 eligibility / cap / sparsity
def _wrapped(name="gsat"):
    graph = seven_node_graph()
    wrapper, x = WRAPPERS[name](graph)
    with torch.no_grad():
        target = int(wrapper(x, graph.edge_index).argmax(-1))
    return graph, wrapper, x, target


def test_k_is_capped_at_the_explanations_own_support():
    graph, wrapper, x, target = _wrapped()
    importance = np.zeros(7)
    importance[[2, 3]] = [1.0, 0.5]          # 2 of the 3 eligible nodes (2 complaint, 3 measurement, 5 vital)
    plus = fv2.fidelity_plus_v2(wrapper, x, graph.token, graph.edge_index, graph.node_type,
                                importance, target, k=3)
    assert plus["k"] == 2 and plus["k_requested"] == 3 and plus["support_count"] == 2


def test_a_dense_explanation_is_not_capped():
    graph, wrapper, x, target = _wrapped()
    plus = fv2.fidelity_plus_v2(wrapper, x, graph.token, graph.edge_index, graph.node_type,
                                np.full(7, 0.3), target, k=3)
    assert plus["k"] == 3 == plus["k_requested"]


def test_curve_spans_the_support_and_reports_a_width_normalised_change():
    graph, wrapper, x, target = _wrapped()
    importance = np.zeros(7)
    importance[[2, 5]] = [1.0, 0.7]
    curve = fv2.deletion_curve(wrapper, x, graph.token, graph.edge_index, graph.node_type,
                               importance, target, num_points=3)
    assert max(curve["k_values"]) <= 2 and curve["support_count"] == 2
    assert curve["explained_fraction"] == pytest.approx(max(curve["k_values"]) / 3)
    assert curve["mean_change"] == pytest.approx(curve["auc"] / curve["explained_fraction"])


def test_sparsity_is_computed_on_eligible_nodes_only():
    node_type = seven_node_graph().node_type            # eligible: nodes 2, 3, 5
    spread = np.array([9.0, 9.0, 1.0, 1.0, 9.0, 1.0, 9.0])   # protected nodes are loud but irrelevant
    result = fv2.sparsity_over_eligible(node_type, spread)
    assert result["eligible_node_count"] == 3 and result["support_fraction"] == 1.0
    concentrated = np.array([0, 0, 10.0, 0, 0, 0.0, 0])
    sparse = fv2.sparsity_over_eligible(node_type, concentrated)
    assert sparse["support_count"] == 1 and sparse["mass90_sparsity"] > result["mass90_sparsity"]


# ------------------------------------------------------------------ B2 random reference
def test_random_reference_uses_the_same_removal_path_so_full_removal_matches_exactly():
    graph, wrapper, x, target = _wrapped()
    reference = fv2.random_reference(wrapper, x, graph.token, graph.edge_index, graph.node_type,
                                     target, [1, 3], repeats=6, seed=7)
    everything = fv2.fidelity_plus_v2(wrapper, x, graph.token, graph.edge_index, graph.node_type,
                                      np.full(7, 1.0), target, k=3)["prob"]
    # at k = all eligible nodes, the order cannot matter: every random draw must equal it
    assert np.allclose(reference.deletion[3], everything, atol=1e-6)


def test_an_oracle_explanation_beats_the_random_average():
    graph, wrapper, x, target = _wrapped()
    args = (wrapper, x, graph.token, graph.edge_index, graph.node_type)
    single = {}
    for node in (2, 3, 5):
        importance = np.zeros(7); importance[node] = 1.0
        single[node] = fv2.fidelity_plus_v2(*args, importance, target, k=1)["prob"]
    best = max(single, key=single.get)
    oracle = np.full(7, 1e-3); oracle[best] = 1.0
    reference = fv2.random_reference(*args, target, [1, 2, 3], repeats=40, seed=3)
    plus = fv2.fidelity_plus_v2(*args, oracle, target, k=1)
    comparison = reference.compare(
        fidelity_plus=plus,
        fidelity_minus=fv2.fidelity_minus_v2(*args, oracle, target, k=1),
        deletion_curve=fv2.deletion_curve(*args, oracle, target, num_points=3),
        insertion_curve=fv2.insertion_curve(*args, oracle, target, num_points=3))
    assert comparison["fidelity_plus"]["difference"] >= -1e-9   # the best single node >= the average one
    assert 0.0 <= comparison["fidelity_plus"]["beats_random_fraction"] <= 1.0


def test_comparing_at_a_k_the_reference_never_computed_is_refused_clearly():
    graph, wrapper, x, target = _wrapped()
    args = (wrapper, x, graph.token, graph.edge_index, graph.node_type)
    reference = fv2.random_reference(*args, target, [1], repeats=3, seed=1)
    importance = np.full(7, 0.4)
    with pytest.raises(ValueError, match="not computed at k="):
        reference.compare(
            fidelity_plus=fv2.fidelity_plus_v2(*args, importance, target, k=2),
            fidelity_minus=fv2.fidelity_minus_v2(*args, importance, target, k=1),
            deletion_curve=fv2.deletion_curve(*args, importance, target, num_points=3),
            insertion_curve=fv2.insertion_curve(*args, importance, target, num_points=3))


def test_random_reference_is_deterministic_in_its_seed():
    graph, wrapper, x, target = _wrapped()
    args = (wrapper, x, graph.token, graph.edge_index, graph.node_type, target, [1, 2])
    a = fv2.random_reference(*args, repeats=5, seed=11)
    b = fv2.random_reference(*args, repeats=5, seed=11)
    c = fv2.random_reference(*args, repeats=5, seed=12)
    assert all(np.array_equal(a.deletion[k], b.deletion[k]) for k in a.ks)
    assert any(not np.array_equal(a.deletion[k], c.deletion[k]) for k in a.ks)


# ------------------------------------------------------------------ B3 clinical IG
LAYOUT = list(NODE_KINDS) + ["scaled_value", "has_value", "context.age", "context.age.has_value",
                              "context.gender:<missing>", "context.gender:<unknown>",
                              "context.gender=\"F\""]


def _x_for_layout():
    x = torch.arange(1, 3 * len(LAYOUT) + 1, dtype=torch.float32).view(3, len(LAYOUT))
    node_type = torch.tensor([NODE_KINDS.index("patient"), NODE_KINDS.index("visit"),
                              NODE_KINDS.index("vital")])
    return x, node_type


def test_not_recorded_baseline_keeps_kind_and_clears_everything_recorded():
    x, node_type = _x_for_layout()
    baseline = ig_clinical.not_recorded_baseline(x, node_type, LAYOUT)
    kinds = slice(0, len(NODE_KINDS))
    assert torch.equal(baseline[:, kinds], x[:, kinds])
    value = LAYOUT.index("scaled_value"), LAYOUT.index("has_value")
    assert torch.count_nonzero(baseline[:, list(value)]) == 0
    missing = LAYOUT.index("context.gender:<missing>")
    assert baseline[0, missing] == 1.0 and baseline[1, missing] == 0 and baseline[2, missing] == 0
    assert baseline[0, LAYOUT.index("context.gender=\"F\"")] == 0


def test_baseline_layout_must_match_the_feature_width():
    x, node_type = _x_for_layout()
    with pytest.raises(ValueError, match="columns"):
        ig_clinical.not_recorded_baseline(x, node_type, LAYOUT[:-1])


class _Linear(torch.nn.Module):
    def forward(self, x, edge_index, **kwargs):
        return torch.stack([x[:, 0].sum() + 2 * x[:, 1].sum(),
                            x[:, 1].sum() - x[:, 0].sum()]).view(1, 2)


class _Quadratic(torch.nn.Module):
    def forward(self, x, edge_index, **kwargs):
        return torch.stack([(x ** 2).sum(), (x ** 3).sum()]).view(1, 2)


def _tiny_inputs():
    x = torch.tensor([[1.0, 2.0], [-1.0, 0.5], [3.0, -1.5]])
    return x, torch.zeros_like(x), torch.empty((2, 0), dtype=torch.long)


def test_ig_is_exact_on_a_linear_model_and_zero_when_nothing_changes():
    x, baseline, edges = _tiny_inputs()
    attribution, completeness = ig_clinical.integrated_gradients(_Linear(), x, edges, baseline, 0, 4)
    assert completeness["completeness_error_abs"] < 1e-5
    none, _ = ig_clinical.integrated_gradients(_Linear(), x, edges, x.clone(), 0, 4)
    assert torch.count_nonzero(none) == 0


def test_completeness_error_shrinks_with_steps_on_a_curved_model():
    x, baseline, edges = _tiny_inputs()
    errors = {s: ig_clinical.integrated_gradients(_Quadratic(), x, edges, baseline, 1, s)[1]
              ["completeness_error_rel"] for s in (2, 8, 64)}
    assert errors[64] < errors[8] < errors[2]


def test_step_choice_is_driven_by_the_completeness_error():
    picked = ig_clinical.choose_steps({8: [0.2, 0.01], 16: [0.06, 0.02], 32: [0.04, 0.01]}, 0.05)
    assert picked["steps"] == 32 and picked["completeness_met"]   # 16 fails on its worst graph
    unmet = ig_clinical.choose_steps({8: [0.5], 16: [0.4]}, 0.05)
    assert unmet["steps"] == 16 and unmet["completeness_met"] is False


def test_reference_bank_draws_real_rows_of_the_same_token_and_kind_else_falls_back():
    graph = types.SimpleNamespace(
        x=torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        token=torch.tensor([5, 5, 6]), node_type=torch.tensor([5, 5, 5]))
    bank = ig_clinical.ReferenceBank.from_graphs([graph])
    fallback = torch.full((2, 2), -1.0)
    rng = np.random.default_rng(0)
    drawn = bank.draw(torch.tensor([5, 9]), torch.tensor([5, 5]), fallback, rng)
    assert drawn[0].tolist() in ([1.0, 1.0], [2.0, 2.0])      # a real token-5 row
    assert drawn[1].tolist() == [-1.0, -1.0]                  # unknown token: fallback


# ------------------------------------------------------------------ B4 audit controls
def test_weight_randomisation_changes_every_weight_leaves_buffers_and_original_alone():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4))
    model.register_buffer("structure", torch.arange(3))
    before = [p.detach().clone() for p in model.parameters()]
    clone = audit.randomise_weights(model, seed=5)
    assert all(not torch.equal(a, b) for a, b in zip(before, clone.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert torch.equal(clone.structure, model.structure)
    again = audit.randomise_weights(model, seed=5)
    assert all(torch.equal(a, b) for a, b in zip(clone.parameters(), again.parameters()))


def test_agreement_identifies_same_reversed_and_constant_rankings():
    node_type = seven_node_graph().node_type
    a = np.array([0, 0, 3.0, 2.0, 0, 1.0, 0])
    same = audit.agreement(a, a, node_type)
    assert same["spearman"] == pytest.approx(1.0) and same["top_k_jaccard"] == 1.0
    reversed_ = np.array([0, 0, 1.0, 2.0, 0, 3.0, 0])
    assert audit.agreement(a, reversed_, node_type)["spearman"] == pytest.approx(-1.0)
    assert audit.agreement(a, np.ones(7), node_type)["spearman"] is None


def test_perturbation_touches_only_measured_values():
    x = torch.zeros(4, len(NODE_KINDS) + 2)
    x[:, audit.HAS_VALUE_COLUMN] = torch.tensor([1.0, 0.0, 1.0, 0.0])
    perturbed, measured = audit.perturb_values(x, 0.5, seed=1)
    changed = (perturbed != x).any(dim=1)
    assert measured == 2 and changed.tolist() == [True, False, True, False]
    assert torch.equal(perturbed[:, :audit.VALUE_COLUMN], x[:, :audit.VALUE_COLUMN])


# ------------------------------------------------------------------ B5 edge dependence
def test_edge_dependence_is_zero_for_a_model_that_never_reads_edges_and_reports_edge_count():
    class NoEdges(torch.nn.Module):
        def forward(self, x, edge_index, batch=None, edge_mask=None, **kw):
            return x.sum(dim=0, keepdim=True)[:, :2]

    x = torch.tensor([[1.0, 2.0], [3.0, 1.0]])
    edges = torch.tensor([[0, 1], [1, 0]])
    result = audit.edge_dependence(NoEdges(), x, edges, 0)
    assert result["probability_drop"] == 0 and result["edge_count"] == 2
    assert result["prediction_unchanged_without_edges"] is True
    assert audit.edge_dependence(NoEdges(), x, torch.empty((2, 0), dtype=torch.long), 0)["status"] == "not_applicable"


# ------------------------------------------------------------------ B7 AUROC / AUPRC
def test_discrimination_is_one_for_perfect_scores_and_chance_for_uninformative_ones():
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    perfect = np.eye(3)[y]
    report = discrimination.discrimination_report(perfect, y)
    assert report["macro"]["auroc"] == pytest.approx(1.0) and report["macro"]["auprc"] == pytest.approx(1.0)
    flat = discrimination.discrimination_report(np.full((9, 3), 1 / 3), y)
    assert flat["macro"]["auroc"] == pytest.approx(0.5)


def test_undefined_classes_are_excluded_not_averaged_in():
    y = np.array([0, 1, 0, 1])
    proba = np.array([[.9, .1, 0], [.2, .8, 0], [.7, .3, 0], [.4, .6, 0]])
    report = discrimination.discrimination_report(proba, y)
    assert report["classes_undefined"] == [2] and report["classes_defined"] == [0, 1]


def test_patient_equal_weights_give_every_patient_the_same_total():
    subjects = np.array(["a", "a", "a", "b", "c", "c"])
    weights = discrimination.patient_equal_weights(subjects)
    totals = {p: weights[subjects == p].sum() for p in "abc"}
    assert all(t == pytest.approx(1.0) for t in totals.values())


def test_run_report_reads_a_real_runs_saved_probabilities(tmp_path):
    run_dir = train_one(tmp_path, "run_auc", method="gsat", conv=None)
    report = discrimination.run_report(run_dir)
    assert 0.0 <= report["macro"]["auroc"] <= 1.0 and "auroc_patient_equal" in report["macro"]


def test_discrimination_cli_refuses_to_overwrite(tmp_path, monkeypatch):
    run_dir = train_one(tmp_path, "run_auc2", method="gsat", conv=None)
    out = tmp_path / "auc.json"
    monkeypatch.setattr("sys.argv", ["x", "--run", str(run_dir), "--out", str(out)])
    discrimination.main()
    assert out.exists()
    with pytest.raises(FileExistsError):
        discrimination.main()


# ------------------------------------------------------------------ sampling
def _visits(assignments):
    return [types.SimpleNamespace(subject=s, sample_id=f"{s}-{i}") for i, s in enumerate(assignments)]


def test_patient_sampling_is_order_independent_complete_per_patient_and_seeded():
    visits = _visits(["p1", "p1", "p2", "p3", "p3", "p3", "p4", "p5"])
    chosen = runner.select_explained_subjects(visits, sample_patients=3, sample_seed=1)
    shuffled = runner.select_explained_subjects(visits[::-1], sample_patients=3, sample_seed=1)
    assert {d.subject for d in chosen} == {d.subject for d in shuffled}
    patients = {d.subject for d in chosen}
    assert len(patients) == 3
    assert [d.sample_id for d in chosen] == [d.sample_id for d in visits if d.subject in patients]
    draws = {frozenset(d.subject for d in runner.select_explained_subjects(
        visits, sample_patients=3, sample_seed=seed)) for seed in range(12)}
    assert len(draws) > 1                                       # the seed really changes the draw
    assert len(runner.select_explained_subjects(visits, sample_patients=3, max_subjects=2)) == 2


# ------------------------------------------------------------------ runner-level, real trained models
@pytest.fixture
def trained(tmp_path):
    run_dir = train_one(tmp_path, "run_phase_b", method="gsat", conv=None)
    bundle = runner.load_run_bundle(run_dir)
    binding = bundle.binding
    splits = runner.rebuild_validation_split(binding)
    model, _ = runner.build_model(binding, bundle.state_dict)
    return run_dir, bundle, binding, splits, model, runner.build_wrapper(binding, model)


def test_records_carry_sparsity_random_calibration_and_edge_dependence(tmp_path, trained):
    run_dir, *_ = trained
    out = tmp_path / "ex_b"
    manifest = runner.run_explanations(run_dir, out, max_subjects=3, steps=2, epochs=3,
                                       ig_step_candidates=(2, 4), random_repeats=4)
    assert manifest["failed_count"] == 0, manifest["failures"]
    for name in manifest["record_files"]:
        record = json.loads((out / name).read_text())
        assert set(record["fidelity_v2_intervention"]["eligible_node_kinds"]) == {
            "complaint", "vital", "measurement", "diagnosis"}
        assert record["random_reference"]["repeats"] == 4
        assert record["edge_dependence"]["edge_count"] > 0
        for source, metrics in record["fidelity_v2"].items():
            assert "sparsity" in metrics, source
            assert 0.0 <= metrics["vs_random"]["fidelity_plus"]["beats_random_fraction"] <= 1.0
        assert "builtin" in record["fidelity_v2"]          # gsat has a built-in explanation


def test_clinical_ig_replaces_the_zero_baseline_and_keeps_it_for_comparison(tmp_path, trained):
    run_dir, *_ = trained
    out = tmp_path / "ex_ig"
    manifest = runner.run_explanations(run_dir, out, max_subjects=2, steps=2, epochs=3,
                                       ig_step_candidates=(2, 4, 8), random_repeats=0)
    record = json.loads((out / manifest["record_files"][0]).read_text())
    ig = record["graphxai"]["IntegratedGradExplainer"]["provenance"]
    assert ig["baseline"] == "not_recorded" and ig["steps"] in (2, 4, 8)
    assert {"completeness_error_rel", "completeness_met", "target_logit_change"} <= set(ig)
    assert "IntegratedGradExplainerZeroBaseline" in record["graphxai"]
    assert manifest["clinical_ig"]["steps"] == ig["steps"]
    assert manifest["clinical_ig"]["step_selection"]["worst_relative_error_by_steps"]


def test_real_patient_reference_ig_is_an_extra_source_when_asked_for(tmp_path, trained):
    run_dir, *_ = trained
    out = tmp_path / "ex_refs"
    manifest = runner.run_explanations(run_dir, out, max_subjects=2, steps=2, epochs=3,
                                       ig_step_candidates=(4,), ig_reference_draws=3,
                                       random_repeats=0)
    record = json.loads((out / manifest["record_files"][0]).read_text())
    refs = record["graphxai"]["IntegratedGradExplainerRealPatientRefs"]
    assert refs["status"] == "success" and refs["provenance"]["draws"] == 3


def test_audits_run_and_the_randomised_model_really_changes_the_explanation(tmp_path, trained):
    run_dir, *_ = trained
    out = tmp_path / "ex_audit"
    manifest = runner.run_explanations(run_dir, out, max_subjects=3, steps=2, epochs=3,
                                       ig_step_candidates=(4,), random_repeats=0,
                                       audit_subjects=2, stability_draws=1, stability_sigma=0.0)
    audited = [json.loads((out / n).read_text()) for n in manifest["record_files"]][:2]
    assert all("audit" in r for r in audited)
    assert "audit" not in json.loads((out / manifest["record_files"][2]).read_text())
    spearmans = []
    for record in audited:
        sources = record["audit"]["weight_randomisation"]["sources"]
        spearmans += [v["spearman"] for v in sources.values() if v.get("spearman") is not None]
        # sigma = 0 -> the "perturbed" graph IS the original: same explanation, by construction
        draw = record["audit"]["stability"]["draws"][0]
        assert draw["prediction_changed"] is False
        for name, comparison in draw["sources"].items():
            if name in ("GradExplainer", "builtin"):
                assert comparison["top_k_jaccard"] == 1.0, name
    assert spearmans and min(spearmans) < 0.999        # re-drawing every weight does change an explanation


def test_binding_json_pins_the_run_cohort_settings_and_code(tmp_path, trained):
    run_dir, bundle, binding, *_ = trained
    out = tmp_path / "ex_binding"
    runner.run_explanations(run_dir, out, max_subjects=2, steps=2, epochs=3,
                            ig_step_candidates=(4,), random_repeats=2)
    pinned = json.loads((out / "binding.json").read_text())
    assert pinned["model_run"]["checkpoint_sha256"] == sha256(run_dir / "best.pt")
    assert pinned["model_run"]["binding_json_sha256"] == sha256(run_dir / "binding.json")
    assert pinned["cohort"]["sha256"] == sha256(out / "cohort.json")
    assert pinned["records"]["manifest_sha256"] == sha256(out / "explanation_manifest.json")
    assert pinned["explainer_config"]["clinical_ig"]["baseline"] == "not_recorded"
    assert pinned["explainer_config"]["intervention_contract_version"] == fv2.INTERVENTION_CONTRACT_VERSION
    assert pinned["source_code"]


def test_dated_output_dirs_are_new_and_named_by_method_and_time(tmp_path):
    first = runner.dated_output_dir(tmp_path, "gsat")
    assert first.parent == tmp_path and first.name.startswith("explanations_gsat_") and not first.exists()
    assert runner.dated_output_dir(tmp_path, "clinical_gnn", "hgt").name.startswith("explanations_clinical_gnn_hgt_")


def test_patient_sampling_reaches_the_cohort_file(tmp_path, trained):
    run_dir, bundle, binding, splits, *_ = trained
    out = tmp_path / "ex_sample"
    runner.run_explanations(run_dir, out, sample_patients=2, steps=2, epochs=3,
                            ig_step_candidates=(4,), random_repeats=0)
    cohort = json.loads((out / "cohort.json").read_text())
    assert cohort["sampling"]["sample_patients"] == 2
    assert cohort["sampling"]["patients_explained"] <= 2 and cohort["is_full_fold"] in (True, False)


def test_a_misclassified_patient_also_gets_the_true_class_explained(trained):
    _, _, binding, splits, _, wrapper = trained
    data = splits["validation"][0]
    batch = runner._single_graph_batch(data)
    x = wrapper.set_context(batch)
    with torch.no_grad():
        prediction = int(wrapper(x, batch.edge_index).argmax(-1))
    other = (prediction + 1) % binding["num_classes"]
    data.y = torch.tensor([other])                         # pretend the truth was another class
    record = runner.explain_subject(wrapper, data, method="gsat", conv=None, steps=2, epochs=3,
                                    num_relation_columns=binding["num_relations"],
                                    random_repeats=2)
    block = record["true_class_explanations"]
    assert block["target_class"] == other and record["explained_target"] == "predicted_class"
    assert block["graphxai"]["GNNExplainer"]["status"] == "unsupported"
    assert block["graphxai"]["GradExplainer"]["status"] == "success"
    assert "node_explanation" not in block["graphxai"]["GradExplainer"]   # v1 block targets the prediction
    assert all(m["fidelity_plus"]["target_class"] == other
               for m in block["fidelity_v2"].values() if "fidelity_plus" in m)
    data.y = torch.tensor([prediction])                    # a correct patient gets no such block
    assert "true_class_explanations" not in runner.explain_subject(
        wrapper, data, method="gsat", conv=None, steps=2, epochs=3,
        num_relation_columns=binding["num_relations"], random_repeats=0)


def test_explain_algorithms_can_target_another_class_except_gnnexplainer():
    torch.manual_seed(1234)
    model = Tiny().eval()
    x = torch.tensor([[1., 2.], [3., 1.], [2., 4.]])
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    batch = torch.zeros(3, dtype=torch.long)
    with torch.no_grad():
        predicted = int(model(x, edges, batch=batch).argmax(-1))
    other = 1 - predicted
    default = gx.explain_algorithms(model, x, edges, batch=batch, steps=2, epochs=3)
    redirected = gx.explain_algorithms(model, x, edges, batch=batch, steps=2, epochs=3, target_class=other)
    assert redirected["GNNExplainer"]["status"] == "unsupported"
    assert redirected["GradExplainer"]["provenance"]["target"] == f"class_{other}_logit"
    assert default["GradExplainer"]["node_explanation"]["node_importance"] != \
        redirected["GradExplainer"]["node_explanation"]["node_importance"]
    same = gx.explain_algorithms(model, x, edges, batch=batch, steps=2, epochs=3, target_class=predicted)
    assert same["GNNExplainer"]["status"] == "success"


def test_protgnn_builtin_can_explain_a_chosen_class(tmp_path):
    run_dir = train_one(tmp_path, "run_protgnn_cls", method="protgnn", conv=None)
    bundle = runner.load_run_bundle(run_dir)
    splits = runner.rebuild_validation_split(bundle.binding)
    model, _ = runner.build_model(bundle.binding, bundle.state_dict)
    wrapper = runner.build_wrapper(bundle.binding, model)
    wrapper.set_context(runner._single_graph_batch(splits["validation"][0]))
    classes = bundle.binding["num_classes"]
    for target in range(min(classes, 3)):
        detail = wrapper.adapter.explain_detail(wrapper._raw_context, target)
        assert detail["explained_class"] == target
        assert int(model.prototype_class_ids[detail["prototype_index"]]) == target
