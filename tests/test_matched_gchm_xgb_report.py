"""Synthetic artifact/unit tests only: no models, real data or test-fold scoring."""
from __future__ import annotations

import copy
import itertools
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from comparison.standardized.matched_gchm_xgb_v1 import report as r
from comparison.standardized.performance_review import class_weights


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class NoTestLabels:
    """Sentinel rejects even accidental test-row indexing."""
    def __init__(self, values, train, validation):
        self.values = values
        self.allowed = set(train) | set(validation)

    def __getitem__(self, index):
        index = np.asarray(index)
        assert index.ndim == 1
        assert set(index.tolist()) <= self.allowed, "test labels indexed"
        return self.values[index]


class FakeReference:
    def __init__(self, k=6):
        self.train = np.arange(sum(range(5, k+5)), dtype=np.int64)
        self.val = np.arange(len(self.train), len(self.train)+k*4, dtype=np.int64)
        train_y = np.repeat(np.arange(k), np.arange(5, k+5))
        y = np.tile(np.arange(k), 4)
        # A test sentinel exists in the artifact format but must never be read.
        ys = np.r_[train_y, y, -999]
        counts = np.resize(np.array([0, 1, 2, 4, 5, 9, 10, 12]), len(ys))
        self.arrays = {"y": NoTestLabels(ys, self.train, self.val),
                       "node_ptr": np.r_[0, np.cumsum(counts+1)]}
        self.fingerprint = r.PINNED_CONTRACT
        self.contract = {"labels": [f"class_{i}" for i in range(k)], "temporal_clean": False,
                         "input_sha256": "synthetic-input-sha", "source_bindings": {"fixture": "synthetic"},
                         "limitations": {"fixture": "synthetic-only"}}

    def fold(self, n):
        assert n in (0, 1), "test fold requested"
        return self.train if n == 0 else self.val


@pytest.fixture
def context():
    return r.cohort_context(FakeReference())


@pytest.fixture
def protocol():
    return r.read_json(r.PROTOCOL)


def probability_matrix(y, k, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(len(y), k)).astype(np.float32)
    logits[np.arange(len(y)), y] += 1.2
    return torch.softmax(torch.from_numpy(logits), dim=1).numpy(), logits


def common_binding(ref, ctx, policy, seed):
    return {"contract_sha256": ref.fingerprint, "seed": seed, "weight_policy": policy,
            "train_ordinals_sha256": ctx["train_ordinals_sha256"],
            "validation_ordinals_sha256": ctx["validation_ordinals_sha256"]}


def make_gchm(run, ref, ctx, policy="none", seed=1234):
    run.mkdir(parents=True)
    p, logits = probability_matrix(ctx["y"], len(ctx["labels"]), seed)
    metrics = r.multiclass_metrics(ctx["y"], p.argmax(1), p)
    source = run / "source_snapshot" / "old_training.py"
    source.parent.mkdir()
    source.write_text("# Historical source; no equivalent live path exists.\n")
    b = {**common_binding(ref, ctx, policy, seed), "method": "gchm", "epochs": 30, "limit": None,
         "loss": "ce" if policy == "none" else "sqrt_inverse", "selection": r.GCHM_SELECTION,
         "source_code": {"old_training.py": r.sha(source)}}
    np.savez(run / "cohort.npz", train_ordinals=ctx["train"], validation_ordinals=ctx["validation"])
    np.savez(run / "validation_000.npz", logits=logits, y=ctx["y"], ordinals=ctx["validation"])
    # Every epoch ties on rounded macro F1: first maximum must remain epoch zero.
    history = [{"epoch": i, "validation": metrics, "train_count": len(ctx["train"]),
                "validation_count": len(ctx["y"])} for i in range(30)]
    dump(run / "history.json", history)
    (run / "best.pt").write_bytes(b"not-a-pickle: checkpoint provenance only")
    proof = {"exact_logits": True, "selected_epoch": 0, "validation_count": len(ctx["y"]),
             "checkpoint_sha256": r.sha(run / "best.pt"), "test_evaluated": False}
    dump(run / "replay.json", proof)
    manifest = {"method": "gchm", "status": "completed", "test_evaluated": False,
                "contract_sha256": ref.fingerprint, "binding": b, "next_epoch": 30,
                "checkpoint_path": "best.pt", "metrics_path": "history.json", "replay": proof,
                "selected_counts": [len(ctx["train"]), len(ctx["y"])]}
    dump(run / "run_manifest.json", manifest)
    return p


def make_xgb(run, ref, ctx, protocol_path, policy="none", seed=1234, rounds=1200):
    run.mkdir(parents=True)
    p, _ = probability_matrix(ctx["y"], len(ctx["labels"]), seed+5)
    metrics = r.multiclass_metrics(ctx["y"], p.argmax(1), p)
    protocol = r.read_json(protocol_path)
    (run / "protocol.json").write_bytes(protocol_path.read_bytes())
    source = run / "source_snapshot" / "old_xgb.py"
    source.parent.mkdir()
    source.write_text("# Synthetic historical training source\n")
    b = {**common_binding(ref, ctx, policy, seed), "input_sha256": ref.contract["input_sha256"],
         "protocol_sha256": r.sha(protocol_path), "labels_sha256": ctx["labels_sha256"],
         "selection": r.XGB_SELECTION, "rounds": rounds, "limit": None,
         "params": {**{k: v for k, v in protocol["xgboost"].items() if k != "num_boost_round"},
                    "seed": seed, "disable_default_eval_metric": 1},
         "class_weights": class_weights(ref.arrays["y"][ctx["train"]], len(ctx["labels"]), policy).astype(np.float32).tolist(),
         "source_code": {"old_xgb.py": r.sha(source)}}
    np.savez(run / "cohort.npz", train_ordinals=ctx["train"], validation_ordinals=ctx["validation"])
    for name in ("validation.npz", "final_validation.npz"):
        np.savez(run / name, proba=p, y=ctx["y"], ordinals=ctx["validation"])
    for name in ("model.ubj", "final_model.ubj"):
        (run / name).write_bytes(b"not-a-model: synthetic provenance fixture")
    dump(run / "labels.json", ctx["labels"])
    dump(run / "history.json", [{"iteration": i, "validation_macro_f1": metrics["macro_f1"]} for i in range(rounds)])
    proof = {"exact_proba": True, "max_abs_diff": 0., "validation_count": len(ctx["y"]),
             "selected_iteration": 0, "checkpoint_sha256": r.sha(run / "model.ubj"), "test_evaluated": False}
    dump(run / "replay.json", proof)
    hashes = {str(p.relative_to(run)): r.sha(p) for p in run.rglob("*") if p.is_file()}
    m = {"method": "xgboost", "status": "completed", "test_evaluated": False,
         "contract_sha256": ref.fingerprint, "input_sha256": ref.contract["input_sha256"],
         "protocol_sha256": r.sha(protocol_path), "binding": b, "weight_policy": policy, "seed": seed,
         "selection": r.XGB_SELECTION, "scope": "full_cohort", "selected_iteration": 0,
         "metrics": metrics, "final_metrics": metrics, "artifact_files": hashes,
         "selected_counts": [len(ctx["train"]), len(ctx["y"])]}
    dump(run / "run_manifest.json", m)
    return p


def test_protocol_frozen(protocol):
    r.validate_protocol(protocol)
    protocol["uncertainty"]["replicates"] = 20
    with pytest.raises(ValueError, match="bootstrap"):
        r.validate_protocol(protocol)


def test_first_maximum_uses_rounded_macro_not_balanced_accuracy():
    history = [{"epoch": i, "validation": {"macro_f1": f, "balanced_acc": b}}
               for i, (f, b) in enumerate([(0.6, .4), (.6000004, .9), (.59, .99)])]
    assert r.first_maximum(history, 3)["epoch"] == 0
    with pytest.raises(ValueError, match="fixed-budget"):
        r.first_maximum(history, 30)
    history[2]["epoch"] = 1
    with pytest.raises(ValueError, match="History order"):
        r.first_maximum(history, 3)


def test_cohort_context_never_indexes_test_rows(context):
    assert len(context["y"]) == len(context["validation"])
    assert context["train_support"].tolist() == list(range(5, 11))
    assert np.min(context["concepts"]) == 0


def test_per_class_counts_precision_recall_f1():
    y = np.array([0, 0, 1, 1, 2])
    pred = np.array([0, 1, 1, 2, 1])
    rows = r.class_statistics(y, pred, ["a", "b", "c", "absent"])
    assert rows[1] == {"class_index": 1, "label": "b", "support": 2, "predicted": 3,
                       "TP": 1, "FP": 2, "FN": 1, "precision": 1/3, "recall": .5, "f1": .4}
    assert rows[3]["recall"] is None and rows[3]["precision"] == 0


def test_rare_training_ties_and_concept_boundaries():
    labels = list(map(str, range(30)))
    y = np.repeat(np.arange(30), np.arange(1, 31))
    counts = np.array([0, 1, 2, 4, 5, 9, 10, 11])
    groups, masks, definitions = r.group_definitions(np.ones(30), y, counts, labels)
    assert groups["train_rare_quartile"] == list(range(7))
    assert len(groups["exploratory_validation_le100"]) == 30
    assert definitions["exploratory_validation_le100"]["exploratory"] is True
    assert [m.sum() for m in masks.values()] == [2, 2, 2, 2]
    assert np.all(sum(m.astype(int) for m in masks.values()) == 1)


def test_group_precision_counts_false_positives_from_outside_group(context):
    y = context["y"]
    p = np.zeros((len(y), 6)); p[:, 0] = 1
    groups, masks, _ = r.group_definitions(context["train_support"], y, context["concepts"], context["labels"])
    row = r.score_prediction(y, p, context["labels"], groups, masks)
    assert row["class_groups"]["train_rare_quartile"]["FP"] == len(y)-sum(y == 0)
    assert row["class_groups"]["train_rare_quartile"]["mean_precision"] == pytest.approx(1/6)


def test_empty_subgroups_retained(context):
    p, _ = probability_matrix(context["y"], 6)
    row = r.score_prediction(context["y"], p, context["labels"], {"empty": []},
                             {"empty": np.zeros(len(p), dtype=bool)})
    assert row["concept_groups"]["empty"] == {"n": 0, "metrics": None}
    assert row["class_groups"]["empty"]["mean_recall"] is None
    json.dumps(row, allow_nan=False)


def test_bootstrap_identical_predictions_zero_width_and_reproducible(context):
    y = context["y"]
    p, _ = probability_matrix(y, 6)
    result = r.paired_stratified_bootstrap(y, p.argmax(1), p.argmax(1), 6)
    assert result == r.paired_stratified_bootstrap(y, p.argmax(1), p.argmax(1), 6)
    assert result["replicates"] == 2000 and result["seed"] == 20260919
    for value in result["metrics"].values():
        assert value == {"delta": 0., "ci_low": 0., "ci_high": 0.}
    assert all(c["delta"] == c["ci_low"] == c["ci_high"] == 0 for c in result["per_class_recall"])


def test_bootstrap_matches_exhaustive_patient_resampling_distribution():
    # Two patients per class: enumerate all 16 stratified paired patient samples.
    y = np.array([0, 0, 1, 1])
    a = np.array([0, 1, 1, 1]); b = np.array([1, 1, 0, 1])
    exact = {m: [] for m in r.BOOT_METRICS}
    functions = {"macro_f1": lambda y, p: f1_score(y, p, average="macro", zero_division=0),
                 "balanced_acc": balanced_accuracy_score, "accuracy": accuracy_score}
    for picks in itertools.product(range(2), repeat=4):
        idx = np.array([picks[0], picks[1], 2+picks[2], 2+picks[3]])
        for name, fn in functions.items():
            exact[name].append(fn(y[idx], a[idx])-fn(y[idx], b[idx]))
    result = r.paired_stratified_bootstrap(y, a, b, 2, replicates=2000)
    for name, fn in functions.items():
        # Repeat equally likely outcomes to represent the discrete distribution,
        # not linear interpolation between only 16 empirical order statistics.
        expected = np.quantile(np.repeat(exact[name], 1000), [.025, .975])
        assert result["metrics"][name]["ci_low"] == pytest.approx(expected[0])
        assert result["metrics"][name]["ci_high"] == pytest.approx(expected[1])
        assert result["metrics"][name]["delta"] == pytest.approx(fn(y, a)-fn(y, b))
    assert result["per_class_recall"][0]["delta"] == .5
    assert result["per_class_recall"][1]["delta"] == .5


def test_vectorized_metrics_match_sklearn_with_absent_class():
    y = np.array([0, 0, 1, 1]); p = np.array([0, 2, 1, 2])
    tp = np.bincount(y[y == p], minlength=4)
    metrics, _ = r._count_metrics(tp, np.bincount(p, minlength=4), np.bincount(y, minlength=4))
    assert metrics["macro_f1"] == pytest.approx(f1_score(y, p, average="macro", zero_division=0))
    assert metrics["balanced_acc"] == .5
    result = r.paired_stratified_bootstrap(y, p, p, 4, replicates=20)
    assert result["per_class_recall"][3]["delta"] is None


def test_prediction_guards(context):
    p, _ = probability_matrix(context["y"], 6)
    with pytest.raises(ValueError, match="ordinal"):
        r.validate_prediction(p, context["y"], context["validation"][::-1], context)
    with pytest.raises(ValueError, match="label"):
        r.validate_prediction(p, context["y"][::-1], context["validation"], context)
    bad = p.copy(); bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="probability"):
        r.validate_prediction(bad, context["y"], context["validation"], context)
    with pytest.raises(ValueError, match="Unnormalized"):
        r.validate_prediction(p*.5, context["y"], context["validation"], context)


def test_load_gchm_validates_historical_snapshot_without_model_loading(tmp_path, monkeypatch):
    ref = FakeReference(); ctx = r.cohort_context(ref)
    run = tmp_path / "gchm"
    expected = make_gchm(run, ref, ctx)
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("No torch checkpoint loading allowed"))
    p, provenance = r.load_gchm(run, "none", 1234, ref, ctx)
    np.testing.assert_array_equal(p, expected)
    assert provenance["selected_epoch"] == 0
    assert "old_training.py" in provenance["source_snapshot_sha256"]
    (run / "source_snapshot" / "old_training.py").write_text("tampered")
    with pytest.raises(ValueError, match="Hash mismatch"):
        r.load_gchm(run, "none", 1234, ref, ctx)


@pytest.mark.parametrize("mutation,match", [
    ("checkpoint", "Checkpoint SHA"), ("selection", "selection/budget"),
    ("status", "Incomplete"), ("cohort", "Training cohort"),
    ("replay", "Replay file/manifest"), ("metrics", "Recomputed"),
])
def test_gchm_rejects_invalid_provenance(tmp_path, mutation, match):
    ref = FakeReference(); ctx = r.cohort_context(ref); run = tmp_path / "g"
    make_gchm(run, ref, ctx)
    m = r.read_json(run / "run_manifest.json")
    if mutation == "checkpoint":
        (run / "best.pt").write_bytes(b"tamper")
    elif mutation == "selection":
        m["binding"]["selection"] = "validation balanced_acc"
    elif mutation == "status":
        m["status"] = "running"
    elif mutation == "cohort":
        np.savez(run / "cohort.npz", train_ordinals=ctx["train"][::-1], validation_ordinals=ctx["validation"])
    elif mutation == "replay":
        m["replay"]["selected_epoch"] = 1
    elif mutation == "metrics":
        h = r.read_json(run / "history.json")
        h[0]["validation"]["accuracy"] = .123456
        dump(run / "history.json", h)
    dump(run / "run_manifest.json", m)
    with pytest.raises(ValueError, match=match):
        r.load_gchm(run, "none", 1234, ref, ctx)


def test_xgb_loader_matches_actual_runner_schema_and_rejects_tamper(tmp_path, protocol):
    ref = FakeReference(); ctx = r.cohort_context(ref); run = tmp_path / "x"
    pp = tmp_path / "protocol.json"; dump(pp, protocol)
    expected = make_xgb(run, ref, ctx, pp)
    p, provenance = r.load_xgboost(run, "none", 1234, ref, ctx, r.sha(pp))
    np.testing.assert_array_equal(p, expected)
    assert provenance["selected_iteration"] == 0
    (run / "validation.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Hash mismatch"):
        r.load_xgboost(run, "none", 1234, ref, ctx, r.sha(pp))


@pytest.mark.parametrize("mutation,match", [
    ("protocol", "protocol"), ("labels", "Label hash"), ("seed", "Seed/weight"),
    ("rounds", "fixed-budget"), ("weights", "weight mismatch"),
    ("selected", "first-max"), ("metrics", "selected metrics"), ("final_metrics", "final-budget"),
])
def test_xgb_rejects_manifest_inconsistency(tmp_path, protocol, mutation, match):
    ref = FakeReference(); ctx = r.cohort_context(ref); run = tmp_path / "x"
    pp = tmp_path / "protocol.json"; dump(pp, protocol)
    make_xgb(run, ref, ctx, pp)
    m = r.read_json(run / "run_manifest.json")
    if mutation == "protocol": m["protocol_sha256"] = "wrong"
    elif mutation == "labels": m["binding"]["labels_sha256"] = "wrong"
    elif mutation == "seed": m["binding"]["seed"] = 1235
    elif mutation == "rounds": m["binding"]["rounds"] = 2
    elif mutation == "weights": m["binding"]["class_weights"][0] = 2
    elif mutation == "selected": m["selected_iteration"] = 1
    elif mutation == "metrics": m["metrics"]["accuracy"] = .123456
    elif mutation == "final_metrics": m["final_metrics"]["accuracy"] = .123456
    dump(run / "run_manifest.json", m)
    with pytest.raises(ValueError, match=match):
        r.load_xgboost(run, "none", 1234, ref, ctx, r.sha(pp))


def test_hash_path_traversal_and_symlink_escape_rejected(tmp_path):
    root = tmp_path / "root"; root.mkdir()
    outside = tmp_path / "outside"; outside.write_text("outside")
    with pytest.raises(ValueError, match="Unsafe"):
        r.verify_hashes(root, {"../outside": r.sha(outside)})
    (root / "link").symlink_to(outside)
    with pytest.raises(ValueError, match="Unsafe"):
        r.verify_hashes(root, {"link": r.sha(outside)})


@pytest.fixture
def analysis(context, protocol):
    probabilities = {f"{model}/{policy}/{seed}": probability_matrix(context["y"], 6, seed+i+j)[0]
                     for i, model in enumerate(("gchm", "xgboost"))
                     for j, policy in enumerate(protocol["weight_policies"])
                     for seed in protocol["seeds"]}
    # Lower-level fixture uses fewer replicates; frozen production entrypoint cannot.
    protocol = copy.deepcopy(protocol); protocol["uncertainty"]["replicates"] = 30
    return r.analyze_predictions(probabilities, context, protocol), probabilities


def test_complete_analysis_separate_seeds_summaries_ensembles_and_directions(analysis, context):
    result, probabilities = analysis
    assert len(result["rows"]) == 16 and len(result["seed_summaries"]) == 4
    assert len(result["comparisons"]) == 16
    ids = [f"gchm/none/{s}" for s in (1234, 1235, 1236)]
    ensemble = np.mean(np.stack([probabilities[k] for k in ids]), axis=0)
    assert result["rows"]["gchm/none/ensemble"]["metrics"] == r.multiclass_metrics(context["y"], ensemble.argmax(1), ensemble)
    values = [result["rows"][key]["metrics"]["macro_f1"] for key in ids]
    summary = result["seed_summaries"]["gchm/none"]["statistics"]["metrics"]["macro_f1"]
    assert summary == {"mean": np.mean(values), "sample_sd": np.std(values, ddof=1)}
    for key, row in result["comparisons"].items():
        if key.startswith("sqrt_minus_none"):
            assert "/sqrt_inverse/" in row["A"] and "/none/" in row["B"]
        else:
            assert row["A"].startswith("gchm/") and row["B"].startswith("xgboost/")
        assert len(row["bootstrap"]["per_class_recall"]) == 6
        a = result["rows"][row["A"]]["metrics"]["accuracy"]
        b = result["rows"][row["B"]]["metrics"]["accuracy"]
        assert row["metric_delta"]["accuracy"] == a-b
    assert result["test_evaluated"] is False and result["temporal_clean"] is False
    json.dumps(result, allow_nan=False)


def test_analysis_rejects_missing_seed(analysis, context, protocol):
    _, probs = analysis
    probs.pop("gchm/none/1234")
    with pytest.raises(ValueError, match="Missing/extra"):
        r.analyze_predictions(probs, context, protocol)


def test_report_write_no_overwrite_and_turkish_caveats(tmp_path, analysis):
    report, _ = analysis
    targets = r.write_report(report, tmp_path / "reports")
    before = [p.read_bytes() for p in targets]
    markdown = targets[1].read_text()
    for term in ("temporal_clean=false", "precision", "seçim yanlılığı", "hesap bütçesi", "çoklu karşılaştırma", "2000", "20260919"):
        assert term in markdown
    assert json.loads(targets[0].read_text())["test_evaluated"] is False
    with pytest.raises(FileExistsError):
        r.write_report(report, targets[0].parent)
    assert before == [p.read_bytes() for p in targets]
    r.write_report(report, targets[0].parent, overwrite=True)
    assert before == [p.read_bytes() for p in targets]


def test_write_preflights_both_targets_and_rejects_symlink(tmp_path, analysis):
    report, _ = analysis
    (tmp_path / "report.md").write_text("preserve")
    with pytest.raises(FileExistsError): r.write_report(report, tmp_path)
    assert not (tmp_path / "report.json").exists()
    (tmp_path / "report.md").unlink()
    original = tmp_path / "original"; original.write_text("preserve")
    (tmp_path / "report.json").symlink_to(original)
    with pytest.raises(ValueError, match="symlink"):
        r.write_report(report, tmp_path, overwrite=True)
    assert original.read_text() == "preserve"


def test_synthetic_full_artifact_pipeline_and_cli_no_training(tmp_path, protocol, monkeypatch):
    ref = FakeReference(k=30); ctx = r.cohort_context(ref)
    for policy in protocol["weight_policies"]:
        for seed in protocol["seeds"]:
            relative = f"historical/{policy}/seed{seed}"
            protocol["gchm_reuse"][policy][str(seed)] = relative
            make_gchm(tmp_path / relative, ref, ctx, policy, seed)
    pp = tmp_path / "protocol.json"; dump(pp, protocol)
    for policy in protocol["weight_policies"]:
        for seed in protocol["seeds"]:
            make_xgb(tmp_path / "runs" / policy / f"seed{seed}", ref, ctx, pp, policy, seed)
    monkeypatch.setattr(r, "Reference", lambda artifact, expected: ref)
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("No model loading"))
    result = r.build_report(pp, repo=tmp_path, artifact=tmp_path / "synthetic-input")
    assert len(result["provenance"]) == 12
    assert all(c["bootstrap"]["replicates"] == 2000 for c in result["comparisons"].values())
    assert len(result["group_definitions"]["train_rare_quartile"]["class_indices"]) == 7
    assert result["analysis_provenance"]["protocol_sha256"] == r.sha(pp)
    # Exercise CLI output and overwrite preflight without repeating artifact analysis.
    monkeypatch.setattr(r, "HERE", tmp_path)
    monkeypatch.setattr(r, "build_report", lambda *a, **kw: result)
    r.main(["--protocol", str(pp), "--output-dir", str(tmp_path / "reports")])
    assert r.read_json(tmp_path / "reports/report.json")["cohort"]["validation_count"] == len(ctx["y"])
    monkeypatch.setattr(r, "build_report", lambda *a, **kw: pytest.fail("Must refuse before analysis"))
    with pytest.raises(FileExistsError):
        r.main(["--output-dir", str(tmp_path / "reports")])
    with pytest.raises(ValueError, match="historical artifacts"):
        r.main(["--output-dir", str(tmp_path / "runs")])
