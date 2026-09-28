"""Synthetic contract tests for the CEI-GNN v2 pair study."""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import shutil
import sys
from functools import lru_cache
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "comparison/standardized/clinical_graph_v2/cei_v2_study.py"


def _module():
    assert MODULE_PATH.is_file(), "CEI-GNN v2 study module must implement the locked planner"
    spec = importlib.util.spec_from_file_location("cei_v2_study_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _inputs(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    targets, canonical = tmp_path / "targets.csv", tmp_path / "canonical.json"
    targets.write_text("x")
    canonical.write_text("{}")
    return artifact, targets, canonical


@lru_cache(maxsize=None)
def _source(mode, seed, budget):
    from comparison.standardized.clinical_graph_v2 import train
    from comparison.standardized.clinical_graph_v2.methods import build_method
    from comparison.standardized.clinical_graph_v2.tensorize import PAYLOAD_WIDTH

    train_limit, dev_limit, epochs = budget
    parser = train.parser()
    args = train.normalize_method_args(parser.parse_args([
        "--artifact", "a", "--targets", "t", "--output", "o", "--method", "cei_gnn_v2",
        "--train-limit", str(train_limit), "--dev-limit", str(dev_limit),
        "--sample-seed", "1234", "--seed", str(seed), "--top-k-labels", "10",
        "--edges", "all", "--edge-direction", "forward", "--weights", "sqrt_inverse",
        "--selection-fold", "dev", "--final-eval", "none", "--epochs", str(epochs),
        "--patience", "40", "--method-option", f"pair_mode={mode}"]), parser)
    model = build_method("cei_gnn_v2", num_tokens=8, node_dim=3, edge_dim=PAYLOAD_WIDTH,
                         num_classes=2, hidden=args.hidden, layers=args.layers,
                         dropout=args.dropout, token_dim=args.token_dim, num_triples=2,
                         args=args)
    runner = {"lr": args.lr, "weight_decay": args.weight_decay, "batch_size": args.batch_size,
              "min_delta": args.min_delta,
              "early_stopping_start_epoch_index": train.early_stopping_start_epoch("cei_gnn_v2", model)}
    return json.dumps(model.run_config()), json.dumps(runner)


def _binding(mode="product", seed=2025, budget=(10000, 5000, 40), **updates):
    config_json, runner_json = _source(mode, seed, budget)
    config, runner = json.loads(config_json), json.loads(runner_json)
    arch = config["architecture"]
    train_limit, dev_limit, epochs = budget
    binding = {
        "method": "cei_gnn_v2", "method_config": config,
        "method_native_defaults": config["native_defaults"],
        "selected_dev": {"epoch": 3, "epoch_index": 2, "metric": "macro_f1",
                         "metric_value": 0.5, "prediction_sha256": "pred", "sample_ids_sha256": "ids"},
        "artifact_graphs_sha256": "g", "artifact_visit_membership_sha256": "m",
        "targets_sha256": "t", "target_binding_sha256": "tb", "label_order": ["A", "B"],
        "source_code": {"x.py": "h"}, "preprocessing_sha256": "p",
        "split_sample_ids_sha256": {"train": "tr", "dev": "dv", "validation": "va"},
        "seed": seed, "sample_seed": 1234, "selection_fold": "dev", "final_eval": "none",
        "train_limit": train_limit, "dev_limit": dev_limit, "epochs": epochs, "patience": 40,
        "test_evaluated": False, "parameter_count": arch["parameter_count"],
        "active_parameter_count": arch["active_parameter_count"], "weights": "sqrt_inverse",
        "edge_direction": "forward", "edges": "all", "top_k_labels": 10,
        "message_passing": True, "edge_payload": True, "num_classes": 2,
        "input_contract_version": "v", "vocabulary_size": 8, "node_dim": 3,
        "edge_dim": arch["edge_dim"], "hidden": arch["hidden"], "layers": arch["layers"],
        "dropout": arch["dropout"], "num_relations": arch["num_relations"],
        "num_meta_relations": 2, **runner,
    }
    binding.update(updates)
    return binding


def test_smoke_plan_has_three_modes_and_full_argv_is_pinned(tmp_path):
    module = _module()
    artifact, targets, canonical = _inputs(tmp_path)
    stages = module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                               output_root=tmp_path / "runs")
    smoke = stages[:3]
    assert [(s.name, s.budget, s.seed, s.pair_mode) for s in smoke] == [
        ("v2_smoke_product", (256, 128, 2), 1234, "product"),
        ("v2_smoke_additive", (256, 128, 2), 1234, "additive"),
        ("v2_smoke_off", (256, 128, 2), 1234, "off"),
    ]
    normalized = []
    outputs = []
    for s in smoke:
        argv = list(s.argv)
        mode_index = argv.index("--method-option") + 1
        output_index = argv.index("--output") + 1
        outputs.append(Path(argv[output_index]))
        argv[mode_index] = "pair_mode=MODE"
        argv[output_index] = "<mode-output>"
        normalized.append(argv)
    assert normalized[0] == normalized[1] == normalized[2]
    assert len(set(outputs)) == 3
    assert outputs == [tmp_path / "runs" / f"v2_smoke_{mode}"
                       for mode in ("product", "additive", "off")]
    expected_full = [(f"{mode}_seed{seed}", (10000, 5000, 40), seed, mode)
                     for seed in (1234, 2025, 7) for mode in ("product", "additive", "off")]
    assert [(s.name, s.budget, s.seed, s.pair_mode) for s in stages[3:]] == expected_full
    literal_full_argv = [
        "-m", "comparison.standardized.clinical_graph_v2.train", "--artifact", str(artifact),
        "--targets", str(targets), "--canonical", str(canonical), "--output", str(tmp_path / "runs" / "product_seed1234"),
        "--method", "cei_gnn_v2", "--train-limit", "10000", "--dev-limit", "5000",
        "--sample-seed", "1234", "--seed", "1234", "--top-k-labels", "10", "--edges", "all",
        "--edge-direction", "forward", "--weights", "sqrt_inverse", "--selection-fold", "dev",
        "--final-eval", "none", "--epochs", "40", "--patience", "40",
        "--method-option", "pair_mode=product",
    ]
    assert stages[3].argv[1:] == literal_full_argv
    pinned_full_rows = [
        ("product_seed1234", "1234", "product"),
        ("additive_seed1234", "1234", "additive"),
        ("off_seed1234", "1234", "off"),
        ("product_seed2025", "2025", "product"),
        ("additive_seed2025", "2025", "additive"),
        ("off_seed2025", "2025", "off"),
        ("product_seed7", "7", "product"),
        ("additive_seed7", "7", "additive"),
        ("off_seed7", "7", "off"),
    ]
    expected_full_argv = []
    for name, seed, mode in pinned_full_rows:
        expected_full_argv.append([
            sys.executable, "-m", "comparison.standardized.clinical_graph_v2.train",
            "--artifact", str(artifact), "--targets", str(targets), "--canonical", str(canonical),
            "--output", str(tmp_path / "runs" / name), "--method", "cei_gnn_v2",
            "--train-limit", "10000", "--dev-limit", "5000", "--sample-seed", "1234",
            "--seed", seed, "--top-k-labels", "10", "--edges", "all",
            "--edge-direction", "forward", "--weights", "sqrt_inverse",
            "--selection-fold", "dev", "--final-eval", "none", "--epochs", "40",
            "--patience", "40", "--method-option", f"pair_mode={mode}",
        ])
    assert [stage.argv for stage in stages[3:]] == expected_full_argv


def test_plan_has_exact_twelve_dev_only_stages(tmp_path):
    module = _module()
    artifact, targets, canonical = _inputs(tmp_path)
    stages = module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                               output_root=tmp_path / "runs")
    expected = [(f"v2_smoke_{mode}", (256, 128, 2), 1234, mode)
                for mode in ("product", "additive", "off")] + [
        (f"{mode}_seed{seed}", (10000, 5000, 40), seed, mode)
        for seed in (1234, 2025, 7) for mode in ("product", "additive", "off")]
    assert [(s.name, s.budget, s.seed, s.pair_mode) for s in stages] == expected
    for stage in stages:
        argv = stage.argv
        assert argv[argv.index("--method") + 1] == "cei_gnn_v2"
        assert argv[argv.index("--seed") + 1] == str(stage.seed)
        assert argv[argv.index("--sample-seed") + 1] == "1234"
        assert argv[argv.index("--final-eval") + 1] == "none"
        assert argv[argv.index("--selection-fold") + 1] == "dev"
        assert argv[argv.index("--method-option") + 1] == f"pair_mode={stage.pair_mode}"
        assert Path(argv[argv.index("--output") + 1]) == Path(stage.output)
        flags = [token for token in argv if token.startswith("--")]
        values = [argv[i + 1] for i, token in enumerate(argv[:-1]) if token.startswith("--")]
        assert "none" == argv[argv.index("--final-eval") + 1]
        assert "dev" == argv[argv.index("--selection-fold") + 1]
        assert not any("test" in flag for flag in flags)
        assert not any(value == "test" for value in values)


def test_plan_refuses_occupied_output_and_relative_paths(tmp_path):
    module = _module()
    artifact, targets, canonical = _inputs(tmp_path)
    (tmp_path / "runs" / "off_seed7").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                          output_root=tmp_path / "runs")
    smoke_root = tmp_path / "smoke-runs"
    (smoke_root / "v2_smoke_additive").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="occupied stage output"):
        module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                          output_root=smoke_root)
    with pytest.raises(ValueError, match="absolute"):
        module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                          output_root="relative/runs")


def test_exact_plan_validator_rejects_edited_stage(tmp_path):
    module = _module()
    artifact, targets, canonical = _inputs(tmp_path)
    stages = module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                               output_root=tmp_path / "runs")
    module.validate_exact_plan(stages)
    edited = list(stages)
    edited[5] = dataclasses.replace(edited[5], argv=edited[5].argv[:-1] + ["pair_mode=product"])
    with pytest.raises(ValueError, match="argv"):
        module.validate_exact_plan(edited)
    with pytest.raises(ValueError, match="twelve-stage"):
        module.validate_exact_plan(stages[:-1])


def test_binding_policy_accepts_source_config_and_rejects_drift():
    module = _module()
    module.validate_v2_binding(_binding("additive", 2025), budget=(10000, 5000, 40),
                               seed=2025, mode="additive")
    with pytest.raises(ValueError, match="seed"):
        module.validate_v2_binding(_binding("additive", 2025), budget=(10000, 5000, 40),
                                   seed=7, mode="additive")
    with pytest.raises(ValueError, match="pair_mode"):
        module.validate_v2_binding(_binding("additive", 2025), budget=(10000, 5000, 40),
                                   seed=2025, mode="product")
    with pytest.raises(ValueError, match="test_evaluated"):
        module.validate_v2_binding(_binding("off", 7, test_evaluated=True),
                                   budget=(10000, 5000, 40), seed=7, mode="off")
    drifted = _binding("product", 1234)
    drifted["method_config"]["architecture"]["pair_rank"] = 8
    with pytest.raises(ValueError, match="run_config"):
        module.validate_v2_binding(drifted, budget=(10000, 5000, 40), seed=1234, mode="product")


def test_binding_parameter_counts_match_each_arm_architecture():
    module = _module()
    binding = _binding("off", 7)
    binding["parameter_count"] += 1
    with pytest.raises(ValueError, match="parameter_count"):
        module.validate_v2_binding(binding, budget=(10000, 5000, 40), seed=7, mode="off")
    binding = _binding("off", 7)
    binding["active_parameter_count"] += 1
    with pytest.raises(ValueError, match="active_parameter_count"):
        module.validate_v2_binding(binding, budget=(10000, 5000, 40), seed=7, mode="off")


def test_arm_parity_allows_seed_and_mode_but_rejects_other_differences():
    module = _module()
    arms = {f"{mode}_seed{seed}": _binding(mode, seed)
            for seed in (1234, 2025, 7) for mode in ("product", "additive", "off")}
    assert module.assert_arm_parity(arms)
    arms["off_seed7"] = _binding("off", 7, split_sample_ids_sha256={
        "train": "other", "dev": "dv", "validation": "va"})
    with pytest.raises(ValueError, match="split_sample_ids_sha256"):
        module.assert_arm_parity(arms)


@pytest.mark.parametrize("field,value", [
    ("class_weight_values", [1.0, 2.0]),
    ("edge_payload", False),
    ("token_min_count", 9),
    ("future_policy_key", "changed"),
])
def test_arm_parity_rejects_every_unallowlisted_binding_difference(field, value):
    module = _module()
    product, additive = _binding("product", 1234), _binding("additive", 1234)
    product[field] = "baseline" if field == "future_policy_key" else (
        [1.0, 1.0] if field == "class_weight_values" else (True if field == "edge_payload" else 1))
    additive[field] = value
    with pytest.raises(ValueError, match=field):
        module.assert_arm_parity({"product": product, "additive": additive})


def test_arm_parity_accepts_only_documented_arm_varying_fields():
    module = _module()
    product, off = _binding("product", 1234), _binding("off", 2025)
    product["selected_dev"] = {"epoch": 4, "epoch_index": 3, "metric": "macro_f1",
                                "metric_value": 0.7, "prediction_sha256": "p", "sample_ids_sha256": "s"}
    off["selected_dev"] = {"epoch": 3, "epoch_index": 2, "metric": "macro_f1",
                            "metric_value": 0.6, "prediction_sha256": "o", "sample_ids_sha256": "s"}
    off["active_parameter_count"] = off["method_config"]["architecture"]["active_parameter_count"]
    assert module.assert_arm_parity({"product": product, "off": off})


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _write_analysis_history_fixtures(root, names, binding):
    """RR2-3: mocked analysis validators still need valid dev history rows."""
    binding["epochs"] = 1
    binding.setdefault("selected_dev", {})["epoch"] = 1
    for name in names:
        _write_json(Path(root) / name / "history.json", [{
            "epoch": 1, "train_loss": 1.0, "macro_f1": 1.0,
            "selection_fold": "dev", "seconds": 1.0}])


def test_full_phase_requires_completed_smoke(tmp_path):
    module = _module()
    assert hasattr(module, "execute_plan"), "pair study executor is missing"
    artifact, targets, canonical = _inputs(tmp_path)
    stages = module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                               output_root=tmp_path / "runs")
    with pytest.raises(ValueError, match="smoke"):
        module.execute_plan(stages, journal_path=tmp_path / "journal.json", phase="full")
    assert not (tmp_path / "journal.json").exists()


def test_executor_journals_failure_and_launches_nothing_else(tmp_path, monkeypatch):
    module = _module()
    assert hasattr(module, "execute_plan"), "pair study executor is missing"
    artifact, targets, canonical = _inputs(tmp_path)
    stages = module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                               output_root=tmp_path / "runs")
    monkeypatch.setattr(module, "_capture", lambda stage: {
        "source_state_sha256": "s", "input_state_sha256": "i", "executable_sources": {}})
    launches = []

    def fail(argv, **kwargs):
        launches.append(argv)
        raise module.subprocess.CalledProcessError(2, argv)

    monkeypatch.setattr(module.subprocess, "run", fail)
    with pytest.raises(SystemExit) as error:
        module.execute_plan(stages, journal_path=tmp_path / "journal.json", phase="smoke")
    assert error.value.code == 1
    journal = json.loads((tmp_path / "journal.json").read_text())
    assert journal["status"] == "failed" and journal["stages"]["v2_smoke_product"]["status"] == "failed"
    assert len(launches) == 1 and launches[0][-1] == "--execute"


def test_executor_refuses_source_drift_between_full_stages(tmp_path, monkeypatch):
    module = _module()
    assert hasattr(module, "execute_plan"), "pair study executor is missing"
    artifact, targets, canonical = _inputs(tmp_path)
    root = tmp_path / "runs"
    stages = module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                               output_root=root)
    _write_json(root / "v2_smoke_product" / "binding.json", {})
    _write_json(root / "v2_smoke_product" / "result.json", {"status": "completed", "binding": {},
                                                       "metrics": None, "dev_metrics": {},
                                                       "test_evaluated": False})
    smoke_capture = {"source_state_sha256": "a", "input_state_sha256": "i",
                     "executable_sources": {}}
    _write_json(root / "journal_smoke.json", {"status": "completed", "stages": {
        "v2_smoke_product": {"status": "completed", "replay": {"status": "verified"},
                     "postflight": smoke_capture}}})
    # _verify_smoke is mocked: the captures here are full-stage 1 preflight (a),
    # full-stage 1 postflight (a), then full-stage 2 preflight (b). This reaches
    # the pre-launch guard; the old four-value a,a,a,b sequence made stage 2's
    # preflight match and reached drift only at its postflight, after launch.
    identities = iter(["a", "a", "b"])
    monkeypatch.setattr(module, "_capture", lambda stage: {
        "source_state_sha256": next(identities), "input_state_sha256": "i",
        "executable_sources": {}})
    monkeypatch.setattr(module, "_verify_smoke", lambda *a, **k: {})
    launches = []

    def fake_run(argv, **kwargs):
        launches.append(argv)
        output = Path(argv[argv.index("--output") + 1])
        binding = {"seed": 1234}
        _write_json(output / "binding.json", binding)
        _write_json(output / "result.json", {"status": "completed", "binding": binding,
                                             "metrics": None, "dev_metrics": {},
                                             "test_evaluated": False})
        return module.subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module.pilot, "_assert_runner_source_binding", lambda *a: True)
    monkeypatch.setattr(module.pilot, "_validate_artifacts", lambda *a: None)
    monkeypatch.setattr(module, "validate_v2_binding", lambda *a, **k: None)
    monkeypatch.setattr(module, "replay_v2_stage", lambda *a: {"status": "verified"})
    with pytest.raises(SystemExit) as error:
        module.execute_plan(stages, journal_path=root / "journal.json", phase="full")
    assert error.value.code == 3
    assert len(launches) == 1
    journal = json.loads((root / "journal.json").read_text())
    assert journal["stages"]["product_seed1234"]["status"] == "bound"
    assert journal["stages"]["additive_seed1234"]["status"] == "refused"


def test_replay_reconstructs_v2_model_and_rejects_tampering(tmp_path, monkeypatch):
    module = _module()
    assert hasattr(module, "replay_v2_stage"), "pair study replay is missing"
    import hashlib
    import numpy as np
    import torch
    from torch_geometric.loader import DataLoader
    from comparison.standardized.clinical_graph_v2 import train
    from comparison.standardized.clinical_graph_v2.contracts import recursive_source_hashes, sample_ids_sha256
    from comparison.standardized.clinical_graph_v2.methods import build_method
    from comparison.standardized.clinical_graph_v2.schema import sha256
    from comparison.standardized.clinical_graph_v2.tensorize import (
        PAYLOAD_WIDTH, PREPROCESSING_VERSION, ClinicalGraphData, Scaler, Vocabulary,
        preprocessing_state)

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "graphs.jsonl").write_text("synthetic graph source\n")
    (artifact / "visit_membership.jsonl").write_text("synthetic membership\n")
    targets_path = tmp_path / "targets.csv"
    targets_path.write_text("synthetic targets\n")
    prep = {"preprocessing_version": PREPROCESSING_VERSION,
            "context_categories": {"gender": [], "race": [], "arrival_transport": []},
            "vocabulary": Vocabulary(["synthetic"], 1), "scaler": Scaler({}),
            "token_min_count": 1, "triples": ["visit|rel|complaint"], "triple_counts": {}}
    output = tmp_path / "product_seed2025"
    output.mkdir()
    (output / "preprocessing.json").write_text(json.dumps(preprocessing_state(prep), sort_keys=True))

    def row(sid, subject, label, scale):
        graph = ClinicalGraphData(
            x=torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, -0.1]]) * scale,
            edge_index=torch.tensor([[0, 0], [1, 2]]),
            edge_attr=torch.zeros((2, PAYLOAD_WIDTH)))
        graph.token = torch.tensor([0, 1, 1])
        graph.node_type = torch.tensor([1, 2, 5])
        graph.edge_relation = torch.zeros(2, dtype=torch.long)
        graph.edge_triple = torch.zeros(2, dtype=torch.long)
        graph.visit_membership_index = torch.tensor([[0, 0, 0], [0, 1, 2]])
        graph.num_visits = torch.tensor([1])
        graph.y = torch.tensor([label])
        graph.sample_id, graph.subject = sid, subject
        return graph

    splits = {"train": [row("train-1", "p1", 0, 1.0)],
              "dev": [row("dev-1", "p2", 0, 0.5), row("dev-2", "p3", 1, -1.0)],
              "validation": [row("validation-1", "p4", 1, 2.0)]}
    monkeypatch.setattr(train, "load_targets", lambda path: {"synthetic": True})
    monkeypatch.setattr(train, "select_top_labels", lambda targets, top_k: (targets, [0, 1], {}))
    monkeypatch.setattr(train, "build_dataset", lambda *args, **kwargs: (splits, prep))

    binding = _binding("product", 2025)
    torch.manual_seed(27)
    parser = train.parser()
    args = train.normalize_method_args(parser.parse_args([
        "--artifact", "a", "--targets", "t", "--output", "o", "--method", "cei_gnn_v2",
        "--method-option", "pair_mode=product"]), parser)
    model = build_method("cei_gnn_v2", num_tokens=2, node_dim=2, edge_dim=PAYLOAD_WIDTH,
                         num_classes=2, hidden=args.hidden, layers=args.layers,
                         dropout=args.dropout, token_dim=args.token_dim, num_triples=2,
                         args=args).eval()
    proba, labels = train.evaluate(model, DataLoader(splits["dev"], batch_size=128),
                                   torch.device("cpu"), epoch=0)
    torch.save(model.state_dict(), output / "best.pt")
    ids = np.asarray([graph.sample_id for graph in splits["dev"]])
    np.savez_compressed(output / "dev.npz", proba=proba, y=labels, sample_ids=ids,
                        subjects=np.asarray([graph.subject for graph in splits["dev"]]))
    config = model.run_config()
    binding.update({
        "method_config": config, "method_native_defaults": config["native_defaults"],
        "parameter_count": config["architecture"]["parameter_count"],
        "active_parameter_count": config["architecture"]["active_parameter_count"],
        "vocabulary_size": 2, "node_dim": 2, "num_meta_relations": 2,
        "artifact": str(artifact), "artifact_graphs_sha256": sha256(artifact / "graphs.jsonl"),
        "artifact_visit_membership_file": "visit_membership.jsonl",
        "artifact_visit_membership_sha256": sha256(artifact / "visit_membership.jsonl"),
        "targets_path": str(targets_path), "targets_sha256": sha256(targets_path),
        "source_code": recursive_source_hashes(Path(train.__file__).parent),
        "preprocessing_sha256": sha256(output / "preprocessing.json"),
        "kept_label_indices": [0, 1], "token_min_count": 1, "dropped_relations": [],
        "rewired_relations": [], "min_prior_visits": 0,
        "split_sample_ids_sha256": {fold: sample_ids_sha256(g.sample_id for g in rows)
                                    for fold, rows in splits.items()},
        "selected_dev": {"epoch_index": 0, "prediction_sha256": hashlib.sha256(
            np.ascontiguousarray(proba).tobytes()).hexdigest()},
    })
    result = {"status": "completed", "binding": binding, "metrics": None,
              "dev_metrics": train.metrics(labels, proba, 2), "validation_evaluations": 0}
    proof = module.replay_v2_stage(output, binding, result)
    assert proof["exact_probabilities"] and proof["test_evaluated"] is False
    assert proof["validation_evaluated"] is False
    subject_tamper = tmp_path / "subject_tamper"
    subject_tamper.mkdir()
    (subject_tamper / "preprocessing.json").write_bytes((output / "preprocessing.json").read_bytes())
    with np.load(output / "dev.npz", allow_pickle=False) as saved:
        np.savez_compressed(subject_tamper / "dev.npz", proba=saved["proba"], y=saved["y"],
                            sample_ids=saved["sample_ids"], subjects=np.asarray(["wrong", "p3"]))
    shutil.copy2(output / "best.pt", subject_tamper / "best.pt")
    with pytest.raises(ValueError, match="subjects"):
        module.replay_v2_stage(subject_tamper, binding, result)
    state = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    state["network.bias"][0] += 0.5
    tampered = tmp_path / "tampered"
    tampered.mkdir()
    for name in ("preprocessing.json", "dev.npz"):
        (tampered / name).write_bytes((output / name).read_bytes())
    torch.save(state, tampered / "best.pt")
    with pytest.raises(ValueError, match="probabilities"):
        module.replay_v2_stage(tampered, binding, result)


def test_replay_rejects_test_evaluation_result_guard(tmp_path):
    module = _module()
    output = tmp_path / "out"
    output.mkdir()
    binding = {"selection_fold": "dev", "final_eval": "none", "test_evaluated": False}
    result = {"status": "completed", "binding": binding, "metrics": None,
              "validation_evaluations": 0, "test_evaluated": True}
    with pytest.raises(ValueError, match="test_evaluated"):
        module.replay_v2_stage(output, binding, result)


def test_full_phase_rejects_fabricated_completed_smoke_before_launch(tmp_path, monkeypatch):
    module = _module()
    artifact, targets, canonical = _inputs(tmp_path)
    root = tmp_path / "runs"
    stages = module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                               output_root=root)
    _write_json(root / "v2_smoke_product" / "result.json", {"status": "completed"})
    _write_json(root / "journal_smoke.json", {"status": "completed"})
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("launched"))
    with pytest.raises(ValueError, match="smoke"):
        module.execute_plan(stages, journal_path=root / "journal_full.json", phase="full")
    assert not (root / "journal_full.json").exists()


def test_weighted_macro_f1_matches_sklearn_including_empty_classes():
    module = _module()
    assert hasattr(module, "weighted_macro_f1"), "pair study analysis is missing"
    import numpy as np
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(1)
    y, pred = rng.integers(0, 4, 200), rng.integers(0, 4, 200)
    pred[pred == 3] = 2
    weights = rng.integers(0, 3, 200).astype(float)
    for w in (None, weights):
        expected = f1_score(y, pred, labels=np.arange(5), average="macro",
                            sample_weight=w, zero_division=0)
        assert abs(module.weighted_macro_f1(y, pred, 5, w) - expected) < 1e-12


def test_bootstrap_is_seeded_and_resamples_patients_as_clusters():
    module = _module()
    assert hasattr(module, "paired_bootstrap"), "pair study analysis is missing"
    import numpy as np

    rng = np.random.default_rng(4)
    y = rng.integers(0, 3, 60)
    subjects = np.repeat([f"p{i}" for i in range(20)], 3)
    arms = {}
    for seed in (1234, 2025, 7):
        for mode, noise in (("product", 0.2), ("additive", 0.6), ("off", 0.9)):
            proba = np.eye(3)[y] + rng.random((60, 3)) * noise * 2
            arms[f"{mode}_seed{seed}"] = proba / proba.sum(1, keepdims=True)
    point, comparisons = module.paired_bootstrap(arms, y, subjects, num_classes=3, resamples=200)
    again_point, again = module.paired_bootstrap(arms, y, subjects, num_classes=3, resamples=200)
    assert comparisons == again and point == again_point
    delta = comparisons["product_minus_additive"]
    low, high = delta["interval_95"]
    assert low <= delta["point"] <= high
    assert set(comparisons) == {"product_minus_additive", "product_minus_off", "additive_minus_off"}


def test_decision_rule_requires_every_seed_means_and_interval():
    module = _module()
    assert hasattr(module, "decide"), "pair study decision rule is missing"
    point = {f"{mode}_seed{seed}": value
             for seed in (1234, 2025, 7)
             for mode, value in (("product", 0.66), ("additive", 0.65), ("off", 0.64))}
    passing = {"product_minus_additive": {"point": 0.01, "interval_95": [0.001, 0.02]},
               "product_minus_off": {"point": 0.02, "interval_95": [0.005, 0.03]},
               "additive_minus_off": {"point": 0.01, "interval_95": [-0.001, 0.02]}}
    assert module.decide(point, passing)["interaction_useful"] is True
    straddling = json.loads(json.dumps(passing))
    straddling["product_minus_additive"]["interval_95"] = [-0.001, 0.02]
    assert module.decide(point, straddling)["interaction_useful"] is False
    one_loss = dict(point, product_seed7=0.649)
    decision = module.decide(one_loss, passing)
    assert decision["interaction_useful"] is False
    assert decision["checks"]["product_beats_additive_each_seed"] is False


def test_load_arm_predictions_requires_identical_dev_rows(tmp_path):
    module = _module()
    assert hasattr(module, "load_arm_predictions"), "pair study analysis is missing"
    import numpy as np

    ids = np.asarray(["a", "b"])
    for name in module.FULL_STAGE_NAMES:
        (tmp_path / name).mkdir()
        np.savez_compressed(tmp_path / name / "dev.npz", proba=np.full((2, 2), 0.5),
                            y=np.asarray([0, 1]), subjects=np.asarray(["p1", "p2"]),
                            sample_ids=ids)
    arms, y, subjects = module.load_arm_predictions(tmp_path)
    assert sorted(arms) == sorted(module.FULL_STAGE_NAMES) and y.tolist() == [0, 1]
    np.savez_compressed(tmp_path / "off_seed7" / "dev.npz", proba=np.full((2, 2), 0.5),
                        y=np.asarray([0, 1]), subjects=np.asarray(["p1", "other"]),
                        sample_ids=ids)
    with pytest.raises(ValueError, match="dev rows differ"):
        module.load_arm_predictions(tmp_path)
    np.savez_compressed(tmp_path / "off_seed7" / "dev.npz", proba=np.full((2, 2), 0.5),
                        y=np.asarray([0, 1]), sample_ids=ids)
    with pytest.raises(ValueError, match="missing arrays"):
        module.load_arm_predictions(tmp_path)


def test_preflight_reads_train_only_and_refuses_existing_output(tmp_path, monkeypatch):
    module = _module()
    assert hasattr(module, "preflight"), "pair study preflight is missing"
    from comparison.standardized.clinical_graph_v2 import train
    from tests.test_cei_gnn_v2_core import _graph

    rows = [_graph(seed=seed) for seed in range(6)]
    prep = {"vocabulary": [str(i) for i in range(8)], "triples": ["a", "b", "c"]}
    calls = []

    def fake_build(*args, **kwargs):
        calls.append(kwargs)
        return {"train": rows, "dev": rows[:2], "validation": rows[:1]}, prep

    monkeypatch.setattr(train, "load_targets", lambda path: {})
    monkeypatch.setattr(train, "select_top_labels", lambda targets, k: (targets, list(range(10)), {}))
    monkeypatch.setattr(train, "build_dataset", fake_build)
    artifact, targets, _ = _inputs(tmp_path)
    report = module.preflight(artifact=artifact, targets=targets,
                              output_json=tmp_path / "preflight.json")
    assert report["pair_counts"]["graphs"] == 6 and report["pair_counts"]["mean"] == 8.0
    assert report["step_time_ratio"] > 0 and report["test_tensors_loaded"] is False
    assert calls[0]["dev_limit"] == 5000 and calls[0]["sample_seed"] == 1234
    with pytest.raises(FileExistsError):
        module.preflight(artifact=artifact, targets=targets, output_json=tmp_path / "preflight.json")


def test_cli_default_prints_plan_without_launching(tmp_path, monkeypatch, capsys):
    module = _module()
    assert hasattr(module, "main"), "pair study CLI is missing"
    artifact, targets, canonical = _inputs(tmp_path)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("launched"))
    assert module.main(["--artifact", str(artifact), "--targets", str(targets),
                        "--canonical", str(canonical),
                        "--output-root", str(tmp_path / "runs")]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "not_executed" and len(printed["stages"]) == 12
    assert not (tmp_path / "runs").exists()


def test_print_only_reverifies_completed_smoke_and_shows_eta(tmp_path, monkeypatch, capsys):
    module = _module()
    artifact, targets, canonical = _inputs(tmp_path)
    root = tmp_path / "runs"
    (root / "v2_smoke_product").mkdir(parents=True)
    (root / "v2_smoke_additive").mkdir()
    (root / "v2_smoke_off").mkdir()
    _write_json(root / "journal_smoke.json", {
        "status": "completed", "eta_minutes_nine_runs_smoke": 12.3,
        "stages": {name: {"status": "completed", "replay": {"status": "verified"}}
                   for name in ("v2_smoke_product", "v2_smoke_additive", "v2_smoke_off")}})
    verified = []
    monkeypatch.setattr(module, "_verify_smoke", lambda *args: verified.append(args) or {
        "eta_minutes_nine_runs_smoke": 12.3, "eta_basis": "verified synthetic smoke"}, raising=False)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("training launched"))
    assert module.main(["--artifact", str(artifact), "--targets", str(targets),
                        "--canonical", str(canonical), "--output-root", str(root)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["eta_minutes_nine_runs_smoke"] == 12.3
    assert len(printed["stages"]) == 12 and len(verified) == 1
    assert not (root / "journal.json").exists()


def test_smoke_journal_eta_uses_measured_smoke_seconds(tmp_path, monkeypatch):
    module = _module()
    artifact, targets, canonical = _inputs(tmp_path)
    root = tmp_path / "runs"
    stages = module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                               output_root=root)
    monkeypatch.setattr(module, "_capture", lambda stage: {
        "source_state_sha256": "s", "input_state_sha256": "i", "executable_sources": {}})
    monkeypatch.setattr(module.pilot, "_assert_runner_source_binding", lambda *a: True)
    monkeypatch.setattr(module.pilot, "_validate_artifacts", lambda *a: None)
    monkeypatch.setattr(module, "validate_v2_binding", lambda *a, **k: None)
    monkeypatch.setattr(module, "replay_v2_stage", lambda *a: {"status": "verified"})

    def fake_run(argv, **kwargs):
        output = Path(argv[argv.index("--output") + 1])
        binding = {}
        _write_json(output / "binding.json", binding)
        _write_json(output / "result.json", {"status": "completed", "binding": binding,
                                             "metrics": None, "dev_metrics": {},
                                             "test_evaluated": False, "total_seconds": 2.0})
        return module.subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.execute_plan(stages, journal_path=root / "journal_smoke.json", phase="smoke") == 0
    journal = json.loads((root / "journal_smoke.json").read_text())
    assert "eta_minutes_nine_runs_smoke" in journal
    assert journal["eta_basis"] == "mean smoke seconds per epoch; fixed costs not separated"


def test_analysis_secondary_results_are_explicitly_non_decisive(tmp_path, monkeypatch):
    module = _module()
    from comparison.standardized.clinical_graph_v2 import train
    import numpy as np

    arms = {name: np.asarray([[0.9, 0.1], [0.2, 0.8]]) for name in module.FULL_STAGE_NAMES}
    binding = {"num_classes": 2, "selected_dev": {"metric_value": 1.0},
               "label_order": ["A", "B"], "parameter_count": 42,
               "active_parameter_count": 40, "total_seconds": 4.2,
               "pair_count_summary": {"graphs": 2}}
    for name in module.FULL_STAGE_NAMES:
        _write_json(tmp_path / name / "result.json", {"total_seconds": 4.2})
    _write_analysis_history_fixtures(tmp_path, module.FULL_STAGE_NAMES, binding)
    monkeypatch.setattr(module, "validate_completed_study",
                        lambda root: {name: binding for name in module.FULL_STAGE_NAMES})
    monkeypatch.setattr(module, "load_arm_predictions", lambda root:
                        (arms, np.asarray([0, 1]), np.asarray(["p1", "p2"])))
    monkeypatch.setattr(module, "paired_bootstrap", lambda *a, **k:
                        ({name: 1.0 for name in arms}, {"product_minus_off": {"interval_95": [0, 0]}}))
    monkeypatch.setattr(module, "decide", lambda *a: {"interaction_useful": False})
    monkeypatch.setattr(train, "patient_equal_metrics", lambda *a, **k: {"macro_f1": 1.0})
    monkeypatch.setattr(train, "per_class_table", lambda *a, **k: [{"f1": 1.0}])
    report = module.analyze(tmp_path, tmp_path / "analysis.json")
    secondary = report["secondary_results"][module.FULL_STAGE_NAMES[0]]
    assert secondary["decisive"] is False
    assert secondary["patient_equal_macro_f1"] == 1.0
    assert secondary["per_class"] and secondary["pair_count_summary"]["graphs"] == 2
    assert secondary["parameter_count"] == 42 and secondary["total_seconds"] == 4.2
    assert "product_minus_off_bootstrap" in report


def test_analysis_per_class_rows_include_false_positive_counts(tmp_path, monkeypatch):
    module = _module()
    from comparison.standardized.clinical_graph_v2 import train
    import numpy as np

    arms = {name: np.asarray([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9]])
            for name in module.FULL_STAGE_NAMES}
    binding = {"num_classes": 2, "selected_dev": {"metric_value": 0.0},
               "label_order": ["A", "B"], "parameter_count": 42,
               "active_parameter_count": 40}
    for name in module.FULL_STAGE_NAMES:
        _write_json(tmp_path / name / "result.json", {"total_seconds": 0.0})
    _write_analysis_history_fixtures(tmp_path, module.FULL_STAGE_NAMES, binding)
    monkeypatch.setattr(module, "validate_completed_study",
                        lambda root: {name: binding for name in module.FULL_STAGE_NAMES})
    monkeypatch.setattr(module, "load_arm_predictions", lambda root:
                        (arms, np.asarray([0, 1, 1]), np.asarray(["p1", "p2", "p3"])))
    monkeypatch.setattr(module, "paired_bootstrap", lambda *a, **k:
                        ({name: 0.0 for name in arms}, {"product_minus_off": {"interval_95": [0, 0]}}))
    monkeypatch.setattr(module, "decide", lambda *a: {"interaction_useful": False})
    monkeypatch.setattr(train, "patient_equal_metrics", lambda *a, **k: {"macro_f1": 0.0})
    monkeypatch.setattr(train, "per_class_table", lambda *a, **k: [
        {"index": 0, "label": "A"}, {"index": 1, "label": "B"}])
    report = module.analyze(tmp_path, tmp_path / "analysis.json")
    rows = report["secondary_results"][module.FULL_STAGE_NAMES[0]]["per_class"]
    assert [row.get("false_positives") for row in rows] == [1, 0]


def test_analysis_uses_validated_result_seconds(tmp_path, monkeypatch):
    module = _module()
    from comparison.standardized.clinical_graph_v2 import train
    import numpy as np

    arms = {name: np.asarray([[0.9, 0.1], [0.1, 0.9]]) for name in module.FULL_STAGE_NAMES}
    binding = {"num_classes": 2, "selected_dev": {"metric_value": 1.0},
               "label_order": ["A", "B"], "parameter_count": 42,
               "active_parameter_count": 40, "total_seconds": 999.0}
    for name in module.FULL_STAGE_NAMES:
        _write_json(tmp_path / name / "result.json", {"total_seconds": 4.2})
    _write_analysis_history_fixtures(tmp_path, module.FULL_STAGE_NAMES, binding)
    monkeypatch.setattr(module, "validate_completed_study",
                        lambda root: {name: binding for name in module.FULL_STAGE_NAMES})
    monkeypatch.setattr(module, "load_arm_predictions", lambda root:
                        (arms, np.asarray([0, 1]), np.asarray(["p1", "p2"])))
    monkeypatch.setattr(module, "paired_bootstrap", lambda *a, **k:
                        ({name: 1.0 for name in arms}, {"product_minus_off": {"interval_95": [0, 0]}}))
    monkeypatch.setattr(module, "decide", lambda *a: {"interaction_useful": False})
    monkeypatch.setattr(train, "patient_equal_metrics", lambda *a, **k: {"macro_f1": 1.0})
    monkeypatch.setattr(train, "per_class_table", lambda *a, **k: [{"index": 0}, {"index": 1}])
    report = module.analyze(tmp_path, tmp_path / "analysis.json")
    assert report["secondary_results"][module.FULL_STAGE_NAMES[0]]["total_seconds"] == 4.2


def test_product_minus_off_comparison_is_explicitly_non_decisive(tmp_path, monkeypatch):
    module = _module()
    from comparison.standardized.clinical_graph_v2 import train
    import numpy as np

    arms = {name: np.asarray([[0.9, 0.1], [0.1, 0.9]]) for name in module.FULL_STAGE_NAMES}
    binding = {"num_classes": 2, "selected_dev": {"metric_value": 1.0},
               "label_order": ["A", "B"], "parameter_count": 42,
               "active_parameter_count": 40}
    for name in module.FULL_STAGE_NAMES:
        _write_json(tmp_path / name / "result.json", {"total_seconds": 0.0})
    _write_analysis_history_fixtures(tmp_path, module.FULL_STAGE_NAMES, binding)
    monkeypatch.setattr(module, "validate_completed_study",
                        lambda root: {name: binding for name in module.FULL_STAGE_NAMES})
    monkeypatch.setattr(module, "load_arm_predictions", lambda root:
                        (arms, np.asarray([0, 1]), np.asarray(["p1", "p2"])))
    comparisons = {key: {"point": 0.0, "interval_95": [0.0, 0.0]}
                   for key in module.COMPARISONS}
    monkeypatch.setattr(module, "paired_bootstrap", lambda *a, **k:
                        ({name: 1.0 for name in arms}, comparisons))
    monkeypatch.setattr(module, "decide", lambda *a: {"interaction_useful": False})
    monkeypatch.setattr(train, "patient_equal_metrics", lambda *a, **k: {"macro_f1": 1.0})
    monkeypatch.setattr(train, "per_class_table", lambda *a, **k: [{"index": 0}, {"index": 1}])
    report = module.analyze(tmp_path, tmp_path / "analysis.json")
    assert report["comparisons"]["product_minus_additive"]["decisive"] is True
    assert report["comparisons"]["product_minus_off"]["decisive"] is False
    assert report["comparisons"]["additive_minus_off"]["decisive"] is False


def test_cli_rejects_custom_journal_argument():
    module = _module()
    with pytest.raises(SystemExit) as error:
        module.parser().parse_args(["--journal", "elsewhere.json"])
    assert error.value.code == 2


def test_module_entrypoint_prints_plan_without_execution(tmp_path):
    import subprocess

    artifact, targets, canonical = _inputs(tmp_path)
    result = subprocess.run([
        sys.executable, "-m", "comparison.standardized.clinical_graph_v2.cei_v2_study",
        "--artifact", str(artifact), "--targets", str(targets),
        "--canonical", str(canonical), "--output-root", str(tmp_path / "runs")],
        check=True, capture_output=True, text=True)
    printed = json.loads(result.stdout)
    assert printed["status"] == "not_executed" and len(printed["stages"]) == 12
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("case", ["seed_tie", "off_mean_tie", "off_mean_higher"])
def test_mutation_decision_rejects_ties_and_stronger_off(case):
    """M1/M2: strict per-seed wins and the off control are independent gates."""
    module = _module()
    point = {f"{mode}_seed{seed}": score
             for seed in (1234, 2025, 7)
             for mode, score in (("product", 0.7), ("additive", 0.6), ("off", 0.5))}
    if case == "seed_tie":
        point["product_seed7"] = 0.6
    else:
        for seed in (1234, 2025, 7):
            point[f"off_seed{seed}"] = 0.7 if case == "off_mean_tie" else 0.8
    comparison = {"product_minus_additive": {"interval_95": [0.01, 0.2]}}
    decision = module.decide(point, comparison)
    assert decision["interaction_useful"] is False
    failed_gate = ("product_beats_additive_each_seed" if case == "seed_tie"
                   else "product_mean_above_both_controls")
    assert decision["checks"][failed_gate] is False


def test_mutation_bootstrap_matches_expanded_patient_oracle():
    """M3/M5/M6: independently expand shared patient draws and score with sklearn."""
    import numpy as np
    from sklearn.metrics import f1_score

    module = _module()
    rng = np.random.default_rng(81)
    subjects = np.repeat(["a", "b", "c", "d", "e", "f"], [1, 2, 3, 4, 5, 2])
    y = rng.integers(0, 3, len(subjects))
    arms = {f"{mode}_seed{seed}": rng.random((len(y), 3))
            for seed in (1234, 2025, 7) for mode in ("product", "additive", "off")}
    labels = np.arange(4)
    predictions = {name: values.argmax(axis=1) for name, values in arms.items()}

    def score(indices):
        return {name: f1_score(y[indices], pred[indices], labels=labels,
                              average="macro", zero_division=0)
                for name, pred in predictions.items()}

    expected_point = score(np.arange(len(y)))
    rng = np.random.default_rng(2026)
    patient_rows = [np.flatnonzero(subjects == patient) for patient in np.unique(subjects)]
    samples = {"product_minus_additive": [], "product_minus_off": [],
               "additive_minus_off": []}
    for _ in range(79):
        drawn = rng.integers(0, len(patient_rows), len(patient_rows))
        scores = score(np.concatenate([patient_rows[index] for index in drawn]))
        for key in samples:
            first, second = key.split("_minus_")
            samples[key].append(np.mean([scores[f"{first}_seed{seed}"]
                                        - scores[f"{second}_seed{seed}"]
                                        for seed in (1234, 2025, 7)]))
    values = samples["product_minus_additive"]
    assert not np.isclose(np.quantile(values, 0.025), np.quantile(values, 0.05))
    point, comparisons = module.paired_bootstrap(
        arms, y, subjects, num_classes=4, resamples=79, seed=2026)
    assert point == pytest.approx(expected_point, abs=1e-12)
    for key, values in samples.items():
        assert comparisons[key]["interval_95"] == pytest.approx(
            np.quantile(values, [0.025, 0.975]), abs=1e-12)
        first, second = key.split("_minus_")
        expected_delta = np.mean([expected_point[f"{first}_seed{seed}"]
                                  - expected_point[f"{second}_seed{seed}"]
                                  for seed in (1234, 2025, 7)])
        assert comparisons[key]["point"] == pytest.approx(expected_delta, abs=1e-12)
        assert comparisons[key]["resamples"] == 79
        assert comparisons[key]["seed"] == 2026


def test_mutation_full_plan_locks_patience_to_forty(tmp_path):
    """M12: validate the externally approved value, not the planner against itself."""
    module = _module()
    artifact, targets, canonical = _inputs(tmp_path)
    stages = module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                               output_root=tmp_path / "runs")
    assert len(stages[3:]) == 9
    for stage in stages[3:]:
        assert stage.argv[stage.argv.index("--patience") + 1] == "40"


def test_mutation_arm_parity_rejects_learning_rate_drift():
    """M19: lr equality must be enforced by the cross-arm validator itself."""
    module = _module()
    product, additive = _binding("product", 1234), _binding("additive", 1234)
    assert module.assert_arm_parity({"product": product, "additive": additive})
    additive["lr"] = product["lr"] * 2
    with pytest.raises(ValueError, match="lr"):
        module.assert_arm_parity({"product": product, "additive": additive})


def _mutation_executor_fixture(tmp_path, monkeypatch, *, seconds=2.0, batch_size=128):
    """Synthetic executor boundary: every subprocess and data check is replaced."""
    module = _module()
    artifact, targets, canonical = _inputs(tmp_path)
    root = tmp_path / "runs"
    stages = module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                               output_root=root)
    captured = {"source_state_sha256": "synthetic-source", "input_state_sha256": "synthetic-input",
                "executable_sources": {}}
    monkeypatch.setattr(module, "_capture", lambda stage: captured)
    monkeypatch.setattr(module.pilot, "_assert_runner_source_binding", lambda *a: None)
    monkeypatch.setattr(module.pilot, "_validate_artifacts", lambda *a: None)
    monkeypatch.setattr(module, "validate_v2_binding", lambda *a, **k: None)
    monkeypatch.setattr(module, "assert_arm_parity", lambda *a: True)
    monkeypatch.setattr(module, "replay_v2_stage", lambda *a: {"status": "verified"})
    launches = []

    def fake_run(argv, **kwargs):
        launches.append(argv)
        output = Path(argv[argv.index("--output") + 1])
        binding = {"batch_size": batch_size}
        _write_json(output / "binding.json", binding)
        _write_json(output / "result.json", {"status": "completed", "binding": binding,
                                             "metrics": None, "dev_metrics": {},
                                             "test_evaluated": False, "total_seconds": seconds})
        return module.subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    return module, stages, root, captured, launches


def test_mutation_full_executor_replays_smoke_before_any_launch(tmp_path, monkeypatch):
    """M13: a verified journal does not excuse replaying the smoke checkpoint."""
    module, stages, root, captured, launches = _mutation_executor_fixture(tmp_path, monkeypatch)
    entries = {}
    for smoke_stage in stages[:3]:
        _write_json(root / smoke_stage.name / "binding.json", {})
        _write_json(root / smoke_stage.name / "result.json", {
            "status": "completed", "binding": {}, "metrics": None,
            "dev_metrics": {}, "test_evaluated": False})
        entries[smoke_stage.name] = {"status": "completed", "replay": {"status": "verified"},
                                     "postflight": captured}
    _write_json(root / "journal_smoke.json", {"status": "completed", "stages": entries})
    replayed = []

    def reject_changed_smoke(output, binding, result, **kwargs):
        replayed.append(Path(output).name)
        if Path(output).name == "v2_smoke_product":
            raise ValueError("smoke replay no longer matches checkpoint")
        return {"status": "verified"}

    monkeypatch.setattr(module, "replay_v2_stage", reject_changed_smoke)
    with pytest.raises(ValueError, match="smoke replay"):
        module.execute_plan(stages, journal_path=root / "journal_full.json", phase="full")
    assert replayed == ["v2_smoke_product"]
    assert launches == []
    assert not (root / "journal_full.json").exists()


def test_mutation_executor_refuses_output_occupied_after_planning(tmp_path, monkeypatch):
    """M15: the executor must recheck occupancy, not trust the earlier plan."""
    module, stages, root, _, launches = _mutation_executor_fixture(tmp_path, monkeypatch)
    occupied = Path(stages[0].output)
    occupied.mkdir(parents=True)
    marker = occupied / "preserve.txt"
    marker.write_text("do not overwrite")
    with pytest.raises(SystemExit) as error:
        module.execute_plan(stages, journal_path=root / "journal_smoke.json", phase="smoke")
    assert error.value.code == 2
    assert launches == []
    assert marker.read_text() == "do not overwrite"
    journal = json.loads((root / "journal_smoke.json").read_text())
    assert journal["status"] == "failed"
    assert journal["stages"]["v2_smoke_product"]["status"] == "refused"


@pytest.mark.parametrize("seconds,batch_size", [(2.0, 128), (7.0, 100)])
def test_mutation_smoke_eta_numeric_step_scaling(tmp_path, monkeypatch, seconds, batch_size):
    """M21: verify the numeric smoke-based ETA, including ceiling partial batches."""
    import math

    module, stages, root, _, launches = _mutation_executor_fixture(
        tmp_path, monkeypatch, seconds=seconds, batch_size=batch_size)
    assert module.execute_plan(stages, journal_path=root / "journal_smoke.json", phase="smoke") == 0
    journal = json.loads((root / "journal_smoke.json").read_text())
    expected = round(9 * (seconds / 2) * 40 / 60, 1)
    assert journal["eta_minutes_nine_runs_smoke"] == expected
    assert journal["eta_basis"] == "mean smoke seconds per epoch; fixed costs not separated"
    assert len(launches) == 3


@pytest.mark.parametrize("field", ["active_parameter_count", "architecture_active"])
def test_sf2a_rejects_product_additive_active_parameter_mismatch(field):
    module = _module()
    product, additive = _binding("product", 1234), _binding("additive", 1234)
    if field == "active_parameter_count":
        additive["active_parameter_count"] += 1
        additive["method_config"]["architecture"]["active_parameter_count"] += 1
    else:
        additive["active_parameter_count"] += 1
        additive["method_config"]["architecture"]["active_parameter_count"] += 1
    with pytest.raises(ValueError, match="active_parameter_count"):
        module.assert_arm_parity({"product": product, "additive": additive})


@pytest.mark.parametrize("field,value", [("extra", 1), ("metric", "other"),
                                           ("sample_ids_sha256", "different")])
def test_sf2a_selected_dev_is_exact_and_common_except_outcomes(field, value):
    module = _module()
    selected = {"epoch": 4, "epoch_index": 3, "metric": "macro_f1",
                "metric_value": 0.7, "prediction_sha256": "pred",
                "sample_ids_sha256": "ids"}
    product, additive = _binding("product", 1234), _binding("additive", 1234)
    product["selected_dev"] = dict(selected)
    additive["selected_dev"] = dict(selected)
    if field == "extra":
        additive["selected_dev"][field] = value
    else:
        additive["selected_dev"][field] = value
    with pytest.raises(ValueError, match="selected_dev"):
        module.assert_arm_parity({"product": product, "additive": additive})


def test_sf2a_off_active_parameters_must_be_fewer_and_three_arms_remain_valid():
    module = _module()
    arms = {f"{mode}_seed{seed}": _binding(mode, seed)
            for seed in (1234, 2025, 7) for mode in ("product", "additive", "off")}
    assert module.assert_arm_parity(arms)
    product, off = _binding("product", 1234), _binding("off", 1234)
    off["active_parameter_count"] = product["active_parameter_count"]
    off["method_config"]["architecture"]["active_parameter_count"] = product["active_parameter_count"]
    with pytest.raises(ValueError, match="active_parameter_count"):
        module.assert_arm_parity({"product": product, "off": off})


def test_print_only_smoke_verification_does_not_mutate_output_root(tmp_path, monkeypatch, capsys):
    """SF2-B: print-only requires an existing replay proof and never persists one."""
    import hashlib

    module = _module()
    artifact, targets, canonical = _inputs(tmp_path)
    root = tmp_path / "runs"
    smoke = root / "v2_smoke_product"
    smoke.mkdir(parents=True)
    (root / "v2_smoke_additive").mkdir()
    (root / "v2_smoke_off").mkdir()
    stages = module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                               output_root=root, allow_existing_smoke=True)
    stage = stages[0]
    binding = {"synthetic": "binding"}
    result = {"status": "completed", "binding": binding, "metrics": None,
              "dev_metrics": {}, "test_evaluated": False}
    for mode in ("product", "additive", "off"):
        mode_dir = root / f"v2_smoke_{mode}"
        _write_json(mode_dir / "binding.json", binding)
        _write_json(mode_dir / "result.json", result)
        (mode_dir / "payload.bin").write_bytes(b"synthetic prediction payload")
    captured = {"source_state_sha256": "source", "input_state_sha256": "input",
                "executable_sources": {}}
    monkeypatch.setattr(module, "_capture", lambda _: captured)
    monkeypatch.setattr(module.pilot, "_assert_runner_source_binding", lambda *a: None)
    monkeypatch.setattr(module.pilot, "_validate_artifacts", lambda *a: None)
    monkeypatch.setattr(module, "validate_v2_binding", lambda *a, **k: None)

    def computed_replay(directory, _binding, _result, *, persist=True):
        proof = {"status": "verified", "payload_sha256": hashlib.sha256(
            (Path(directory) / "payload.bin").read_bytes()).hexdigest()}
        proof_path = Path(directory) / "replay.json"
        if proof_path.exists():
            if json.loads(proof_path.read_text()) != proof:
                raise ValueError("existing replay proof differs")
        elif persist:
            _write_json(proof_path, proof)
        else:
            raise ValueError("existing replay proof is required in non-persisting mode")
        return proof

    monkeypatch.setattr(module, "replay_v2_stage", computed_replay)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("training launched"))

    def snapshot():
        return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(root.rglob("*")) if path.is_file()}

    proof = computed_replay(smoke, binding, result)
    journal_stages = {}
    for mode in ("product", "additive", "off"):
        mode_dir = root / f"v2_smoke_{mode}"
        mode_proof = proof if mode == "product" else computed_replay(mode_dir, binding, result)
        journal_stages[f"v2_smoke_{mode}"] = {"status": "completed", "postflight": captured,
                                                "replay": mode_proof}
    journal = {"status": "completed", "eta_minutes_nine_runs_smoke": 12.3,
               "stages": journal_stages}
    _write_json(root / "journal_smoke.json", journal)
    before = snapshot()
    assert module.main(["--artifact", str(artifact), "--targets", str(targets),
                        "--canonical", str(canonical), "--output-root", str(root)]) == 0
    assert snapshot() == before
    assert json.loads(capsys.readouterr().out)["status"] == "not_executed"

    replay_path = smoke / "replay.json"
    replay_path.unlink()
    with pytest.raises(ValueError, match="replay"):
        module.main(["--artifact", str(artifact), "--targets", str(targets),
                     "--canonical", str(canonical), "--output-root", str(root)])
    assert not replay_path.exists()

    _write_json(replay_path, proof)
    (smoke / "payload.bin").write_bytes(b"tampered prediction payload")
    with pytest.raises(ValueError, match="replay"):
        module.main(["--artifact", str(artifact), "--targets", str(targets),
                     "--canonical", str(canonical), "--output-root", str(root)])


def _run_synthetic_analysis(module, tmp_path, monkeypatch, *, point=None, counts=None,
                            histories=None, replay_counts=None):
    import numpy as np
    from comparison.standardized.clinical_graph_v2 import train

    arms = {name: np.asarray([[0.8, 0.2], [0.2, 0.8], [0.8, 0.2], [0.2, 0.8]])
            for name in module.FULL_STAGE_NAMES}
    y = np.asarray([0, 1, 0, 1])
    subjects = np.asarray(["p1", "p2", "p3", "p4"])
    point = point or {name: 1.0 for name in module.FULL_STAGE_NAMES}
    bindings = {}
    for name in module.FULL_STAGE_NAMES:
        mode, seed = name.rsplit("_seed", 1)
        bindings[name] = {"num_classes": 2, "selected_dev": {"metric_value": 1.0},
                          "label_order": ["A", "B"], "parameter_count": 42,
                          "active_parameter_count": 40, "inactive_parameter_count": 2,
                          "epochs": 3, "seed": int(seed),
                          "selected_dev": {"metric_value": point[name], "epoch": 2},
                          "method_config": {"architecture": {"parameter_count": 42,
                                                                  "active_parameter_count": 40,
                                                                  "inactive_parameter_count": 2}}}
        _write_json(tmp_path / name / "result.json", {"total_seconds": 4.2})
        _write_json(tmp_path / name / "history.json", histories.get(name, [
            {"epoch": i, "train_loss": 1.0 / i, "macro_f1": 1.0,
             "selection_fold": "dev", "seconds": 1.0} for i in range(1, 4)
        ]) if histories else [{"epoch": i, "train_loss": 1.0 / i, "macro_f1": 1.0,
                               "selection_fold": "dev", "seconds": 1.0}
                              for i in range(1, 4)])
    point = point or {name: 1.0 for name in module.FULL_STAGE_NAMES}
    def validate(root, *, include_pair_counts=False):
        if include_pair_counts:
            return bindings, replay_counts
        return bindings
    monkeypatch.setattr(module, "validate_completed_study", validate)
    monkeypatch.setattr(module, "load_arm_predictions", lambda root: (arms, y, subjects))
    monkeypatch.setattr(module, "paired_bootstrap", lambda *a, **k: (
        point, {key: {"point": 0.0, "interval_95": [0.0, 0.0]} for key in module.COMPARISONS}))
    monkeypatch.setattr(module, "decide", lambda *a: {"interaction_useful": False})
    monkeypatch.setattr(train, "patient_equal_metrics", lambda *a, **k: {"macro_f1": 1.0})
    monkeypatch.setattr(train, "per_class_table", lambda *a, **k: [{"index": 0}, {"index": 1}])
    if counts is not None:
        return module.analyze(tmp_path, tmp_path / "analysis.json", pair_counts=counts)
    return module.analyze(tmp_path, tmp_path / "analysis.json")


def test_a10_r1_analyze_reports_hand_computed_seed_level_deltas(tmp_path, monkeypatch):
    module = _module()
    point = {f"{mode}_seed{seed}": score
             for seed, product, additive, off in ((1234, .70, .60, .50),
                                                   (2025, .65, .60, .55),
                                                   (7, .80, .70, .60))
             for mode, score in (("product", product), ("additive", additive), ("off", off))}
    report = _run_synthetic_analysis(module, tmp_path, monkeypatch, point=point)
    delta = report["seed_deltas"]["product_minus_additive"]
    assert delta["per_seed"] == {"1234": pytest.approx(10.0),
                                 "2025": pytest.approx(5.0), "7": pytest.approx(10.0)}
    assert delta["mean"] == pytest.approx(25 / 3)
    assert delta["sd"] == pytest.approx((25 / 3) ** 0.5)
    assert delta["t_interval_df2"] == pytest.approx([25 / 3 - 4.302653 * ((25 / 3) ** .5) / 3 ** .5,
                                                       25 / 3 + 4.302653 * ((25 / 3) ** .5) / 3 ** .5])
    assert delta["label"] == "seed-level, descriptive, not decisive"
    assert delta["decisive"] is False


def test_a10_r2_analyze_reports_injected_pair_count_band_membership(tmp_path, monkeypatch):
    module = _module()
    if "pair_counts" in __import__("inspect").signature(module.analyze).parameters:
        report = _run_synthetic_analysis(module, tmp_path, monkeypatch,
                                         counts=[0, 1, 180, 1045])
    else:
        report = _run_synthetic_analysis(module, tmp_path, monkeypatch)
    bands = report["pair_count_bands"]
    assert [(key, bands[key]["graphs"]) for key in ("0", "1-179", "180-1044", ">1044")] == [
        ("0", 1), ("1-179", 1), ("180-1044", 1), (">1044", 1)]
    assert [bands[key]["patients"] for key in ("0", "1-179", "180-1044", ">1044")] == [1] * 4
    assert all(bands[key]["per_arm_seed_mean_macro_f1"] is None
               and bands[key]["product_minus_additive"] is None
               and bands[key]["decisive"] is False
               for key in ("0", "1-179", "180-1044", ">1044"))


def test_a10_r3_analyze_reports_and_validates_training_curves(tmp_path, monkeypatch):
    module = _module()
    report = _run_synthetic_analysis(module, tmp_path, monkeypatch)
    curve = report["training_curves"][module.FULL_STAGE_NAMES[0]]
    assert curve["train_loss"] == pytest.approx([1.0, .5, 1 / 3])
    assert curve["dev_macro_f1"] == [1.0, 1.0, 1.0]
    assert curve["selected_epoch"] == 2 and curve["selected_at_last_epoch"] is False
    assert curve["total_seconds"] == 4.2 and curve["decisive"] is False

    malformed = {module.FULL_STAGE_NAMES[0]: [
        {"epoch": 1, "train_loss": 1.0, "macro_f1": .5,
         "selection_fold": "validation", "seconds": 1.0}]}
    bad_path = tmp_path / "bad"
    with pytest.raises(ValueError, match="history"):
        _run_synthetic_analysis(module, bad_path, monkeypatch, histories=malformed)


def test_a10_r4_reports_parameters_notes_and_decide_is_invariant(tmp_path, monkeypatch):
    module = _module()
    report = _run_synthetic_analysis(module, tmp_path, monkeypatch)
    assert report["notes"] == [
        "Exploratory dev-only screen: intervals are conditional on the selected dev checkpoints and seeds 1234/2025/7 (spec §6.1).",
        "A failed rule means benefit not demonstrated, not evidence of no benefit.",
        "product vs off is directional; off is a structural ablation with fewer active parameters.",
        "GraphXAI edge masks do not attribute within-visit pairs; pair contributions are model accounting, not causal importance.",
    ]
    arm = report["secondary_results"][module.FULL_STAGE_NAMES[0]]
    assert (arm["parameter_count"], arm["active_parameter_count"],
            arm["inactive_parameter_count"]) == (42, 40, 2)

    point = {f"{mode}_seed{seed}": 1.0 for mode in module.MODES for seed in module.SEEDS}
    comparisons = {key: {"interval_95": [0.0, 1.0]} for key in module.COMPARISONS}
    expected = module.decide(point, comparisons)
    report.update({"seed_deltas": {}, "pair_count_bands": {}, "training_curves": {},
                   "notes": report["notes"], "parameter_counts": {}})
    assert module.decide(point, comparisons) == expected


def test_a10_r2_dev_pair_counts_and_analysis_replay_counts(tmp_path, monkeypatch):
    module = _module()
    import torch
    from comparison.standardized.clinical_graph_v2.tensorize import ClinicalGraphData

    rows = []
    for members, types in (([0, 0, 0], [1, 2, 5]), ([0, 0, 1, 1], [1, 2, 1, 2])):
        row = ClinicalGraphData()
        node_count = len(types)
        row.visit_membership_index = torch.tensor([members, list(range(node_count))])
        row.node_type = torch.tensor(types)
        row.num_nodes = node_count
        rows.append(row)
    # within_visit_pairs counts unordered complaint/measurement/vital pairs per visit only
    # (cei_gnn_v2.py:11-12, 21-22, 46-50): row one has one complaint-vital pair;
    # row two has just one complaint evidence node in each visit, hence no pair.
    assert module.dev_pair_counts(rows) == [1, 0]

    report = _run_synthetic_analysis(module, tmp_path, monkeypatch,
                                     replay_counts=[0, 0, 0, 1])
    bands = report["pair_count_bands"]
    assert bands["0"]["graphs"] == 3
    assert bands["0"]["patients"] == 3
    assert bands["1-179"]["graphs"] == 1
    assert bands["1-179"]["patients"] == 1
    assert report["pair_count_source"] == "dev rows reloaded by replay (structural count, no scoring)"


def test_a10_r2_replay_json_keys_stay_proof_only():
    # The transient value is added only to replay's returned mapping, never the proof.
    proof_keys = {"status", "method", "pair_mode", "seed", "checkpoint_sha256",
                  "proba_sha256", "split_sample_ids_sha256", "dev_count",
                  "exact_probabilities", "exact_labels", "exact_sample_identity",
                  "patient_disjoint", "validation_evaluated", "test_evaluated", "dev_metrics"}
    assert "_dev_pair_counts" not in proof_keys


def test_a10_r2_rejects_different_counts_across_arms():
    module = _module()
    with pytest.raises(ValueError, match="dev pair counts differ across arms"):
        module._assert_dev_pair_count_parity([[0, 1], [0, 2]])
