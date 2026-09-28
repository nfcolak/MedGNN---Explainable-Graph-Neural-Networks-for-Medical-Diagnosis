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


def test_plan_has_exact_ten_dev_only_stages(tmp_path):
    module = _module()
    artifact, targets, canonical = _inputs(tmp_path)
    stages = module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                               output_root=tmp_path / "runs")
    expected = [("v2_smoke", (256, 128, 2), 1234, "product")] + [
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
    with pytest.raises(ValueError, match="ten-stage"):
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


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


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
    assert journal["status"] == "failed" and journal["stages"]["v2_smoke"]["status"] == "failed"
    assert len(launches) == 1 and launches[0][-1] == "--execute"


def test_executor_refuses_source_drift_between_full_stages(tmp_path, monkeypatch):
    module = _module()
    assert hasattr(module, "execute_plan"), "pair study executor is missing"
    artifact, targets, canonical = _inputs(tmp_path)
    root = tmp_path / "runs"
    stages = module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                               output_root=root)
    _write_json(root / "v2_smoke" / "binding.json", {})
    _write_json(root / "v2_smoke" / "result.json", {"status": "completed", "binding": {},
                                                       "metrics": None, "dev_metrics": {},
                                                       "test_evaluated": False})
    smoke_capture = {"source_state_sha256": "a", "input_state_sha256": "i",
                     "executable_sources": {}}
    _write_json(root / "journal_smoke.json", {"status": "completed", "stages": {
        "v2_smoke": {"status": "completed", "replay": {"status": "verified"},
                     "postflight": smoke_capture}}})
    identities = iter(["a", "a", "a", "b"])
    monkeypatch.setattr(module, "_capture", lambda stage: {
        "source_state_sha256": next(identities), "input_state_sha256": "i",
        "executable_sources": {}})
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
    state["network.bias"] = state["network.bias"] + 0.5
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
    _write_json(root / "v2_smoke" / "result.json", {"status": "completed"})
    _write_json(root / "journal_smoke.json", {"status": "completed"})
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("launched"))
    with pytest.raises(ValueError, match="smoke"):
        module.execute_plan(stages, journal_path=root / "journal_full.json", phase="full")
    assert not (root / "journal_full.json").exists()
