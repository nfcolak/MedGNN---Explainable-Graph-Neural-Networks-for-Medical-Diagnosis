"""Synthetic contract tests for the bounded CEI development pilot."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = (Path(__file__).parents[1] / "comparison/standardized/clinical_graph_v2/cei_pilot.py")


def _module():
    # Keep the RED failure an assertion about the missing behavior, not collection/import.
    assert MODULE_PATH.is_file(), "CEI pilot protocol module must implement the locked planner"
    spec = importlib.util.spec_from_file_location("cei_pilot_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    import sys
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _bindings(**updates):
    base = {
        "method": "cei_gnn",
        "artifact_graphs_sha256": "graph-hash",
        "artifact_visit_membership_sha256": "membership-hash",
        "targets_sha256": "targets-hash",
        "target_binding_sha256": "target-binding-hash",
        "label_order": ["A", "B"],
        "source_code": {"methods/cei.py": "source-hash"},
        "preprocessing_sha256": "prep-hash",
        "split_sample_ids_sha256": {"train": "train-hash", "dev": "dev-hash"},
        "seed": 1234,
        "sample_seed": 1234,
        "selection_fold": "dev",
        "final_eval": "none",
        "train_limit": 10000,
        "dev_limit": 5000,
        "epochs": 40,
        "patience": 40,
        "test_evaluated": False,
        "parameter_count": 123,
        "active_parameter_count": 120,
        "method_config": {"architecture": {"parameter_count": 123, "active_parameter_count": 120}},
        "edge_direction": "forward", "edges": "all", "top_k_labels": 10,
        "message_passing": True, "edge_payload": True, "num_classes": 2,
        "input_contract_version": "prep-v1", "weights": "sqrt_inverse",
    }
    base.update(updates)
    return base


def test_planner_builds_exact_bounded_four_stages_with_separate_seeds(tmp_path):
    module = _module()
    artifact = tmp_path / "artifact"
    targets = tmp_path / "targets.csv"
    canonical = tmp_path / "canonical.json"
    artifact.mkdir()
    targets.write_text("sample_id,subject_id,target,split\n")
    canonical.write_text(json.dumps({"classes": []}))
    output = tmp_path / "runs"
    output.mkdir()

    stages = module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                               output_root=output)

    assert [stage.name for stage in stages] == [
        "cei_smoke", "protgnn_control", "cei_candidate", "cei_product_off"
    ]
    assert len({stage.output for stage in stages}) == 4
    assert all(Path(stage.output).is_absolute() for stage in stages)
    assert [stage.budget for stage in stages] == [(256, 128, 2), (10000, 5000, 40),
                                                  (10000, 5000, 40), (10000, 5000, 40)]
    for stage in stages:
        argv = stage.argv
        assert isinstance(argv, list)
        assert "--execute" not in argv
        assert argv[argv.index("--sample-seed") + 1] == "1234"
        assert argv[argv.index("--selection-fold") + 1] == "dev"
        assert argv[argv.index("--final-eval") + 1] == "none"
        assert argv[argv.index("--seed") + 1] == str(stage.seed)
    assert all(stage.seed == 1234 for stage in stages)


def test_plan_refuses_missing_or_occupied_paths(tmp_path):
    module = _module()
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    targets = tmp_path / "targets.csv"
    targets.write_text("x")
    canonical = tmp_path / "canonical.json"
    canonical.write_text("{}")
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "cei_smoke").mkdir()
    with pytest.raises(FileExistsError, match="cei_smoke"):
        module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                          output_root=output)
    with pytest.raises(FileNotFoundError, match="artifact"):
        module.build_plan(artifact=tmp_path / "absent", targets=targets, canonical=canonical,
                          output_root=tmp_path / "fresh")


def test_common_bindings_compare_nested_source_and_require_contract_fields():
    module = _module()
    left, right = _bindings(), _bindings()
    module.assert_common_bindings(left, right)
    right["source_code"]["methods/cei.py"] = "changed"
    with pytest.raises(ValueError, match="source_code"):
        module.assert_common_bindings(left, right)
    for missing in ("artifact_graphs_sha256", "source_code", "split_sample_ids_sha256"):
        malformed = _bindings()
        del malformed[missing]
        with pytest.raises(ValueError, match=missing):
            module.assert_common_bindings(malformed, _bindings())


def test_only_declared_treatment_fields_may_differ_and_unknowns_stay_strict():
    module = _module()
    left = _bindings(method="protgnn", use_interactions=None)
    right = _bindings(method="cei_gnn", use_interactions=False)
    module.assert_common_bindings(left, right)
    right["split_sample_ids_sha256"]["dev"] = "different-dev-sample"
    with pytest.raises(ValueError, match="split_sample_ids_sha256.dev"):
        module.assert_common_bindings(left, right)
    right = _bindings(method="cei_gnn", use_interactions=False)
    right["surprise"] = "not silently ignored"
    with pytest.raises(ValueError, match="surprise"):
        module.assert_common_bindings(left, right)


def test_report_validator_rejects_incomplete_and_smoke_is_not_an_arm(tmp_path):
    module = _module()
    incomplete = tmp_path / "cei_candidate"
    incomplete.mkdir()
    (incomplete / "binding.json").write_text(json.dumps(_bindings()))
    others = []
    for name in ("protgnn_control", "cei_product_off"):
        directory = tmp_path / name
        directory.mkdir()
        others.append(directory)
    with pytest.raises(ValueError, match="result.json"):
        module.validate_completed_stages([incomplete, *others])

    stages = []
    for name, treatment in (("protgnn_control", {"method": "protgnn", "use_interactions": None}),
                            ("cei_candidate", {"method": "cei_gnn", "use_interactions": True}),
                            ("cei_product_off", {"method": "cei_gnn", "use_interactions": False})):
        directory = tmp_path / name
        directory.mkdir(exist_ok=True)
        binding = _bindings(**treatment)
        binding["method_config"] = {
            "method": binding["method"],
            "effective_settings": ({"use_interactions": treatment.get("use_interactions")}
                                   if binding["method"] == "cei_gnn" else {}),
            "architecture": {"parameter_count": 123, "active_parameter_count": 120},
        }
        result = {"status": "completed", "binding": binding, "metrics": None,
                  "dev_metrics": {"macro_f1": 0.1}, "test_evaluated": False}
        (directory / "binding.json").write_text(json.dumps(binding))
        (directory / "result.json").write_text(json.dumps(result))
        stages.append(directory)
    with pytest.raises(ValueError, match="history.json|dev.npz|best.pt"):
        module.validate_completed_stages(stages)


def test_executor_journals_failures_and_checks_bindings_before_results(tmp_path, monkeypatch):
    module = _module()
    output = tmp_path / "out"
    output.mkdir()
    journal = output / "journal.json"
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    targets, canonical = tmp_path / "targets.csv", tmp_path / "canonical.json"
    targets.write_text("x")
    canonical.write_text("{}")
    stages = module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                               output_root=output)
    monkeypatch.setattr(module, "capture_bindings", lambda stage: {
        "git_revision": "rev", "source_state_sha256": "source", "argv_sha256": "argv",
        "inputs": {},
    })
    calls = []
    def fail_run(argv, **kwargs):
        calls.append(argv)
        raise module.subprocess.CalledProcessError(2, argv)
    monkeypatch.setattr(module.subprocess, "run", fail_run)
    with pytest.raises(SystemExit) as error:
        module.execute_plan(stages, journal_path=journal)
    assert error.value.code != 0
    saved = json.loads(journal.read_text())
    assert saved["stages"]["cei_smoke"]["status"] == "failed"
    assert calls


def test_executor_refuses_source_drift_between_stages(tmp_path, monkeypatch):
    module = _module()
    root = tmp_path / "runs"
    root.mkdir()
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    targets, canonical = tmp_path / "targets.csv", tmp_path / "canonical.json"
    targets.write_text("x")
    canonical.write_text("{}")
    stages = module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                               output_root=root)
    tampered = list(stages)
    tampered[1] = module.Stage(**{**tampered[1].__dict__,
                                 "argv": tampered[1].argv[:-1]})
    launches = []
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: launches.append(a))
    with pytest.raises(ValueError, match="argv"):
        module.execute_plan(tampered, journal_path=root / "journal.json")
    assert launches == []
    assert not (root / "journal.json").exists()


def test_budget_guard_rejects_changed_full_budget_and_patience():
    module = _module()
    valid = _bindings(patience=40)
    module.validate_pilot_binding(valid, expected_budget=(10000, 5000, 40), expected_seed=1234)
    with pytest.raises(ValueError, match="budget"):
        module.validate_pilot_binding(_bindings(train_limit=9999, patience=40),
                                      expected_budget=(10000, 5000, 40), expected_seed=1234)
    with pytest.raises(ValueError, match="patience"):
        module.validate_pilot_binding(_bindings(patience=1),
                                      expected_budget=(10000, 5000, 40), expected_seed=1234)
    with pytest.raises(ValueError, match="test"):
        module.validate_pilot_binding(_bindings(test_evaluated=True, patience=40),
                                      expected_budget=(10000, 5000, 40), expected_seed=1234)


def test_actual_method_configs_compare_common_settings_but_allow_cei_treatment():
    module = _module()
    common = _bindings(method="cei_gnn", patience=40)
    common["method_config"] = {
        "method": "cei_gnn", "adaptation_version": "cei-v1",
        "native_defaults": {"hidden": 64, "interaction_rank": 8},
        "effective_settings": {"hidden": 64, "interaction_rank": 8, "use_interactions": True},
        "architecture": {"parameter_count": 321, "active_parameter_count": 321},
    }
    off = json.loads(json.dumps(common))
    off["method_config"]["effective_settings"]["use_interactions"] = False
    off["use_interactions"] = False
    assert module.assert_common_bindings(common, off)
    off["method_config"]["effective_settings"]["unreviewed_nested_flag"] = True
    with pytest.raises(ValueError, match="effective_settings"):
        module.assert_common_bindings(common, off)


def test_capture_bindings_detects_same_path_input_byte_drift(tmp_path, monkeypatch):
    module = _module()
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "manifest.json").write_text("first")
    targets = tmp_path / "targets.csv"
    targets.write_bytes(b"targets-a")
    canonical = tmp_path / "canonical.json"
    canonical.write_bytes(b"canonical")
    stage = module.Stage("arm", ["python", "train.py", "--artifact", str(artifact),
        "--targets", str(targets), "--canonical", str(canonical)], str(tmp_path / "out"),
        1234, (10000, 5000, 40), "candidate")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="rev\n"))
    before = module.capture_bindings(stage)
    targets.write_bytes(b"targets-b")
    after = module.capture_bindings(stage)
    assert module._execution_identity(before) != module._execution_identity(after)


def test_executor_refuses_occupied_journal_without_modifying_it(tmp_path):
    module = _module()
    journal = tmp_path / "journal.json"
    journal.write_text("preserve evidence\\n")
    stage = module.Stage("cei_smoke", ["python", "train.py"], str(tmp_path / "cei_smoke"),
                         1234, (256, 128, 2), "smoke")
    with pytest.raises((FileExistsError, ValueError)):
        module.execute_plan([stage], journal_path=journal)
    assert journal.read_text() == "preserve evidence\\n"


def test_validator_refuses_json_only_arms_and_nonfinite_history(tmp_path):
    module = _module()
    dirs = []
    for name, method, treatment in (("protgnn_control", "protgnn", None),
                                    ("cei_candidate", "cei_gnn", True),
                                    ("cei_product_off", "cei_gnn", False)):
        directory = tmp_path / name
        directory.mkdir()
        binding = _bindings(method=method, patience=40)
        binding["method_config"] = {
        "method": binding["method"],
        "effective_settings": ({"use_interactions": treatment, "hidden": 64}
                               if method == "cei_gnn" else {}),
        "architecture": {"parameter_count": 123, "active_parameter_count": 120},
        }
        (directory / "binding.json").write_text(json.dumps(binding))
        (directory / "result.json").write_text(json.dumps({"status": "completed", "binding": binding,
            "metrics": None, "dev_metrics": {"macro_f1": 0.1}, "test_evaluated": False,
            "history": [{"epoch": 1, "train_loss": 1.0, "seconds": 1.0}], "total_seconds": 1.0}))
        dirs.append(directory)
    with pytest.raises(ValueError, match="dev.npz|best.pt|replay"):
        module.validate_completed_stages(dirs)


def test_execution_rejects_any_stage_argv_drift_before_first_subprocess(tmp_path, monkeypatch):
    module = _module()
    root = tmp_path / "runs"
    root.mkdir()
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    targets = tmp_path / "targets.csv"
    targets.write_text("x")
    canonical = tmp_path / "canonical.json"
    canonical.write_text("{}")
    stages = module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                               output_root=root)
    tampered = list(stages)
    tampered[2] = module.Stage(**{**tampered[2].__dict__,
                                 "argv": tampered[2].argv + ["--patience", "1"]})
    launches = []
    monkeypatch.setattr(module, "capture_bindings", lambda stage: {
        "source_state_sha256": "source", "input_state_sha256": "input",
        "common_config_sha256": "config"})
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: launches.append(a))
    with pytest.raises((ValueError, SystemExit)):
        module.execute_plan(tampered, journal_path=root / "journal.json")
    assert launches == []
    assert not (root / "journal.json").exists()


def test_actual_method_run_configs_are_accepted_and_source_defaults_are_locked():
    module = _module()
    from comparison.standardized.clinical_graph_v2.methods import build_method
    from comparison.standardized.clinical_graph_v2.tensorize import PAYLOAD_WIDTH

    args = SimpleNamespace(num_relations=3)
    common = dict(num_tokens=8, node_dim=3, edge_dim=PAYLOAD_WIDTH, num_classes=2,
                 hidden=4, layers=1, dropout=0.0, token_dim=3,
                 num_triples=2, args=args)
    cei_on = build_method("cei_gnn", **common)
    cei_off = build_method("cei_gnn", **{**common,
        "args": SimpleNamespace(num_relations=3,
                                 method_options={"use_interactions": False})})
    prot = build_method("protgnn", **{**common,
        "args": SimpleNamespace(epochs=40, patience=40)})
    on_config, off_config, prot_config = cei_on.run_config(), cei_off.run_config(), prot.run_config()
    left = _bindings(method="cei_gnn", method_config=on_config, patience=40)
    right = _bindings(method="cei_gnn", method_config=off_config, patience=40)
    left["use_interactions"], right["use_interactions"] = True, False
    assert module.assert_common_bindings(left, right)
    control = _bindings(method="protgnn", method_config=prot_config, patience=40,
                        use_interactions=None)
    candidate = _bindings(method="cei_gnn", method_config=on_config, patience=40,
                          use_interactions=True)
    assert module.assert_common_bindings(control, candidate)
    drifted = json.loads(json.dumps(prot_config))
    drifted["native_schedule"]["warm_epochs"] += 1
    with pytest.raises(ValueError, match="default|schedule|method_config"):
        module.assert_common_bindings(control, _bindings(method="protgnn",
            method_config=drifted, patience=40, use_interactions=None))


def test_common_comparator_accepts_runner_native_early_stop_start_per_method():
    module = _module()
    from comparison.standardized.clinical_graph_v2.methods import build_method
    from comparison.standardized.clinical_graph_v2.tensorize import PAYLOAD_WIDTH

    common = dict(num_tokens=8, node_dim=3, edge_dim=PAYLOAD_WIDTH, num_classes=2,
                  hidden=4, layers=1, dropout=0.0, token_dim=3,
                  num_triples=2)
    prot = build_method("protgnn", **common,
                        args=SimpleNamespace(epochs=40, patience=40, num_relations=3))
    cei = build_method("cei_gnn", **common,
                       args=SimpleNamespace(epochs=40, patience=40, num_relations=3,
                                            method_options={"use_interactions": True}))
    control = _bindings(method="protgnn", method_config=prot.run_config(),
                        early_stopping_start_epoch_index=prot.proj_epochs,
                        use_interactions=None, patience=40)
    candidate = _bindings(method="cei_gnn", method_config=cei.run_config(),
                          early_stopping_start_epoch_index=0,
                          use_interactions=True, patience=40)
    assert module.assert_common_bindings(control, candidate)
    wrong_schedule = dict(control, early_stopping_start_epoch_index=0)
    with pytest.raises(ValueError, match="schedule|early_stopping"):
        module.assert_common_bindings(wrong_schedule, candidate)



def test_replay_stage_reconstructs_real_model_and_rejects_tampered_artifacts(tmp_path, monkeypatch):
    module = _module()
    import hashlib
    import numpy as np
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from comparison.standardized.clinical_graph_v2 import train
    from comparison.standardized.clinical_graph_v2.contracts import recursive_source_hashes, sample_ids_sha256
    from comparison.standardized.clinical_graph_v2.methods import build_method
    from comparison.standardized.clinical_graph_v2.schema import sha256
    from comparison.standardized.clinical_graph_v2.tensorize import (
        PAYLOAD_WIDTH, PREPROCESSING_VERSION, Scaler, Vocabulary, preprocessing_state)

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "graphs.jsonl").write_text("synthetic graph source\n")
    (artifact / "visit_membership.jsonl").write_text("synthetic membership\n")
    targets_path = tmp_path / "targets.csv"
    targets_path.write_text("synthetic targets\n")
    prep = {"preprocessing_version": PREPROCESSING_VERSION,
            "context_categories": {"gender": [], "race": [], "arrival_transport": []},
            "vocabulary": Vocabulary(["synthetic"], 1), "scaler": Scaler({}),
            "token_min_count": 1, "triples": ["patient|rel|visit"], "triple_counts": {}}
    prep_json = preprocessing_state(prep)
    output = tmp_path / "stage"
    output.mkdir()
    (output / "preprocessing.json").write_text(json.dumps(prep_json, sort_keys=True))
    rows = {}
    for sid, subject, label in (("train-1", "patient-1", 0), ("dev-1", "patient-2", 0),
                                ("dev-2", "patient-3", 1), ("validation-1", "patient-4", 1)):
        rows[sid] = Data(
            x=torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32),
            token=torch.tensor([0, 0], dtype=torch.long),
            node_type=torch.tensor([0, 0], dtype=torch.long),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_attr=torch.zeros((2, PAYLOAD_WIDTH), dtype=torch.float32),
            edge_relation=torch.zeros(2, dtype=torch.long),
            edge_triple=torch.zeros(2, dtype=torch.long),
            y=torch.tensor([label], dtype=torch.long), sample_id=sid, subject=subject)
    splits = {"train": [rows["train-1"]], "dev": [rows["dev-1"], rows["dev-2"]],
              "validation": [rows["validation-1"]]}
    monkeypatch.setattr(train, "load_targets", lambda path: {"synthetic": True})
    monkeypatch.setattr(train, "select_top_labels", lambda targets, top_k: (targets, [0, 1], {}))
    monkeypatch.setattr(train, "build_dataset", lambda *args, **kwargs: (splits, prep))

    torch.manual_seed(27)
    common_args = SimpleNamespace(num_relations=3)
    model = build_method("cei_gnn", num_tokens=2, node_dim=2, edge_dim=PAYLOAD_WIDTH,
        num_classes=2, hidden=4, layers=1, dropout=0.0, token_dim=3, num_triples=2,
        args=common_args)
    config = model.run_config()
    model.eval()
    proba, labels = train.evaluate(model, DataLoader(splits["dev"], batch_size=2),
                                   torch.device("cpu"), epoch=0)
    torch.save(model.state_dict(), output / "best.pt")
    prediction_hash = hashlib.sha256(np.ascontiguousarray(proba).tobytes()).hexdigest()
    ids = np.asarray([row.sample_id for row in splits["dev"]])
    np.savez_compressed(output / "dev.npz", proba=proba, y=labels, sample_ids=ids)
    binding = {
        "method": "cei_gnn", "method_config": config, "method_native_defaults": config["native_defaults"],
        "artifact": str(artifact), "artifact_graphs_sha256": sha256(artifact / "graphs.jsonl"),
        "artifact_visit_membership_file": "visit_membership.jsonl",
        "artifact_visit_membership_sha256": sha256(artifact / "visit_membership.jsonl"),
        "targets_path": str(targets_path), "targets_sha256": sha256(targets_path),
        "source_code": recursive_source_hashes(Path(train.__file__).parent),
        "preprocessing_sha256": sha256(output / "preprocessing.json"),
        "top_k_labels": 2, "kept_label_indices": [0, 1], "edges": "all",
        "train_limit": 1, "token_min_count": 1, "seed": 1234,
        "dropped_relations": [], "rewired_relations": [], "min_prior_visits": 0,
        "edge_direction": "forward", "dev_limit": 2, "sample_seed": 1234,
        "split_sample_ids_sha256": {fold: sample_ids_sha256(row.sample_id for row in data)
                                     for fold, data in splits.items()},
        "selection_fold": "dev", "final_eval": "none", "test_evaluated": False,
        "num_classes": 2, "num_relations": 3, "vocabulary_size": 2,
        "num_meta_relations": 2, "node_dim": 2, "edge_dim": PAYLOAD_WIDTH,
        "hidden": 4, "layers": 1, "dropout": 0.0, "batch_size": 2,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "selected_dev": {"epoch_index": 0, "prediction_sha256": prediction_hash},
    }
    result = {"status": "completed", "binding": binding, "metrics": None,
              "dev_metrics": train.metrics(labels, proba, 2), "validation_evaluations": 0}
    replay = module.replay_stage(output, binding, result)
    assert replay["exact_probabilities"] and replay["exact_labels"]
    assert replay["exact_sample_identity"] and replay["test_evaluated"] is False
    assert replay["inputs_verified"] and replay["source_verified"]
    assert replay["validation_evaluated"] is False
    assert json.loads((output / "replay.json").read_text())["checkpoint_sha256"] == sha256(output / "best.pt")

    tampered = tmp_path / "tampered"
    tampered.mkdir()
    for name in ("preprocessing.json", "best.pt", "dev.npz"):
        (tampered / name).write_bytes((output / name).read_bytes())
    state = torch.load(tampered / "best.pt", map_location="cpu", weights_only=True)
    first_key = next(iter(state))
    state[first_key] = state[first_key] + 0.1
    torch.save(state, tampered / "best.pt")
    (tampered / "replay.json").write_text((output / "replay.json").read_text())
    with pytest.raises(ValueError, match="probabilities|proof"):
        module.replay_stage(tampered, binding, result)

    for array_name in ("proba", "sample_ids", "y"):
        corrupted = tmp_path / ("bad-" + array_name)
        corrupted.mkdir()
        for filename in ("preprocessing.json", "best.pt", "dev.npz", "replay.json"):
            (corrupted / filename).write_bytes((output / filename).read_bytes())
        with np.load(corrupted / "dev.npz", allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        if array_name == "proba":
            arrays[array_name][0, 0] += 0.01
        elif array_name == "y":
            arrays[array_name][0] = 1 - arrays[array_name][0]
        else:
            arrays[array_name][0] = "tampered-id"
        np.savez_compressed(corrupted / "dev.npz", **arrays)
        with pytest.raises(ValueError, match="probabilities|labels|IDs|sample"):
            module.replay_stage(corrupted, binding, result)


def test_checkpoint_replay_failure_prevents_launching_next_stage(tmp_path, monkeypatch):
    module = _module()
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    targets, canonical = tmp_path / "targets.csv", tmp_path / "canonical.json"
    targets.write_text("x")
    canonical.write_text("{}")
    stages = module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                               output_root=tmp_path / "runs")
    monkeypatch.setattr(module, "capture_bindings", lambda stage: {
        "source_state_sha256": "source", "input_state_sha256": "inputs",
        "common_config_sha256": "configuration"})
    monkeypatch.setattr(module, "_validate_artifacts", lambda *args: None)
    replay_calls = []
    def reject_replay(*args):
        replay_calls.append(args[0])
        raise ValueError("synthetic checkpoint replay mismatch")
    monkeypatch.setattr(module, "replay_stage", reject_replay)
    launches = []
    def synthetic_training(argv, **kwargs):
        launches.append(argv)
        output = Path(argv[argv.index("--output") + 1])
        output.mkdir(parents=True)
        binding = _bindings(use_interactions=True, train_limit=256, dev_limit=128, epochs=2)
        binding["method_config"] = {"effective_settings": {"use_interactions": True}}
        (output / "binding.json").write_text(json.dumps(binding))
        (output / "result.json").write_text(json.dumps({
            "status": "completed", "binding": binding, "metrics": None,
            "dev_metrics": {"macro_f1": 0.1}, "test_evaluated": False}))
        return SimpleNamespace(stdout="", stderr="")
    monkeypatch.setattr(module.subprocess, "run", synthetic_training)
    with pytest.raises(SystemExit):
        module.execute_plan(stages, journal_path=tmp_path / "journal.json")
    assert len(launches) == 1
    assert len(replay_calls) == 1
