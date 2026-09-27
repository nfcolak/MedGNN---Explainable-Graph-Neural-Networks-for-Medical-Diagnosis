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
    stages = [module.Stage(name=name, argv=["python", "train.py"], output=str(output / name),
                           seed=1234, budget=budget, treatment=treatment)
              for name, budget, treatment in (("cei_smoke", (256, 128, 2), "smoke"),
                  ("protgnn_control", (10000, 5000, 40), "control"),
                  ("cei_candidate", (10000, 5000, 40), "candidate"),
                  ("cei_product_off", (10000, 5000, 40), "product_off"))]
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
    journal = root / "journal.json"
    stage_data = (("cei_smoke", (256, 128, 2), "smoke"),
        ("protgnn_control", (10000, 5000, 40), "control"),
        ("cei_candidate", (10000, 5000, 40), "candidate"),
        ("cei_product_off", (10000, 5000, 40), "product_off"))
    stages = [module.Stage(name=name, argv=["python", "train.py", "--output", str(root / name)],
                     output=str(root / name), seed=1234, budget=budget, treatment=treatment)
              for name, budget, treatment in stage_data]
    identities = iter(("source-a", "source-a", "source-b"))
    monkeypatch.setattr(module, "capture_bindings", lambda stage: {
        "git_revision": "rev", "source_state_sha256": next(identities),
        "argv_sha256": "argv", "inputs": {},
    })
    launches = []
    def fake_run(argv, **kwargs):
        launches.append(argv)
        output = Path(argv[argv.index("--output") + 1])
        output.mkdir()
        binding = _bindings()
        (output / "binding.json").write_text(json.dumps(binding))
        (output / "result.json").write_text(json.dumps({
            "status": "completed", "binding": binding, "metrics": None,
            "dev_metrics": {"macro_f1": 0.1},
        }))
        return SimpleNamespace(stdout="", stderr="")
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as error:
        module.execute_plan(stages, journal_path=journal)

    assert error.value.code != 0
    assert len(launches) == 1
    assert launches[0][-1] == "--execute"
    saved = json.loads(journal.read_text())
    assert saved["stages"]["cei_smoke"]["status"] == "failed"
    assert "protgnn_control" not in saved["stages"]
    assert saved["status"] == "failed"


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
