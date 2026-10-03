"""Regression tests for the three-mode smoke execution boundary."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "comparison/standardized/clinical_graph_v2/studies/cei/cei_v2_study.py"


def _module():
    spec = importlib.util.spec_from_file_location("comparison.standardized.clinical_graph_v2.studies.cei.cei_v2_study_smoke3_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_completed_study_validation_selects_only_nine_full_stages(tmp_path, monkeypatch):
    module = _module()
    full_names = [
        "product_seed1234", "additive_seed1234", "off_seed1234",
        "product_seed2025", "additive_seed2025", "off_seed2025",
        "product_seed7", "additive_seed7", "off_seed7",
    ]
    for name in full_names:
        stage_dir = tmp_path / name
        stage_dir.mkdir()
        (stage_dir / "binding.json").write_text("{}")
        (stage_dir / "result.json").write_text(json.dumps({"status": "completed",
                                                              "binding": {}, "dev_metrics": {},
                                                              "metrics": None,
                                                              "test_evaluated": False}))
    visited = []
    monkeypatch.setattr(module, "validate_v2_binding", lambda *a, **k: None)
    monkeypatch.setattr(module, "_check_stage_result", lambda *a: None)
    monkeypatch.setattr(module.pilot, "_validate_artifacts", lambda *a: None)
    monkeypatch.setattr(module, "replay_v2_stage", lambda directory, *a, **k:
                        visited.append(Path(directory).name) or {"status": "verified"})
    monkeypatch.setattr(module, "assert_arm_parity", lambda bindings: None)

    bindings = module.validate_completed_study(tmp_path)

    assert list(bindings) == full_names
    assert visited == full_names


def test_smoke_executor_launches_all_modes_once_in_order(tmp_path, monkeypatch, capsys):
    module = _module()
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    targets, canonical = tmp_path / "targets.csv", tmp_path / "canonical.json"
    targets.write_text("synthetic")
    canonical.write_text("{}")
    root = tmp_path / "runs"
    stages = module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                               output_root=root)
    monkeypatch.setattr(module, "_capture", lambda stage: {
        "source_state_sha256": "s", "input_state_sha256": "i", "executable_sources": {}})
    monkeypatch.setattr(module.pilot, "_assert_runner_source_binding", lambda *a: None)
    monkeypatch.setattr(module.pilot, "_validate_artifacts", lambda *a: None)
    monkeypatch.setattr(module, "validate_v2_binding", lambda *a, **k: None)
    monkeypatch.setattr(module, "_check_stage_result", lambda *a: None)
    monkeypatch.setattr(module, "replay_v2_stage", lambda *a, **k: {"status": "verified"})
    launches = []

    def run(argv, **kwargs):
        launches.append(list(argv))
        output = Path(argv[argv.index("--output") + 1])
        output.mkdir()
        binding = {"total_seconds": 4, "batch_size": 128}
        result = {"total_seconds": 4, "binding": binding, "status": "completed",
                  "dev_metrics": {}, "metrics": None, "test_evaluated": False}
        (output / "binding.json").write_text(json.dumps(binding))
        (output / "result.json").write_text(json.dumps(result))
        return module.subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", run)
    code = module.execute_plan(stages, journal_path=root / "journal_smoke.json", phase="smoke")

    assert code == 0
    assert [argv[argv.index("--method-option") + 1] for argv in launches] == [
        "pair_mode=product", "pair_mode=additive", "pair_mode=off"]
    journal = json.loads((root / "journal_smoke.json").read_text())
    assert sorted(journal["stages"]) == ["v2_smoke_additive", "v2_smoke_off", "v2_smoke_product"]
    summary = json.loads(capsys.readouterr().out)
    assert [item["stage"] for item in summary["smoke_timing_summary"]] == [
        "v2_smoke_product", "v2_smoke_additive", "v2_smoke_off"]
    assert all(item["seconds_per_epoch"] == 2 for item in summary["smoke_timing_summary"])
    assert sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()) == [
        "journal_smoke.json", "v2_smoke_additive/binding.json", "v2_smoke_additive/result.json",
        "v2_smoke_off/binding.json", "v2_smoke_off/result.json",
        "v2_smoke_product/binding.json", "v2_smoke_product/result.json"]


def test_smoke_eta_uses_mean_of_heterogeneous_mode_timings(tmp_path, monkeypatch):
    module = _module()
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    targets, canonical = tmp_path / "targets.csv", tmp_path / "canonical.json"
    targets.write_text("synthetic")
    canonical.write_text("{}")
    root = tmp_path / "runs"
    stages = module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                               output_root=root)
    monkeypatch.setattr(module, "_capture", lambda stage: {
        "source_state_sha256": "s", "input_state_sha256": "i", "executable_sources": {}})
    monkeypatch.setattr(module.pilot, "_assert_runner_source_binding", lambda *a: True)
    monkeypatch.setattr(module.pilot, "_validate_artifacts", lambda *a: None)
    monkeypatch.setattr(module, "validate_v2_binding", lambda *a, **k: None)
    monkeypatch.setattr(module, "replay_v2_stage", lambda *a: {"status": "verified"})
    seconds_per_epoch = iter((2.0, 8.0, 14.0))

    def fake_run(argv, **kwargs):
        output = Path(argv[argv.index("--output") + 1])
        output.mkdir()
        binding = {}
        (output / "binding.json").write_text(json.dumps(binding))
        (output / "result.json").write_text(json.dumps({
            "status": "completed", "binding": binding, "metrics": None,
            "dev_metrics": {}, "test_evaluated": False,
            "total_seconds": next(seconds_per_epoch) * 2}))
        return module.subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.execute_plan(stages, journal_path=root / "journal_smoke.json", phase="smoke") == 0
    journal = json.loads((root / "journal_smoke.json").read_text())
    expected_mean_eta = round(9 * (((2.0 + 8.0 + 14.0) / 3) * 40) / 60, 1)
    product_only_eta = round(9 * 2.0 * 40 / 60, 1)
    assert journal["eta_minutes_nine_runs_smoke"] == expected_mean_eta
    assert journal["eta_minutes_nine_runs_smoke"] != product_only_eta
