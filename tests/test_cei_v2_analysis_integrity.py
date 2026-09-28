"""Synthetic integration regressions for CEI-GNN v2 analysis integrity."""
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _study_tests():
    spec = importlib.util.spec_from_file_location(
        "cei_v2_study_existing_tests", ROOT / "tests" / "test_cei_v2_study.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _real_replay_fixture(tmp_path, monkeypatch):
    """Run the established synthetic real-model replay test and capture its bound arm."""
    existing = _study_tests()
    study = existing._module()
    existing._module = lambda: study
    captured = {}
    original = study.replay_v2_stage

    def capture(output, binding, result, **kwargs):
        captured.setdefault("output", Path(output))
        captured.setdefault("binding", binding)
        captured.setdefault("result", result)
        return original(output, binding, result, **kwargs)

    monkeypatch.setattr(study, "replay_v2_stage", capture)
    existing.test_replay_reconstructs_v2_model_and_rejects_tampering(tmp_path, monkeypatch)
    monkeypatch.setattr(study, "replay_v2_stage", original)
    assert captured["output"].joinpath("replay.json").is_file()
    return study, captured


def _make_single_arm_validator(study, captured, monkeypatch):
    arm = "product_seed2025"
    captured["output"].joinpath("binding.json").write_text(json.dumps(captured["binding"]))
    captured["output"].joinpath("result.json").write_text(json.dumps(captured["result"]))
    stage = study.Stage(arm, [], str(captured["output"]), 2025,
                        (10000, 5000, captured["binding"].get("epochs", 40)), "product")
    monkeypatch.setattr(study, "stage_specs", lambda: (("smoke", (256, 128, 2), 1234, "product"),
                                                        (arm, stage.budget, 2025, "product")))
    monkeypatch.setattr(study, "validate_v2_binding", lambda *args, **kwargs: None)
    monkeypatch.setattr(study, "assert_arm_parity", lambda *args, **kwargs: True)
    monkeypatch.setattr(study, "_check_stage_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(study.pilot, "_validate_artifacts", lambda *args, **kwargs: None)
    return arm


def test_analysis_rejects_missing_replay_proof_without_recreating_it(tmp_path, monkeypatch):
    study, captured = _real_replay_fixture(tmp_path, monkeypatch)
    arm = _make_single_arm_validator(study, captured, monkeypatch)
    captured["output"].joinpath("replay.json").unlink()

    with pytest.raises(ValueError, match="replay proof"):
        study.validate_completed_study(tmp_path, include_pair_counts=True)

    assert not captured["output"].joinpath("replay.json").exists()
    assert arm == "product_seed2025"


def test_analysis_refuses_output_inside_arm_before_validation_or_write(tmp_path, monkeypatch):
    study = _study_tests()._module()
    arm_dir = tmp_path / "product_seed1234"
    arm_dir.mkdir()
    monkeypatch.setattr(study, "validate_completed_study",
                        lambda *args, **kwargs: pytest.fail("validator must not run"))
    destination = arm_dir / "analysis.json"

    with pytest.raises(ValueError, match="inside.*arm"):
        study.analyze(tmp_path, destination)

    assert not destination.exists()
