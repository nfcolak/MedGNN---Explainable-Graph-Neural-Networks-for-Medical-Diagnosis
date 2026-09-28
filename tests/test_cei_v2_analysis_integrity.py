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


def test_analysis_fails_if_history_disappears_after_validation(tmp_path, monkeypatch):
    import numpy as np

    study = _study_tests()._module()
    from comparison.standardized.clinical_graph_v2 import train

    names = study.FULL_STAGE_NAMES
    bindings = {}
    arms = {name: np.asarray([[0.9, 0.1], [0.1, 0.9]]) for name in names}
    y, subjects = np.asarray([0, 1]), np.asarray(["p1", "p2"])
    for name in names:
        arm_dir = tmp_path / name
        arm_dir.mkdir()
        bindings[name] = {"num_classes": 2, "selected_dev": {"metric_value": 1.0, "epoch": 1},
                          "label_order": ["A", "B"], "parameter_count": 42,
                          "active_parameter_count": 40, "epochs": 1}
        (arm_dir / "history.json").write_text(json.dumps([
            {"epoch": 1, "train_loss": 1.0, "macro_f1": 1.0, "selection_fold": "dev"}]))
        (arm_dir / "result.json").write_text(json.dumps({"total_seconds": 1.0}))

    def validate_then_delete(root, *, include_pair_counts=False):
        first = next(iter(bindings))
        (Path(root) / first / "history.json").unlink()
        return (bindings, None) if include_pair_counts else bindings

    monkeypatch.setattr(study, "validate_completed_study", validate_then_delete)
    monkeypatch.setattr(study, "load_arm_predictions", lambda root: (arms, y, subjects))
    monkeypatch.setattr(study, "paired_bootstrap", lambda *args, **kwargs: (
        {name: 1.0 for name in names},
        {key: {"point": 0.0, "interval_95": [0.0, 0.0]} for key in study.COMPARISONS}))
    monkeypatch.setattr(train, "patient_equal_metrics", lambda *args, **kwargs: {"macro_f1": 1.0})
    monkeypatch.setattr(train, "per_class_table", lambda *args, **kwargs: [{"index": 0}, {"index": 1}])

    with pytest.raises(ValueError, match="history.json missing"):
        study.analyze(tmp_path, tmp_path / "analysis.json")

    assert not (tmp_path / "analysis.json").exists()


def test_decision_is_invariant_to_pair_count_and_curve_reporting(tmp_path, monkeypatch):
    import numpy as np

    study = _study_tests()._module()
    from comparison.standardized.clinical_graph_v2 import train

    names = study.FULL_STAGE_NAMES
    arms = {name: np.asarray([[0.9, 0.1], [0.1, 0.9]]) for name in names}
    y, subjects = np.asarray([0, 1]), np.asarray(["p1", "p2"])
    bindings = {name: {"num_classes": 2,
                       "selected_dev": {"metric_value": 1.0, "epoch": 1},
                       "label_order": ["A", "B"], "parameter_count": 42,
                       "active_parameter_count": 40, "epochs": 2}
                for name in names}
    monkeypatch.setattr(study, "load_arm_predictions", lambda root: (arms, y, subjects))
    monkeypatch.setattr(study, "paired_bootstrap", lambda *args, **kwargs: (
        {name: 1.0 for name in names},
        {key: {"point": 0.0, "interval_95": [0.0, 0.0]}
         for key in study.COMPARISONS}))
    monkeypatch.setattr(train, "patient_equal_metrics", lambda *args, **kwargs: {"macro_f1": 1.0})
    monkeypatch.setattr(train, "per_class_table", lambda *args, **kwargs: [{"index": 0}, {"index": 1}])

    reports = []
    for index, (replay_counts, injected_counts, curve) in enumerate((
            ([0, 1], [180, 1045], 0.1),
            (None, None, 9.0))):
        root = tmp_path / f"case-{index}"
        root.mkdir()
        for name in names:
            arm_dir = root / name
            arm_dir.mkdir()
            (arm_dir / "history.json").write_text(json.dumps([
                {"epoch": 1, "train_loss": curve, "macro_f1": 1.0, "selection_fold": "dev"},
                {"epoch": 2, "train_loss": curve / 2, "macro_f1": 1.0, "selection_fold": "dev"}]))
            (arm_dir / "result.json").write_text(json.dumps({"total_seconds": 1.0}))
        monkeypatch.setattr(study, "validate_completed_study",
                            lambda root, *, include_pair_counts=False:
                            (bindings, replay_counts) if include_pair_counts else bindings)
        report = study.analyze(root, root / "analysis.json", pair_counts=injected_counts)
        assert study.decide(report["dev_macro_f1"], report["comparisons"]) == report["decision"]
        reports.append(report)

    assert reports[0]["decision"] == reports[1]["decision"]


def test_replay_counts_fill_analysis_bands_without_expanding_proof(tmp_path, monkeypatch):
    import numpy as np

    study = _study_tests()._module()
    from comparison.standardized.clinical_graph_v2 import train

    names = study.FULL_STAGE_NAMES
    counts = [0, 1, 180, 1045]
    arms = {name: np.asarray([[0.9, 0.1], [0.1, 0.9],
                              [0.9, 0.1], [0.1, 0.9]]) for name in names}
    y, subjects = np.asarray([0, 1, 0, 1]), np.asarray(["p1", "p2", "p3", "p4"])
    bindings = {name: {"num_classes": 2,
                       "selected_dev": {"metric_value": 1.0, "epoch": 1},
                       "label_order": ["A", "B"], "parameter_count": 42,
                       "active_parameter_count": 40, "epochs": 1}
                for name in names}
    root = tmp_path / "study"
    root.mkdir()
    for name in names:
        arm_dir = root / name
        arm_dir.mkdir()
        (arm_dir / "history.json").write_text(json.dumps([
            {"epoch": 1, "train_loss": 1.0, "macro_f1": 1.0, "selection_fold": "dev"}]))
        (arm_dir / "result.json").write_text(json.dumps({"total_seconds": 1.0}))
    monkeypatch.setattr(study, "validate_completed_study",
                        lambda root, *, include_pair_counts=False:
                        (bindings, counts) if include_pair_counts else bindings)
    monkeypatch.setattr(study, "load_arm_predictions", lambda root: (arms, y, subjects))
    monkeypatch.setattr(study, "paired_bootstrap", lambda *args, **kwargs: (
        {name: 1.0 for name in names},
        {key: {"point": 0.0, "interval_95": [0.0, 0.0]}
         for key in study.COMPARISONS}))
    monkeypatch.setattr(train, "patient_equal_metrics", lambda *args, **kwargs: {"macro_f1": 1.0})
    monkeypatch.setattr(train, "per_class_table", lambda *args, **kwargs: [{"index": 0}, {"index": 1}])

    report = study.analyze(root, root / "analysis.json")
    bands = report["pair_count_bands"]
    assert [bands[key]["graphs"] for key in ("0", "1-179", "180-1044", ">1044")] == [1, 1, 1, 1]
    assert report["pair_count_source"] == "dev rows reloaded by replay (structural count, no scoring)"


def test_real_replay_keeps_persisted_proof_keys_pre_fix2e(tmp_path, monkeypatch):
    study, captured = _real_replay_fixture(tmp_path, monkeypatch)
    returned = study.replay_v2_stage(captured["output"], captured["binding"],
                                     captured["result"], persist=False)
    persisted = json.loads(captured["output"].joinpath("replay.json").read_text())
    expected_keys = {"status", "method", "pair_mode", "seed", "checkpoint_sha256",
                     "proba_sha256", "split_sample_ids_sha256", "dev_count",
                     "exact_probabilities", "exact_labels", "exact_sample_identity",
                     "patient_disjoint", "validation_evaluated", "test_evaluated", "dev_metrics"}
    assert set(persisted) == expected_keys
    assert "_dev_pair_counts" in returned
    assert "_dev_pair_counts" not in persisted
