"""Configurable cohort regression tests: synthetic inputs, never GraphXAI runs."""
import json
from pathlib import Path

import pytest

from comparison.standardized import build_explanation_cohort as builder
from comparison.standardized import run_explanations as runner
from shared.lib import explanation_contract as contract
from shared.lib.benchmark_contract import EXPECTED_CLASSES, EXPECTED_FOLD_COUNTS


@pytest.fixture
def inputs(tmp_path):
    folds = {}
    rows = ["subject_id,disease_1"]
    for fold, count in EXPECTED_FOLD_COUNTS.items():
        for index in range(count):
            subject = f"fixture-{fold}-{index}"
            folds[subject] = fold
            rows.append(f"{subject},{EXPECTED_CLASSES[index % len(EXPECTED_CLASSES)]}")
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"fold": folds, "classes": list(EXPECTED_CLASSES)}))
    dataset = tmp_path / "data.csv"
    dataset.write_text("\n".join(rows) + "\n")
    return split, dataset


def test_orchestrator_propagates_count_and_cohort():
    from comparison.standardized.run_all import command_plan
    plan = command_plan(cohort_size=73, cohort_path=Path("custom.json"))
    assert all(c[c.index("--cohort-size") + 1] == "73" for c in plan["explain"])
    assert all(c[c.index("--cohort") + 1] == "custom.json" for c in plan["explain"])


def test_default_is_500_and_selection_stays_deterministic(inputs):
    split, dataset = inputs
    first = builder.build_artifact(split, dataset)
    assert first["count"] == 500
    assert first == builder.build_artifact(split, dataset)
    assert builder._parser().parse_args([]).n == 500
    assert contract.COHORT_SIZE == 500


@pytest.mark.parametrize("count", [50, 73, 500])
def test_explicit_count_roundtrip_and_no_silent_legacy_upgrade(inputs, tmp_path, count):
    split, dataset = inputs
    payload = builder.build_artifact(split, dataset, n=count)
    cohort = tmp_path / "cohort.json"
    cohort.write_text(json.dumps(payload))
    loaded = contract.load_explanation_cohort(
        cohort, split_path=split, dataset_path=dataset, expected_count=count
    )
    assert len(loaded.subject_ids) == count
    if count != 500:
        with pytest.raises(ValueError, match="500 subjects"):
            contract.load_explanation_cohort(cohort, split_path=split, dataset_path=dataset)


@pytest.mark.parametrize("count", [0, -1, True, 1.5, "50", None])
def test_invalid_expected_count_fails_before_io(count):
    with pytest.raises(ValueError, match="positive exact integer"):
        contract.load_explanation_cohort(
            "absent", split_path="absent", dataset_path="absent", expected_count=count
        )


@pytest.mark.parametrize("count", [50, 73, 500])
def test_runner_propagates_one_count_and_cohort_to_all_methods(tmp_path, count):
    commands = runner.build_commands(
        topology="star", seed=1234, cohort_path=tmp_path / "cohort.json",
        cohort_size=count, require_checkpoints=False,
    )
    assert len(commands) == 3
    for command in commands:
        assert command[command.index("--cohort-size") + 1] == str(count)
        assert command[command.index("--cohort") + 1] == str(tmp_path / "cohort.json")
    default = runner.build_commands(topology="star", seed=1234, require_checkpoints=False)
    assert all(c[c.index("--cohort-size") + 1] == "500" for c in default)

@pytest.mark.parametrize("count", [50, 500])
def test_actual_manifest_counts_and_shared_ids(inputs, tmp_path, count):
    from test_standardized_explanations import _base_record
    split, dataset = inputs
    payload = builder.build_artifact(split, dataset, n=count)
    cohort_path = tmp_path / "cohort.json"
    cohort_path.write_text(json.dumps(payload))
    output = tmp_path / "outputs"
    for method in contract.STANDARDIZED_METHODS:
        manifest = contract.write_standardized_explanations(
            output_dir=output / method,
            records=[_base_record(method, subject) for subject in payload["subject_ids"]],
            method=method, topology="star", seed=1234, cohort_path=cohort_path,
            split_path=split, dataset_path=dataset, checkpoint_path=None,
            allow_missing_checkpoint=True, expected_count=count,
        )
        assert manifest["subject_count"] == len(manifest["record_files"]) == count
    summary = runner.validate_cross_method_outputs(
        output, cohort_path=cohort_path, split_path=split, dataset_path=dataset,
        topology="star", seed=1234, cohort_size=count,
    )
    assert summary["subject_count"] == count
    assert summary["subject_ids"] == payload["subject_ids"]
    manifest_path = output / "gsat" / "explanation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["subject_count"] = count + 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest contract mismatch"):
        runner.validate_cross_method_outputs(
            output, cohort_path=cohort_path, split_path=split, dataset_path=dataset,
            topology="star", seed=1234, cohort_size=count,
        )


@pytest.mark.parametrize("module_name", list(runner._METHOD_MODULES.values()))
@pytest.mark.parametrize("count", [None, 50, 73])
def test_method_cli_accepts_shared_count_without_running_model(monkeypatch, module_name, count):
    import importlib
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "main", lambda **kwargs: kwargs)
    argv = ["--checkpoint", "unused", "--graph-structure", "star",
            "--canonical-split", "unused", "--cohort", "unused", "--dataset", "unused",
            "--seed", "1234", "--out-dir", "unused"]
    if count is not None:
        argv += ["--cohort-size", str(count)]
    assert module.cli(argv)["cohort_size"] == (500 if count is None else count)
