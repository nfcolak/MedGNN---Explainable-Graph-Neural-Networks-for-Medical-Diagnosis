"""Configurable cohort regression tests: synthetic inputs, never GraphXAI runs."""
import json
from pathlib import Path

import pytest

from comparison.standardized import build_explanation_cohort as builder
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
