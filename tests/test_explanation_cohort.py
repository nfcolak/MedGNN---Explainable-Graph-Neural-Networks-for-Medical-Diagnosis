"""Tests for the fixed, prediction-independent explanation cohort."""
from collections import Counter
import inspect
import json
from pathlib import Path

import pytest

from comparison.standardized.build_explanation_cohort import (
    build_artifact,
    select_subjects,
)
from shared.lib.benchmark_contract import file_sha256, load_canonical_split


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_SPLIT = REPO_ROOT / "comparison" / "canonical_split.json"
DATASET = REPO_ROOT / "data" / "merged_ed.csv"
ARTIFACT = REPO_ROOT / "comparison" / "standardized" / "explanation_subjects.json"


def _toy_inputs(per_class=12):
    classes = ["alpha", "beta", "gamma"]
    fold = {"train": 0, "validation": 1}
    labels = {"train": "alpha", "validation": "beta"}
    for class_index, class_name in enumerate(classes):
        for item_index in range(per_class):
            subject = f"test-{class_index}-{item_index:02d}"
            fold[subject] = 2
            labels[subject] = class_name
    return {"fold": fold, "classes": classes}, labels


def test_select_subjects_is_deterministic_unique_test_only_and_stratified():
    split, labels = _toy_inputs()

    first = select_subjects(split, labels, n=9, seed=1234)
    second = select_subjects(split, labels, n=9, seed=1234)

    assert first == second
    assert first == sorted(first)
    assert len(first) == len(set(first)) == 9
    assert all(split["fold"][subject] == 2 for subject in first)
    assert Counter(labels[subject] for subject in first) == {
        "alpha": 3,
        "beta": 3,
        "gamma": 3,
    }
    assert select_subjects(split, labels, n=9, seed=1235) != first


def test_selection_ignores_non_test_labels_and_exposes_no_prediction_input():
    split, labels = _toy_inputs()
    expected = select_subjects(split, labels, n=9, seed=1234)
    changed_non_test: dict[str, object] = dict(labels)
    changed_non_test["train"] = object()
    changed_non_test["validation"] = "not-a-canonical-class"
    changed_non_test["model-output-only"] = {"prediction": 29, "score": 1.0}

    assert select_subjects(split, changed_non_test, n=9, seed=1234) == expected
    assert tuple(inspect.signature(select_subjects).parameters) == (
        "split",
        "labels",
        "n",
        "seed",
    )


@pytest.mark.parametrize("n", [0, -1, True, 1.5, "9", None])
def test_select_subjects_rejects_invalid_n(n):
    split, labels = _toy_inputs()
    with pytest.raises(ValueError, match="n must be a positive exact integer"):
        select_subjects(split, labels, n=n, seed=1234)


@pytest.mark.parametrize("seed", [-1, True, 1.5, "1234", None])
def test_select_subjects_rejects_invalid_seed(seed):
    split, labels = _toy_inputs()
    with pytest.raises(ValueError, match="seed must be a non-negative exact integer"):
        select_subjects(split, labels, n=9, seed=seed)


def test_select_subjects_rejects_insufficient_eligible_test_subjects():
    split, labels = _toy_inputs(per_class=2)
    with pytest.raises(ValueError, match="insufficient eligible test subjects"):
        select_subjects(split, labels, n=7, seed=1234)


def test_select_subjects_rejects_missing_test_label():
    split, labels = _toy_inputs()
    del labels["test-0-00"]
    with pytest.raises(ValueError, match="missing labels for 1 test subjects"):
        select_subjects(split, labels, n=9, seed=1234)


def test_select_subjects_rejects_unknown_test_class():
    split, labels = _toy_inputs()
    labels["test-0-00"] = "unknown"
    with pytest.raises(ValueError, match="unknown canonical class"):
        select_subjects(split, labels, n=9, seed=1234)


@pytest.mark.parametrize(
    ("duplicate_label", "message"),
    [("alpha", "duplicate"), ("beta", "ambiguous")],
)
def test_select_subjects_rejects_duplicate_or_ambiguous_test_labels(
    duplicate_label, message
):
    split, labels = _toy_inputs()
    label_rows = list(labels.items())
    label_rows.append(("test-0-00", duplicate_label))

    with pytest.raises(ValueError, match=message):
        select_subjects(split, label_rows, n=9, seed=1234)


def test_select_subjects_rejects_invalid_split_class_contract():
    split, labels = _toy_inputs()
    split["classes"] = ["alpha", "alpha", "gamma"]
    with pytest.raises(ValueError, match="unique non-empty strings"):
        select_subjects(split, labels, n=9, seed=1234)


def test_checked_in_artifact_matches_authoritative_split_and_dataset():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    split = load_canonical_split(CANONICAL_SPLIT)
    regenerated = build_artifact(CANONICAL_SPLIT, DATASET, n=50, seed=1234)

    assert artifact == regenerated
    assert set(artifact) == {
        "schema",
        "schema_version",
        "seed",
        "count",
        "canonical_split",
        "dataset",
        "class_counts",
        "subject_ids",
    }
    assert artifact["schema"] == "medgnn.explanation_cohort"
    assert artifact["schema_version"] == 1
    assert artifact["seed"] == 1234
    assert artifact["count"] == 50
    assert artifact["canonical_split"] == {
        "path": "comparison/canonical_split.json",
        "sha256": file_sha256(CANONICAL_SPLIT),
    }
    assert artifact["dataset"] == {
        "path": "data/merged_ed.csv",
        "sha256": file_sha256(DATASET),
    }

    subject_ids = artifact["subject_ids"]
    assert subject_ids == sorted(subject_ids)
    assert len(subject_ids) == len(set(subject_ids)) == 50
    assert all(split["fold"].get(subject) == 2 for subject in subject_ids)
    assert list(artifact["class_counts"]) == split["classes"]
    assert set(artifact["class_counts"].values()) <= set(range(1, 51))
    assert sum(artifact["class_counts"].values()) == 50
