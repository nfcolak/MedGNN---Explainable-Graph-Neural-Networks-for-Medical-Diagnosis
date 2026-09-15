"""Build the fixed canonical test-subject cohort used for explanations."""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import csv
import json
from pathlib import Path
import random
import sys
from typing import Any


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.lib.benchmark_contract import file_sha256, load_canonical_split
from shared.lib.explanation_contract import COHORT_SIZE


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_PATH = PROJECT_ROOT / "comparison" / "canonical_split.json"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "merged_ed.csv"
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("explanation_subjects.json")
SCHEMA = "medgnn.explanation_cohort"
SCHEMA_VERSION = 1
TEST_FOLD = 2


def _validate_request(n: Any, seed: Any) -> None:
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive exact integer.")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative exact integer.")


def _validated_split(split: Any) -> tuple[dict[str, int], list[str]]:
    if not isinstance(split, Mapping):
        raise ValueError("split must be a mapping with 'fold' and 'classes'.")
    folds = split.get("fold")
    classes = split.get("classes")
    if not isinstance(folds, Mapping):
        raise ValueError("split 'fold' must be a subject-to-fold mapping.")
    if (
        not isinstance(classes, Sequence)
        or isinstance(classes, (str, bytes))
        or not classes
        or any(type(class_name) is not str or not class_name for class_name in classes)
        or len(set(classes)) != len(classes)
    ):
        raise ValueError("split classes must be unique non-empty strings.")

    normalized_folds: dict[str, int] = {}
    for subject, fold_id in folds.items():
        if type(subject) is not str or not subject:
            raise ValueError("split subject IDs must be non-empty strings.")
        if type(fold_id) is not int or fold_id not in {0, 1, TEST_FOLD}:
            raise ValueError("split fold values must be exact integers 0, 1, or 2.")
        normalized_folds[subject] = fold_id
    return normalized_folds, list(classes)


def _label_items(labels: Any) -> Iterable[tuple[Any, Any]]:
    if isinstance(labels, Mapping):
        return labels.items()
    if isinstance(labels, (str, bytes)):
        raise ValueError("labels must be a mapping or iterable of subject-label pairs.")
    try:
        return iter(labels)
    except TypeError as exc:
        raise ValueError(
            "labels must be a mapping or iterable of subject-label pairs."
        ) from exc


def _validated_test_labels(
    test_subjects: set[str], labels: Any, classes: Sequence[str]
) -> dict[str, str]:
    canonical_classes = set(classes)
    label_by_subject: dict[str, str] = {}
    for item in _label_items(labels):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("labels iterable entries must be subject-label pairs.")
        subject, label = item
        if subject not in test_subjects:
            continue
        if type(subject) is not str or not subject:
            raise ValueError("test subject IDs must be non-empty strings.")
        if subject in label_by_subject:
            if label_by_subject[subject] == label:
                raise ValueError(f"duplicate label for test subject {subject!r}.")
            raise ValueError(f"ambiguous labels for test subject {subject!r}.")
        if type(label) is not str or not label or label not in canonical_classes:
            raise ValueError(
                f"test subject {subject!r} has unknown canonical class {label!r}."
            )
        label_by_subject[subject] = label

    missing = test_subjects - set(label_by_subject)
    if missing:
        sample = sorted(missing)[:3]
        raise ValueError(
            f"missing labels for {len(missing)} test subjects; sample={sample}."
        )
    return label_by_subject


def _allocate_quotas(
    classes: Sequence[str], grouped: Mapping[str, Sequence[str]], n: int
) -> dict[str, int]:
    total = sum(len(grouped[class_name]) for class_name in classes)
    quotas = {class_name: 0 for class_name in classes}

    if n >= len(classes):
        empty = [class_name for class_name in classes if not grouped[class_name]]
        if empty:
            raise ValueError(
                "insufficient eligible test subjects for class-stratified selection; "
                f"classes without test subjects={empty}."
            )
        quotas = {class_name: 1 for class_name in classes}

    ideal = {
        class_name: n * len(grouped[class_name]) / total for class_name in classes
    }
    remaining = n - sum(quotas.values())
    class_position = {class_name: index for index, class_name in enumerate(classes)}
    while remaining:
        eligible = [
            class_name
            for class_name in classes
            if quotas[class_name] < len(grouped[class_name])
        ]
        if not eligible:
            raise ValueError("insufficient eligible test subjects.")
        chosen = max(
            eligible,
            key=lambda class_name: (
                ideal[class_name] - quotas[class_name],
                len(grouped[class_name]) - quotas[class_name],
                -class_position[class_name],
            ),
        )
        quotas[chosen] += 1
        remaining -= 1
    return quotas


def select_subjects(split: Any, labels: Any, n: int = COHORT_SIZE, seed: int = 1234) -> list[str]:
    """Select deterministic, unique, approximately stratified test subject IDs."""
    _validate_request(n, seed)
    folds, classes = _validated_split(split)
    test_subjects = {subject for subject, fold_id in folds.items() if fold_id == TEST_FOLD}
    if len(test_subjects) < n:
        raise ValueError(
            "insufficient eligible test subjects: "
            f"requested {n}, found {len(test_subjects)}."
        )

    label_by_subject = _validated_test_labels(test_subjects, labels, classes)
    grouped: dict[str, list[str]] = {class_name: [] for class_name in classes}
    for subject in sorted(test_subjects):
        grouped[label_by_subject[subject]].append(subject)
    quotas = _allocate_quotas(classes, grouped, n)

    rng = random.Random(seed)
    selected: list[str] = []
    for class_name in classes:
        candidates = grouped[class_name].copy()
        rng.shuffle(candidates)
        selected.extend(candidates[: quotas[class_name]])

    selected.sort()
    if len(selected) != n or len(set(selected)) != n:
        raise RuntimeError("cohort selection failed its exact count and uniqueness invariant.")
    return selected


def _load_canonical_label_rows(
    dataset_path: Path, canonical_subjects: set[str]
) -> list[tuple[str, Any]]:
    if (
        not dataset_path.exists()
        or not dataset_path.is_file()
        or dataset_path.stat().st_size == 0
    ):
        raise ValueError(f"dataset must be a non-empty regular file: {dataset_path}")
    rows: list[tuple[str, Any]] = []
    try:
        with dataset_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            if "subject_id" not in fields or "disease_1" not in fields:
                raise ValueError("dataset must contain subject_id and disease_1 columns.")
            for row in reader:
                subject = row.get("subject_id")
                if type(subject) is str and subject in canonical_subjects:
                    rows.append((subject, row.get("disease_1")))
    except UnicodeDecodeError as exc:
        raise ValueError("dataset must be valid UTF-8 CSV.") from exc
    return rows


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def build_artifact(
    split_path: Any = DEFAULT_SPLIT_PATH,
    dataset_path: Any = DEFAULT_DATASET_PATH,
    n: int = COHORT_SIZE,
    seed: int = 1234,
) -> dict[str, Any]:
    """Build and fully validate the explanation-cohort JSON payload."""
    _validate_request(n, seed)
    split_file = Path(split_path)
    dataset_file = Path(dataset_path)
    split = load_canonical_split(split_file)
    canonical_subjects = set(split["fold"])
    label_rows = _load_canonical_label_rows(dataset_file, canonical_subjects)
    subject_ids = select_subjects(split, label_rows, n=n, seed=seed)
    label_by_subject = dict(label_rows)
    selected_counts = Counter(label_by_subject[subject] for subject in subject_ids)

    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "count": len(subject_ids),
        "canonical_split": {
            "path": _display_path(split_file),
            "sha256": file_sha256(split_file),
        },
        "dataset": {
            "path": _display_path(dataset_file),
            "sha256": file_sha256(dataset_file),
        },
        "class_counts": {
            class_name: selected_counts[class_name] for class_name in split["classes"]
        },
        "subject_ids": subject_ids,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--n", type=int, default=COHORT_SIZE, help="Test subjects (default: 500).")
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = build_artifact(args.split, args.dataset, n=args.n, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {artifact['count']} test subjects to {args.output} "
        f"across {len(artifact['class_counts'])} classes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
