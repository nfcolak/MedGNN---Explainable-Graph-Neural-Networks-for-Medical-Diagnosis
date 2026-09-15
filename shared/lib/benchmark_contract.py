"""Shared validation and reproducibility helpers for benchmark runs."""
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from shared.lib.graph_structures import (
    GRAPH_STRUCTURES,
    SUPPORTED_METHODS,
    resolve as resolve_structure,
)


ALLOWED_SEEDS: Tuple[int, ...] = (1234, 1235, 1236)
PRIMARY_STRUCTURES: Tuple[str, ...] = tuple(
    name for name, metadata in GRAPH_STRUCTURES.items() if metadata["primary"]
)
EXPECTED_FOLD_COUNTS: Dict[int, int] = {0: 59607, 1: 7448, 2: 7456}
EXPECTED_CLASSES: Tuple[str, ...] = (
    "Acute kidney failure",
    "Acute pharyngitis",
    "Acute upper respiratory infection",
    "Alcohol abuse with intoxication",
    "Anemia",
    "Anxiety disorder",
    "Back or spine pain",
    "Cardiovascular risk factor",
    "Dehydration",
    "Diabetes mellitus",
    "End stage renal disease",
    "Epilepsy",
    "GI bleed",
    "Head injury",
    "Heart failure",
    "Hypokalemia",
    "Hypotension",
    "Hypothyroidism",
    "Laceration w/o fb of l idx fngr w/o damage to nail",
    "Limb injury or pain",
    "Lower respiratory disease",
    "Major depressive disorder",
    "Multiple fractures of ribs",
    "Non-ST elevation (NSTEMI) myocardial infarction",
    "Sepsis",
    "Skin or soft-tissue infection",
    "UTI or pyelonephritis",
    "Unsp intestnl obst",
    "Unspecified asthma with (acute) exacerbation",
    "Unspecified atrial fibrillation",
)
EXPECTED_CLASS_COUNT = len(EXPECTED_CLASSES)


@dataclass(frozen=True)
class BenchmarkSpec:
    """One validated method/topology/seed benchmark run."""

    method: str
    structure: str
    seed: int

    def __post_init__(self) -> None:
        if self.method not in SUPPORTED_METHODS:
            choices = ", ".join(SUPPORTED_METHODS)
            raise ValueError(
                f"Unknown benchmark method {self.method!r}; choose one of: {choices}."
            )

        resolved = resolve_structure(self.structure, self.method)
        if resolved != self.structure:
            raise ValueError(
                f"Benchmark structure must use canonical name {resolved!r}, "
                f"got {self.structure!r}."
            )

        if type(self.seed) is not int or self.seed not in ALLOWED_SEEDS:
            choices = ", ".join(str(seed) for seed in ALLOWED_SEEDS)
            raise ValueError(
                f"Benchmark seed must be one of: {choices}; got {self.seed!r}."
            )


def load_canonical_split(path: Any) -> Dict[str, Any]:
    """Load and validate the frozen 30-class benchmark split."""
    split_path = Path(path)
    with split_path.open("r", encoding="utf-8") as handle:
        split = json.load(handle)

    if not isinstance(split, dict):
        raise ValueError("Canonical split must be a JSON object.")

    missing = {"fold", "classes"} - set(split)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Canonical split is missing required fields: {names}.")

    folds = split["fold"]
    if not isinstance(folds, dict):
        raise ValueError("Canonical split 'fold' must be a subject-to-fold object.")
    if any(
        type(fold_id) is not int or fold_id not in EXPECTED_FOLD_COUNTS
        for fold_id in folds.values()
    ):
        raise ValueError(
            "Canonical split fold values must be integer identifiers 0, 1, or 2."
        )

    counts = dict(Counter(folds.values()))
    if counts != EXPECTED_FOLD_COUNTS:
        raise ValueError(
            "Canonical split fold counts do not match the benchmark contract: "
            f"expected {EXPECTED_FOLD_COUNTS}, got {counts}."
        )

    classes = split["classes"]
    if not isinstance(classes, list):
        raise ValueError("Canonical split 'classes' must be an ordered list.")
    if any(type(class_name) is not str or not class_name for class_name in classes):
        raise ValueError("Every canonical split class must be a non-empty string.")
    if len(classes) != EXPECTED_CLASS_COUNT or len(set(classes)) != len(classes):
        raise ValueError(
            "Canonical split must contain exactly 30 unique classes in fixed order."
        )
    if tuple(classes) != EXPECTED_CLASSES:
        raise ValueError(
            "Canonical split class identity and order do not match the benchmark contract."
        )

    if "n_kept" in split and split["n_kept"] != len(folds):
        raise ValueError(
            "Canonical split 'n_kept' does not match the number of subjects."
        )

    return split


def file_sha256(path: Any) -> str:
    """Return the lowercase SHA-256 digest of a file's exact bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_primary_matrix(specs: Iterable[BenchmarkSpec]) -> None:
    """Require exactly one run for every method/primary-topology/seed cell."""
    actual_specs = list(specs)
    if any(not isinstance(spec, BenchmarkSpec) for spec in actual_specs):
        raise ValueError("Every primary benchmark matrix entry must be a BenchmarkSpec.")

    expected = {
        BenchmarkSpec(method, structure, seed)
        for method, structure, seed in product(
            SUPPORTED_METHODS, PRIMARY_STRUCTURES, ALLOWED_SEEDS
        )
    }
    actual = set(actual_specs)
    duplicates = len(actual_specs) - len(actual)
    missing = expected - actual
    unexpected = actual - expected

    if duplicates or missing or unexpected:
        raise ValueError(
            "Invalid primary benchmark matrix: "
            f"duplicates={duplicates}, missing={len(missing)}, "
            f"unexpected={len(unexpected)}."
        )
