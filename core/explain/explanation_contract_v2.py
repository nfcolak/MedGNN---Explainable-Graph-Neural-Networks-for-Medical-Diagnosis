"""Cohort contract v2 for explaining finished clinical_graph_v2/v3 runs.

v1 (`explanation_contract.py`) is bound to the canonical star-graph split, TEST_FOLD = 2
and exactly 500 subjects, which conflicts with ADR-002 (the test fold is never
evaluated). v2 explains only the validation fold or the ADR-008 dev split of one
finished run, hash-pinned to that run's `binding.json`. v1 is not edited:
historical outputs stay valid under v1, and a v2 cohort is never a v1 cohort.
"""
from __future__ import annotations

from typing import Any, Sequence

from core.contracts import sample_ids_sha256

COHORT_SCHEMA = "medgnn.explanation_cohort"
COHORT_SCHEMA_VERSION = 2
ALLOWED_FOLDS = ("validation", "dev")


def build_cohort_v2(binding: dict[str, Any], *, fold: str,
                    fold_sample_ids: Sequence[str],
                    explained_sample_ids: Sequence[str]) -> dict[str, Any]:
    """Cohort = the explained prefix of one fold, bound to the run that scored it.

    `fold_sample_ids` is the whole fold in training order; its hash must equal the
    one the run recorded, so the cohort cannot silently describe a different
    population. `explained_sample_ids` may be a prefix of it (a capped run) and is
    recorded as such, never presented as the full fold."""
    if fold not in ALLOWED_FOLDS:
        raise ValueError(f"explanations may only use folds {ALLOWED_FOLDS}, not {fold!r} "
                         "(the held-out test fold is never evaluated, ADR-002)")
    recorded = binding.get("split_sample_ids_sha256", {}).get(fold)
    actual = sample_ids_sha256(fold_sample_ids)
    if recorded is None or recorded != actual:
        raise ValueError(f"{fold} sample ids do not match binding.json's "
                         f"split_sample_ids_sha256[{fold!r}]")
    explained = list(explained_sample_ids)
    if len(set(explained)) != len(explained) or not set(explained) <= set(fold_sample_ids):
        raise ValueError("explained sample ids must be unique members of the fold")
    return {
        "schema": COHORT_SCHEMA,
        "schema_version": COHORT_SCHEMA_VERSION,
        "fold": fold,
        "fold_sample_count": len(fold_sample_ids),
        "fold_sample_ids_sha256": actual,
        "preprocessing_sha256": binding["preprocessing_sha256"],
        "explained_sample_ids": explained,
        "explained_count": len(explained),
        "is_full_fold": len(explained) == len(fold_sample_ids),
    }


def validate_cohort_v2(cohort: dict[str, Any], binding: dict[str, Any]) -> None:
    """Re-check a written cohort against the run it claims to describe."""
    if cohort.get("schema") != COHORT_SCHEMA or cohort.get("schema_version") != COHORT_SCHEMA_VERSION:
        raise ValueError(f"not a {COHORT_SCHEMA!r} version {COHORT_SCHEMA_VERSION} cohort")
    if cohort.get("fold") not in ALLOWED_FOLDS:
        raise ValueError(f"cohort fold must be one of {ALLOWED_FOLDS}")
    if cohort["fold_sample_ids_sha256"] != binding["split_sample_ids_sha256"][cohort["fold"]]:
        raise ValueError("cohort fold hash differs from binding.json")
    if cohort["preprocessing_sha256"] != binding["preprocessing_sha256"]:
        raise ValueError("cohort preprocessing hash differs from binding.json")
    ids = cohort["explained_sample_ids"]
    if len(set(ids)) != len(ids) or cohort["explained_count"] != len(ids):
        raise ValueError("cohort explained ids are not unique or the count is wrong")
