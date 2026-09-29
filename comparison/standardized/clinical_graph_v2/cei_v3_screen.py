"""CEI-GNN v3 screen fold: deterministic selector, hash-bound record, fold encoder.

Unit U4 of the v3 extensions split (EXT spec §9; v3 spec §5 as amended by §12
F4–F7, EXT §4 E3). Stub: behaviour is added step by step under TDD.
"""
from __future__ import annotations

import random
import sys
from collections.abc import Iterator
from typing import Optional

from .train import load_targets, select_top_labels

SCREEN_FOLD_NAME = 'screen'
SELECTOR_VERSION = 'cei_v3_screen_selector_v1'
SEED_STRING_FORMAT = 'screen-{screen_seed}'
DEFAULT_SCREEN_LIMIT = 5000
DEFAULT_SCREEN_SEED = 20260929
ELIGIBILITY_RULE = ("TRAIN-fold rows of the Top-k-filtered targets, not in train_ids or "
                    "dev_ids, subject not in {targets[sid][2] for sid in train_ids | dev_ids}, "
                    "and in `eligible` when given (the select_dev_ids eligibility set)")


def selector_environment() -> dict:
    """Interpreter binding for the screen draw (v3 §12.5).

    `random.sample` on a list is deterministic per interpreter version but its
    algorithm is an implementation detail, so `sys.version` is bound.
    """
    return {'python_version': sys.version,
            'selector_version': SELECTOR_VERSION,
            'seed_string_format': SEED_STRING_FORMAT}


def load_screen_targets(targets_path, *, top_k_labels=10):
    """Load the target sidecar and apply the same Top-k filter as train.py line 597.

    Returns `(targets, kept_labels)`; the selector must receive this filtered object
    (v3 §12.4), never the raw sidecar.
    """
    targets = load_targets(targets_path)
    if top_k_labels is None:
        return targets, None
    targets, kept, _dropped = select_top_labels(targets, int(top_k_labels))
    return targets, list(kept)


def excluded_subjects(targets, train_ids, dev_ids) -> set:
    """Subjects of the PRE-eligibility train sample and the dev split (v3 §12.4)."""
    return {targets[sid][2] for sid in set(train_ids) | set(dev_ids)}


def screen_candidates(targets, train_ids, dev_ids, *, eligible=None) -> list:
    """Sorted candidate TRAIN rows outside train/dev and their subjects (v3 §12.4).

    Mirrors `train.select_dev_ids` (train.py lines 333–337) with the dev split added
    to the exclusions; `entry[1] == 'train'` is filtered explicitly.
    """
    train_ids, dev_ids = set(train_ids), set(dev_ids)
    excluded = excluded_subjects(targets, train_ids, dev_ids)
    return sorted(sid for sid, entry in targets.items()
                  if entry[1] == 'train' and sid not in train_ids and sid not in dev_ids
                  and entry[2] not in excluded
                  and (eligible is None or sid in eligible))


def select_screen_ids(targets, train_ids, dev_ids, *, screen_limit=DEFAULT_SCREEN_LIMIT,
                      screen_seed=DEFAULT_SCREEN_SEED, eligible=None) -> list:
    """Deterministic patient-disjoint screen draw (v3 §5, §12.4–5).

    Fails closed when fewer than `screen_limit` candidates remain (v3 §12.6); the
    seed and algorithm are frozen and never adapted to outcomes.
    """
    if isinstance(screen_limit, bool) or int(screen_limit) < 1:
        raise ValueError('screen_limit must be a positive integer')
    if isinstance(screen_seed, bool) or type(screen_seed) is not int:
        raise ValueError('screen_seed must be bound as an integer')
    screen_limit = int(screen_limit)
    train_ids, dev_ids = set(train_ids), set(dev_ids)
    candidates = screen_candidates(targets, train_ids, dev_ids, eligible=eligible)
    if len(candidates) < screen_limit:
        raise ValueError(f'only {len(candidates)} patient-disjoint TRAIN rows remain for '
                         f'a {screen_limit}-row screen split')
    rng = random.Random(SEED_STRING_FORMAT.format(screen_seed=screen_seed))
    chosen = list(rng.sample(candidates, screen_limit))
    excluded = excluded_subjects(targets, train_ids, dev_ids)
    if len(set(chosen)) != screen_limit:
        raise AssertionError('screen draw produced duplicate sample ids')
    if set(chosen) & (train_ids | dev_ids):
        raise AssertionError('screen draw overlaps train or dev sample ids')
    if any(targets[sid][2] in excluded for sid in chosen):
        raise AssertionError('screen draw shares a subject with train or dev')
    return chosen


def screen_record(targets_path, screen_ids, train_ids, dev_ids, *, artifact_sha256,
                  preprocessing_sha256, serialization_version,
                  screen_limit=DEFAULT_SCREEN_LIMIT, screen_seed=DEFAULT_SCREEN_SEED,
                  top_k_labels=10, eligible=None) -> dict:
    return {}


def assert_screen_replay(record, targets_path, train_ids, dev_ids, *, eligible=None) -> None:
    return None


def encode_rows(artifact, prep_state, ids, *, fold: str, edge_direction: str,
                allow_validation: bool = False,
                approval_record: Optional[dict] = None) -> Iterator:
    return iter(())
