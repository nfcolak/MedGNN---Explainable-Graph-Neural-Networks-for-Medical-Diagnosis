"""CEI-GNN v3 screen fold: deterministic selector, hash-bound record, fold encoder.

Unit U4 of the v3 extensions split (EXT spec §9; v3 spec §5 as amended by §12
F4–F7, EXT §4 E3). Stub: behaviour is added step by step under TDD.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Optional

import torch

from .contracts import (VISIT_MEMBERSHIP_FILENAME, iter_graphs_with_membership,
                        sample_ids_sha256)
from .schema import sha256
from .tensorize import encode_graph, load_preprocessing
from .train import load_targets, select_top_labels

SCREEN_FOLD_NAME = 'screen'
SELECTOR_VERSION = 'cei_v3_screen_selector_v1'
SERIALIZATION_VERSION = 'json_compact_utf8_v1'  # contracts.sample_ids_sha256 layout
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
    """Hash-bound screen fold record (v3 §5 "Hash binding and replay", §12.5–6).

    The record holds aggregates and SHA-256 digests only; it never lists sample or
    subject identifiers. `screen_ids` must be exactly what the frozen selector draws
    from the immutable target file and the bound parent ids, in order.
    """
    if serialization_version != SERIALIZATION_VERSION:
        raise ValueError(f'unsupported id serialization version {serialization_version!r}; '
                         f'this module binds {SERIALIZATION_VERSION!r}')
    screen_ids = list(screen_ids)
    train_ids, dev_ids = set(train_ids), set(dev_ids)
    targets, kept = load_screen_targets(targets_path, top_k_labels=top_k_labels)
    expected = select_screen_ids(targets, train_ids, dev_ids, screen_limit=screen_limit,
                                 screen_seed=screen_seed, eligible=eligible)
    if screen_ids != expected:
        raise ValueError('screen_ids differ from the frozen selector draw (order, content '
                         'or count); the record binds only the reproducible draw')
    candidates = screen_candidates(targets, train_ids, dev_ids, eligible=eligible)
    excluded = sorted(excluded_subjects(targets, train_ids, dev_ids))
    environment = selector_environment()
    return {
        'fold': SCREEN_FOLD_NAME,
        'selector_version': SELECTOR_VERSION,
        'python_version': environment['python_version'],
        'seed_string_format': SEED_STRING_FORMAT,
        'seed_string': SEED_STRING_FORMAT.format(screen_seed=screen_seed),
        'screen_seed': int(screen_seed),
        'screen_limit': int(screen_limit),
        'eligibility_rule': ELIGIBILITY_RULE,
        'eligible_bound': eligible is not None,
        'top_k_labels': None if top_k_labels is None else int(top_k_labels),
        'class_order': None if kept is None else list(range(len(kept))),
        'kept_label_indices': kept,
        'serialization_version': SERIALIZATION_VERSION,
        'targets_sha256': sha256(Path(targets_path)),
        'screen_sample_ids_sha256': sample_ids_sha256(screen_ids),
        'excluded_subjects_sha256': sample_ids_sha256(excluded),
        'excluded_subject_count': len(excluded),
        'row_count': len(screen_ids),
        'distinct_subject_count': len({targets[sid][2] for sid in screen_ids}),
        'candidate_row_count': len(candidates),
        'candidate_subject_count': len({targets[sid][2] for sid in candidates}),
        'artifact_sha256': artifact_sha256,
        'preprocessing_sha256': preprocessing_sha256,
        'parent_train_sample_ids_sha256': sample_ids_sha256(sorted(train_ids)),
        'parent_train_count': len(train_ids),
        'parent_dev_sample_ids_sha256': sample_ids_sha256(sorted(dev_ids)),
        'parent_dev_count': len(dev_ids),
    }


_REPLAY_KEYS = ('fold', 'selector_version', 'seed_string_format', 'seed_string', 'screen_seed',
                'screen_limit', 'eligibility_rule', 'eligible_bound', 'top_k_labels',
                'class_order', 'kept_label_indices', 'serialization_version', 'targets_sha256',
                'screen_sample_ids_sha256', 'excluded_subjects_sha256',
                'excluded_subject_count', 'row_count', 'distinct_subject_count',
                'candidate_row_count', 'candidate_subject_count',
                'parent_train_sample_ids_sha256', 'parent_train_count',
                'parent_dev_sample_ids_sha256', 'parent_dev_count')


def assert_screen_replay(record, targets_path, train_ids, dev_ids, *, eligible=None) -> None:
    """Recompute the draw from the immutable target file and bound parents.

    Raises ValueError on any drift: target bytes, parent ids, seed, limit, versions,
    ordered screen hash, counts, excluded subjects, duplicates, or a subject shared
    with train or dev. Pure: `record` is not modified.
    """
    if not isinstance(record, dict):
        raise ValueError('screen record must be a dict')
    missing = [key for key in _REPLAY_KEYS if key not in record]
    if missing:
        raise ValueError(f'screen record lacks bound fields: {missing}')
    if record['fold'] != SCREEN_FOLD_NAME:
        raise ValueError('screen record fold name differs')
    if record['selector_version'] != SELECTOR_VERSION:
        raise ValueError('screen record was produced by another selector version')
    if record['serialization_version'] != SERIALIZATION_VERSION:
        raise ValueError('screen record binds another id serialization version')
    if record['seed_string_format'] != SEED_STRING_FORMAT:
        raise ValueError('screen record binds another seed string format')
    if type(record['screen_seed']) is not int or type(record['screen_limit']) is not int:
        raise ValueError('screen record seed/limit must be integers')
    if record['seed_string'] != SEED_STRING_FORMAT.format(screen_seed=record['screen_seed']):
        raise ValueError('screen record seed string differs from its seed')
    if record['eligibility_rule'] != ELIGIBILITY_RULE:
        raise ValueError('screen record binds another eligibility rule')
    if record['eligible_bound'] != (eligible is not None):
        raise ValueError('screen record eligibility set presence differs from replay input')
    targets_path = Path(targets_path)
    if record['targets_sha256'] != sha256(targets_path):
        raise ValueError('target sidecar bytes differ from the screen record')
    train_ids, dev_ids = set(train_ids), set(dev_ids)
    if record['parent_train_sample_ids_sha256'] != sample_ids_sha256(sorted(train_ids)):
        raise ValueError('parent train sample ids differ from the screen record')
    if record['parent_dev_sample_ids_sha256'] != sample_ids_sha256(sorted(dev_ids)):
        raise ValueError('parent dev sample ids differ from the screen record')
    if record['parent_train_count'] != len(train_ids) or record['parent_dev_count'] != len(dev_ids):
        raise ValueError('parent sample counts differ from the screen record')
    targets, kept = load_screen_targets(targets_path, top_k_labels=record['top_k_labels'])
    class_order = None if kept is None else list(range(len(kept)))
    if record['class_order'] != class_order or record['kept_label_indices'] != kept:
        raise ValueError('class order differs from the screen record')
    excluded = sorted(excluded_subjects(targets, train_ids, dev_ids))
    if (record['excluded_subjects_sha256'] != sample_ids_sha256(excluded)
            or record['excluded_subject_count'] != len(excluded)):
        raise ValueError('excluded subject set differs from the screen record')
    replay = select_screen_ids(targets, train_ids, dev_ids, screen_limit=record['screen_limit'],
                               screen_seed=record['screen_seed'], eligible=eligible)
    if len(replay) != record['row_count'] or len(set(replay)) != record['row_count']:
        raise ValueError('replayed screen row count differs or has duplicate ids')
    if record['screen_sample_ids_sha256'] != sample_ids_sha256(replay):
        raise ValueError('ordered screen sample ids differ from the screen record')
    if set(replay) & (train_ids | dev_ids):
        raise ValueError('replayed screen ids intersect train or dev ids')
    replay_subjects = {targets[sid][2] for sid in replay}
    if replay_subjects & set(excluded):
        raise ValueError('replayed screen subjects intersect train or dev subjects')
    if record['distinct_subject_count'] != len(replay_subjects):
        raise ValueError('screen distinct subject count differs from the screen record')
    candidates = screen_candidates(targets, train_ids, dev_ids, eligible=eligible)
    if (record['candidate_row_count'] != len(candidates)
            or record['candidate_subject_count'] != len({targets[s][2] for s in candidates})):
        raise ValueError('candidate counts differ from the screen record')
    return None


FOLD_SPLITS = {'dev': 'train', 'screen': 'train', 'validation': 'validation'}


def fold_split(fold: str) -> str:
    """Target-sidecar split that a fold's rows must carry (v3 §12.7); no test value."""
    if fold not in FOLD_SPLITS:
        raise ValueError(f"fold must be one of {sorted(FOLD_SPLITS)}, not {fold!r}; "
                         'there is no test fold value')
    return FOLD_SPLITS[fold]


def _resolve_preprocessing(prep_state, preprocessing_sha256):
    """Return (state dict, sha256 of its preprocessing.json bytes), refusing drift."""
    if isinstance(prep_state, dict):
        state = prep_state
        # train.py line 737 layout, so a state dict hashes like its written file.
        payload = (json.dumps(state, indent=2, sort_keys=True) + '\n').encode('utf-8')
        digest = hashlib.sha256(payload).hexdigest()
    else:
        path = Path(prep_state)
        if not path.is_file():
            raise ValueError('preprocessing state path does not exist')
        digest = sha256(path)
        state = json.loads(path.read_text())
    if preprocessing_sha256 is not None and digest != preprocessing_sha256:
        raise ValueError('preprocessing_sha256 differs from the encoder preprocessing state; '
                         'the checkpoint binding and this encoder do not match')
    return state, digest


def _check_approval(fold, allow_validation, approval_record):
    if fold != 'validation':
        return
    if allow_validation is not True:
        raise ValueError("fold='validation' requires allow_validation=True and the bound "
                         'item-5 approval record (EXT §4.2)')
    if not isinstance(approval_record, dict) or approval_record.get('allow_validation') is not True:
        raise ValueError("fold='validation' requires an approval record with "
                         "allow_validation: true (EXT §4.2); validation is not scored otherwise")


def encode_rows(artifact, prep_state, ids, *, fold: str, edge_direction: str,
                allow_validation: bool = False,
                approval_record: Optional[dict] = None,
                targets=None, preprocessing_sha256: Optional[str] = None) -> Iterator:
    """Encode exactly `ids` of one fold with a frozen preprocessing state (v3 §12.7).

    `prep_state` is the run's `preprocessing.json` path or its loaded dict; when
    `preprocessing_sha256` is given it must equal that file's digest. `targets` is
    the Top-k-filtered targets dict (or the sidecar path, filtered with Top-10).
    Every id is checked against `targets` and `fold_split(fold)` before the artifact
    is opened; `split == 'test'` rows are skipped as text before `encode_graph` in
    every fold, and a test id can never be requested because no fold maps to it.
    Rows are yielded in artifact order with `y` and `subject` attached.
    """
    expected_split = fold_split(fold)
    _check_approval(fold, allow_validation, approval_record)
    if targets is None:
        raise ValueError('targets (Top-k-filtered targets dict or sidecar path) are required')
    if not isinstance(targets, dict):
        targets, _kept = load_screen_targets(targets, top_k_labels=10)
    ids = list(ids)
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate ids requested for encoding')
    for sid in ids:
        entry = targets.get(sid)
        if entry is None:
            raise ValueError(f'id is not a member of the targets: {fold} fold refuses it')
        if entry[1] != expected_split:
            raise ValueError(f"id carries split {entry[1]!r}, not the {expected_split!r} split "
                             f"that fold={fold!r} requires")
    state, _digest = _resolve_preprocessing(prep_state, preprocessing_sha256)
    prep = load_preprocessing(state)
    wanted = set(ids)
    artifact = Path(artifact)
    graphs_path = artifact / 'graphs.jsonl'
    membership_path = artifact / VISIT_MEMBERSHIP_FILENAME

    def rows():
        seen = set()
        for graph, visit_membership in iter_graphs_with_membership(graphs_path, membership_path):
            sid = graph['sample_id']
            entry = targets.get(sid)
            if entry is None:
                continue
            y, split, subject = entry
            if split == 'test':
                continue  # held out; dropped as text in every fold, never tensorised
            if sid not in wanted:
                continue
            if split != expected_split:
                raise ValueError('fold/split mismatch detected before encoding')
            data = encode_graph(graph, prep, visit_membership, edge_direction=edge_direction)
            data.y = torch.tensor([y], dtype=torch.long)
            data.subject = subject
            seen.add(sid)
            yield data
        if seen != wanted:
            raise ValueError(f'{len(wanted - seen)} requested {fold} rows are absent from the '
                             'artifact; the target sidecar is not artifact-local')
    return rows()
