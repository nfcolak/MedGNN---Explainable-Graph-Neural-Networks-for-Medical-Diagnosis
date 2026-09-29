"""Tests for the CEI-GNN v3 screen fold (unit U4): selector, record replay, encoder.

Synthetic targets dicts and tiny tmp_path artifacts only. No real data, no
training, no test fold: `split == 'test'` rows exist in fixtures solely to prove
they are dropped before tensorisation.
"""
import random
import sys

import pytest

from comparison.standardized.clinical_graph_v2 import cei_v3_screen as screen
from comparison.standardized.clinical_graph_v2 import train

# ------------------------------------------------------------------ fixtures


def synthetic_targets(n_train=6000, subjects=5500, n_validation=40, n_test=20,
                      num_labels=train.NUM_CLASSES):
    """Top-k-filtered-shaped targets: sid -> (target, split, subject).

    `subjects < n_train` so some patients own several TRAIN rows, which is what the
    subject-disjointness assertions need to bite on.
    """
    targets = {}
    for i in range(n_train):
        targets[f'tr-{i:05d}'] = (i % num_labels, 'train', f'p-{i % subjects:05d}')
    for i in range(n_validation):
        targets[f'va-{i:05d}'] = (i % num_labels, 'validation', f'pv-{i:05d}')
    for i in range(n_test):
        targets[f'te-{i:05d}'] = (i % num_labels, 'test', f'pt-{i:05d}')
    return targets


def parent_ids(targets, n_train=300, n_dev=100):
    """A train sample plus a patient-disjoint dev split drawn like train.py."""
    train_ids = set(random.Random(1234).sample(
        sorted(sid for sid, e in targets.items() if e[1] == 'train'), n_train))
    dev_ids = train.select_dev_ids(targets, train_ids, n_dev, 1234)
    return train_ids, dev_ids


# ------------------------------------------------------- step 1: selector


def test_selector_is_deterministic_and_bound_to_the_seed_string():
    targets = synthetic_targets()
    train_ids, dev_ids = parent_ids(targets)
    first = screen.select_screen_ids(targets, train_ids, dev_ids)
    second = screen.select_screen_ids(targets, train_ids, dev_ids)
    assert first == second
    assert isinstance(first, list)
    assert len(first) == 5000
    # v3 §12.5: the draw is random.Random(f'screen-{screen_seed}').sample(candidates, limit)
    candidates = screen.screen_candidates(targets, train_ids, dev_ids)
    assert len(candidates) >= 5000
    expected = random.Random('screen-20260929').sample(candidates, 5000)
    assert first == expected
    other_seed = screen.select_screen_ids(targets, train_ids, dev_ids, screen_seed=1)
    assert other_seed != first


def test_selector_returns_exactly_5000_distinct_rows_by_default():
    targets = synthetic_targets()
    train_ids, dev_ids = parent_ids(targets)
    chosen = screen.select_screen_ids(targets, train_ids, dev_ids)
    assert len(chosen) == 5000
    assert len(set(chosen)) == 5000
    assert screen.DEFAULT_SCREEN_LIMIT == 5000
    assert screen.DEFAULT_SCREEN_SEED == 20260929


def test_selector_fails_closed_when_fewer_candidates_remain():
    targets = synthetic_targets(n_train=800, subjects=800)
    train_ids, dev_ids = parent_ids(targets)
    with pytest.raises(ValueError, match='patient-disjoint TRAIN rows remain'):
        screen.select_screen_ids(targets, train_ids, dev_ids)
    # The limit is honoured when enough remain and excludes train/dev ids.
    small = screen.select_screen_ids(targets, train_ids, dev_ids, screen_limit=50)
    assert len(small) == 50
    assert not (set(small) & (set(train_ids) | set(dev_ids)))


def test_selector_filters_train_split_explicitly_and_honours_eligible():
    targets = synthetic_targets(n_train=1200, subjects=1200, n_validation=200, n_test=100)
    train_ids, dev_ids = parent_ids(targets, n_train=100, n_dev=50)
    chosen = screen.select_screen_ids(targets, train_ids, dev_ids, screen_limit=200)
    assert all(targets[sid][1] == 'train' for sid in chosen)
    assert not any(sid.startswith(('va-', 'te-')) for sid in chosen)
    eligible = {sid for sid in targets if sid.endswith(('0', '2', '4', '6', '8'))}
    limited = screen.select_screen_ids(targets, train_ids, dev_ids, screen_limit=200,
                                       eligible=eligible)
    assert set(limited) <= eligible
    assert screen.screen_candidates(targets, train_ids, dev_ids, eligible=eligible) == sorted(
        sid for sid, e in targets.items() if e[1] == 'train' and sid not in train_ids
        and sid not in dev_ids and sid in eligible
        and e[2] not in {targets[s][2] for s in train_ids | set(dev_ids)})


def test_selector_uses_top_k_filtered_targets(tmp_path):
    """v3 §12.4: the selector receives targets AFTER select_top_labels (train.py 597)."""
    rows = ['sample_id,target,split,subject_id']
    for i in range(400):
        # labels 0..9 frequent on TRAIN; labels 10 and 11 rare, dropped by Top-10
        label = i % 10 if i < 360 else 10 + (i % 2)
        rows.append(f'row-{i:04d},{label},train,subj-{i:04d}')
    for i in range(20):
        rows.append(f'val-{i:04d},{i % 10},validation,vsubj-{i:04d}')
    for i in range(10):
        rows.append(f'tst-{i:04d},{i % 10},test,tsubj-{i:04d}')
    path = tmp_path / 'targets.csv'
    path.write_text('\n'.join(rows) + '\n')
    targets, kept = screen.load_screen_targets(path, top_k_labels=10)
    assert kept == list(range(10))
    assert not any(sid >= 'row-0360' and sid.startswith('row-') for sid in targets)
    assert len(targets) == 360 + 20 + 10
    assert all(0 <= entry[0] < 10 for entry in targets.values())
    train_ids, dev_ids = parent_ids(targets, n_train=50, n_dev=20)
    chosen = screen.select_screen_ids(targets, train_ids, dev_ids, screen_limit=100)
    assert all(sid < 'row-0360' for sid in chosen)


def test_selector_screen_subjects_are_disjoint_from_train_and_dev_subjects():
    targets = synthetic_targets()
    train_ids, dev_ids = parent_ids(targets)
    chosen = screen.select_screen_ids(targets, train_ids, dev_ids)
    assert len(chosen) == 5000
    excluded = {targets[sid][2] for sid in set(train_ids) | set(dev_ids)}
    screen_subjects = {targets[sid][2] for sid in chosen}
    assert not (screen_subjects & excluded)
    assert not (set(chosen) & (set(train_ids) | set(dev_ids)))
    # Every excluded subject's rows are absent even when the row itself is not a parent id
    sibling_rows = {sid for sid, e in targets.items()
                    if e[1] == 'train' and e[2] in excluded and sid not in train_ids
                    and sid not in dev_ids}
    assert sibling_rows, 'fixture must contain sibling rows of sampled patients'
    assert not (set(chosen) & sibling_rows)


def test_selector_environment_binds_sys_version():
    env = screen.selector_environment()
    assert env.get('python_version') == sys.version
    assert env.get('selector_version') == screen.SELECTOR_VERSION
    assert env.get('seed_string_format') == 'screen-{screen_seed}'
