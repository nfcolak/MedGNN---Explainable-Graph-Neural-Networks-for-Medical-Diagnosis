"""Tests for the CEI-GNN v3 screen fold (unit U4): selector, record replay, encoder.

Synthetic targets dicts and tiny tmp_path artifacts only. No real data, no
training, no test fold: `split == 'test'` rows exist in fixtures solely to prove
they are dropped before tensorisation.
"""
import json
import random
import sys

import pytest

from comparison.standardized.clinical_graph_v2 import cei_v3_screen as screen
from comparison.standardized.clinical_graph_v2 import tensorize as tz
from comparison.standardized.clinical_graph_v2 import train
from comparison.standardized.clinical_graph_v2.contracts import (
    VISIT_MEMBERSHIP_CONTRACT_VERSION, VISIT_MEMBERSHIP_FILENAME, sample_ids_sha256)
from comparison.standardized.clinical_graph_v2.schema import sha256

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


# ------------------------------------------------- step 2: record replay


def write_targets_csv(path, n_train=1500, subjects=1400, n_validation=30, n_test=10):
    rows = ['sample_id,target,split,subject_id']
    for i in range(n_train):
        rows.append(f'tr-{i:05d},{i % train.NUM_CLASSES},train,p-{i % subjects:05d}')
    for i in range(n_validation):
        rows.append(f'va-{i:05d},{i % train.NUM_CLASSES},validation,pv-{i:05d}')
    for i in range(n_test):
        rows.append(f'te-{i:05d},{i % train.NUM_CLASSES},test,pt-{i:05d}')
    path.write_text('\n'.join(rows) + '\n')
    return path


def record_fixture(tmp_path, screen_limit=200):
    targets_path = write_targets_csv(tmp_path / 'targets.csv')
    targets, kept = screen.load_screen_targets(targets_path, top_k_labels=10)
    train_ids, dev_ids = parent_ids(targets, n_train=100, n_dev=50)
    screen_ids = screen.select_screen_ids(targets, train_ids, dev_ids, screen_limit=screen_limit)
    record = screen.screen_record(targets_path, screen_ids, train_ids, dev_ids,
                                  artifact_sha256='a' * 64, preprocessing_sha256='b' * 64,
                                  serialization_version='json_compact_utf8_v1',
                                  screen_limit=screen_limit)
    return targets_path, targets, train_ids, dev_ids, screen_ids, record


def test_screen_record_binds_hashes_counts_and_parents(tmp_path):
    targets_path, targets, train_ids, dev_ids, screen_ids, record = record_fixture(tmp_path)
    assert record.get('fold') == 'screen'
    assert record.get('selector_version') == screen.SELECTOR_VERSION
    assert record.get('screen_seed') == 20260929
    assert record.get('screen_limit') == 200
    assert record.get('seed_string') == 'screen-20260929'
    assert record.get('eligibility_rule') == screen.ELIGIBILITY_RULE
    assert record.get('python_version') == sys.version
    assert record.get('serialization_version') == 'json_compact_utf8_v1'
    assert record.get('targets_sha256') == sha256(targets_path)
    assert record.get('screen_sample_ids_sha256') == sample_ids_sha256(screen_ids)
    excluded = sorted({targets[sid][2] for sid in set(train_ids) | set(dev_ids)})
    assert record.get('excluded_subjects_sha256') == sample_ids_sha256(excluded)
    assert record.get('excluded_subject_count') == len(excluded)
    assert record.get('row_count') == 200
    assert record.get('distinct_subject_count') == len({targets[s][2] for s in screen_ids})
    assert record.get('class_order') == list(range(10))
    assert record.get('top_k_labels') == 10
    assert record.get('artifact_sha256') == 'a' * 64
    assert record.get('preprocessing_sha256') == 'b' * 64
    assert record.get('parent_train_sample_ids_sha256') == sample_ids_sha256(sorted(train_ids))
    assert record.get('parent_dev_sample_ids_sha256') == sample_ids_sha256(sorted(dev_ids))
    assert record.get('parent_train_count') == len(train_ids)
    assert record.get('parent_dev_count') == len(dev_ids)
    # v3 §12.6: candidate aggregates are bound; the record never lists patient ids.
    candidates = screen.screen_candidates(targets, train_ids, dev_ids)
    assert record.get('candidate_row_count') == len(candidates)
    assert record.get('candidate_subject_count') == len({targets[s][2] for s in candidates})
    flat = json.dumps(record)
    assert 'tr-' not in flat and 'p-0' not in flat


def test_screen_record_replay_passes_and_is_pure(tmp_path):
    targets_path, _targets, train_ids, dev_ids, screen_ids, record = record_fixture(tmp_path)
    assert record.get('screen_sample_ids_sha256') == sample_ids_sha256(screen_ids)
    before = json.dumps(record, sort_keys=True)
    assert screen.assert_screen_replay(record, targets_path, train_ids, dev_ids) is None
    assert json.dumps(record, sort_keys=True) == before


@pytest.mark.parametrize('key, value', [
    ('screen_sample_ids_sha256', 'f' * 64),
    ('excluded_subjects_sha256', 'f' * 64),
    ('targets_sha256', 'f' * 64),
    ('row_count', 199),
    ('screen_seed', 20260930),
    ('screen_limit', 199),
    ('parent_train_sample_ids_sha256', 'f' * 64),
    ('parent_dev_sample_ids_sha256', 'f' * 64),
    ('selector_version', 'other'),
    ('serialization_version', 'other'),
    ('class_order', list(range(9))),
])
def test_screen_record_replay_detects_tampering(tmp_path, key, value):
    targets_path, _t, train_ids, dev_ids, _s, record = record_fixture(tmp_path)
    tampered = dict(record)
    tampered[key] = value
    with pytest.raises(ValueError):
        screen.assert_screen_replay(tampered, targets_path, train_ids, dev_ids)


def test_screen_record_replay_detects_changed_targets_file_and_parents(tmp_path):
    targets_path, targets, train_ids, dev_ids, _s, record = record_fixture(tmp_path)
    # different parent ids -> excluded subjects and draw differ
    other_train = set(sorted(train_ids)[:-1]) | {sorted(
        sid for sid in targets if targets[sid][1] == 'train' and sid not in train_ids
        and sid not in dev_ids)[0]}
    with pytest.raises(ValueError):
        screen.assert_screen_replay(record, targets_path, other_train, dev_ids)
    # the immutable target file changed by one byte
    text = targets_path.read_text()
    targets_path.write_text(text.replace('te-00000', 'te-00099', 1))
    with pytest.raises(ValueError):
        screen.assert_screen_replay(record, targets_path, train_ids, dev_ids)


def test_screen_record_rejects_ids_that_the_selector_would_not_draw(tmp_path):
    targets_path = write_targets_csv(tmp_path / 'targets.csv')
    targets, _ = screen.load_screen_targets(targets_path, top_k_labels=10)
    train_ids, dev_ids = parent_ids(targets, n_train=100, n_dev=50)
    screen_ids = screen.select_screen_ids(targets, train_ids, dev_ids, screen_limit=200)
    wrong = list(screen_ids[1:]) + [screen_ids[0]]  # same set, different order
    with pytest.raises(ValueError):
        screen.screen_record(targets_path, wrong, train_ids, dev_ids, artifact_sha256='a' * 64,
                             preprocessing_sha256='b' * 64,
                             serialization_version='json_compact_utf8_v1', screen_limit=200)
    with pytest.raises(ValueError):
        screen.screen_record(targets_path, screen_ids + [screen_ids[0]], train_ids, dev_ids,
                             artifact_sha256='a' * 64, preprocessing_sha256='b' * 64,
                             serialization_version='json_compact_utf8_v1', screen_limit=201)


# ---------------------------------------------- step 3: fold encoder refusals


def clinical_graph(sample_id, value=70.0):
    """One contract-valid graph: patient, index visit, complaint, vital."""
    nodes = [
        {'id': 'p', 'kind': 'patient', 'token': 'patient', 'age': 40,
         'gender': 'F', 'race': 'A', 'arrival_transport': 'WALK IN'},
        {'id': 'v', 'kind': 'visit', 'token': 'visit:index', 'acuity': 2},
        {'id': 'c', 'kind': 'complaint', 'token': 'cc:chest pain'},
        {'id': 'hr', 'kind': 'vital', 'token': 'vital:heartrate', 'unit': 'bpm',
         'value': value, 'time_hours': -1.0, 'available_hours': -1.0},
    ]
    edges = [
        {'source': 'p', 'target': 'v', 'relation': 'has_visit', 'informative': False},
        {'source': 'v', 'target': 'p', 'relation': 'index_visit_of', 'informative': False},
        {'source': 'v', 'target': 'c', 'relation': 'reports_complaint', 'informative': False},
        {'source': 'v', 'target': 'hr', 'relation': 'observed_vital', 'informative': False},
    ]
    return {'sample_id': sample_id, 'nodes': nodes, 'edges': edges,
            'coverage': {'complaints': 1, 'index_measurements': 0,
                         'prior_visits': 0, 'informative_edges': 0}}


def membership(graph):
    kinds = [node['kind'] for node in graph['nodes']]
    return {'contract_version': VISIT_MEMBERSHIP_CONTRACT_VERSION,
            'sample_id': graph['sample_id'], 'visit_ordinals': [0],
            'membership_pairs': [[0, i] for i, kind in enumerate(kinds) if kind != 'patient'],
            'global_node_mask': [kind == 'patient' for kind in kinds]}


def encoder_fixture(tmp_path):
    """Tiny artifact: 6 train rows (2 sampled train, 2 dev, 2 screen), 2 validation, 2 test."""
    root = tmp_path / 'artifact'
    root.mkdir()
    ids = [f'g-{i:02d}' for i in range(10)]
    splits = ['train'] * 6 + ['validation'] * 2 + ['test'] * 2
    graphs = [clinical_graph(sid, 55.0 + 3 * i) for i, sid in enumerate(ids)]
    (root / 'graphs.jsonl').write_text(''.join(json.dumps(g) + '\n' for g in graphs))
    (root / VISIT_MEMBERSHIP_FILENAME).write_text(
        ''.join(json.dumps(membership(g)) + '\n' for g in graphs))
    targets = {sid: (i % 2, split, f'subj-{i:02d}')
               for i, (sid, split) in enumerate(zip(ids, splits))}
    train_ids, dev_ids, screen_ids = ids[0:2], ids[2:4], ids[4:6]
    prep = tz.fit_preprocessing(root / 'graphs.jsonl', set(train_ids), 1,
                                membership_path=root / VISIT_MEMBERSHIP_FILENAME)
    state = tz.preprocessing_state(prep)
    prep_path = tmp_path / 'preprocessing.json'
    prep_path.write_text(json.dumps(state, indent=2, sort_keys=True) + '\n')
    return dict(root=root, targets=targets, train_ids=train_ids, dev_ids=dev_ids,
                screen_ids=screen_ids, validation_ids=ids[6:8], test_ids=ids[8:10],
                state=state, prep_path=prep_path, prep_sha=sha256(prep_path))


def counting_encode_graph(monkeypatch):
    calls = []
    real = screen.encode_graph

    def counted(graph, *args, **kwargs):
        calls.append(graph['sample_id'])
        return real(graph, *args, **kwargs)
    monkeypatch.setattr(screen, 'encode_graph', counted)
    return calls


def test_encoder_yields_exactly_the_requested_rows_with_labels(tmp_path, monkeypatch):
    fx = encoder_fixture(tmp_path)
    calls = counting_encode_graph(monkeypatch)
    rows = list(screen.encode_rows(fx['root'], fx['state'], fx['screen_ids'], fold='screen',
                                   edge_direction='bidirectional', targets=fx['targets'],
                                   preprocessing_sha256=fx['prep_sha']))
    assert [d.sample_id for d in rows] == fx['screen_ids']
    assert calls == fx['screen_ids']
    assert [int(d.y) for d in rows] == [fx['targets'][s][0] for s in fx['screen_ids']]
    assert [d.subject for d in rows] == [fx['targets'][s][2] for s in fx['screen_ids']]
    assert all(d.edge_index.shape[1] > 4 for d in rows), 'bidirectional view expected'
    dev = list(screen.encode_rows(fx['prep_path'].parent / 'artifact', fx['prep_path'],
                                  fx['dev_ids'], fold='dev', edge_direction='forward',
                                  targets=fx['targets']))
    assert [d.sample_id for d in dev] == fx['dev_ids']


def test_encoder_refuses_non_member_ids_before_encoding(tmp_path, monkeypatch):
    fx = encoder_fixture(tmp_path)
    calls = counting_encode_graph(monkeypatch)
    # id absent from the targets entirely
    with pytest.raises(ValueError):
        list(screen.encode_rows(fx['root'], fx['state'], fx['screen_ids'] + ['g-99'],
                                fold='screen', edge_direction='forward', targets=fx['targets']))
    # id from another split than the fold demands (validation row under fold='screen')
    with pytest.raises(ValueError):
        list(screen.encode_rows(fx['root'], fx['state'],
                                fx['screen_ids'] + fx['validation_ids'][:1],
                                fold='screen', edge_direction='forward', targets=fx['targets']))
    # id in the targets but missing from the artifact
    targets = dict(fx['targets'])
    targets['g-77'] = (0, 'train', 'subj-77')
    with pytest.raises(ValueError, match='absent from the artifact'):
        list(screen.encode_rows(fx['root'], fx['state'], fx['screen_ids'] + ['g-77'],
                                fold='screen', edge_direction='forward', targets=targets))
    assert 'g-99' not in calls and 'g-77' not in calls
    assert not (set(calls) & set(fx['validation_ids']))
    with pytest.raises(ValueError):
        list(screen.encode_rows(fx['root'], fx['state'], fx['screen_ids'], fold='holdout',
                                edge_direction='forward', targets=fx['targets']))


def test_encoder_refuses_preprocessing_hash_mismatch(tmp_path, monkeypatch):
    fx = encoder_fixture(tmp_path)
    calls = counting_encode_graph(monkeypatch)
    with pytest.raises(ValueError, match='preprocessing'):
        list(screen.encode_rows(fx['root'], fx['state'], fx['screen_ids'], fold='screen',
                                edge_direction='forward', targets=fx['targets'],
                                preprocessing_sha256='0' * 64))
    with pytest.raises(ValueError, match='preprocessing'):
        list(screen.encode_rows(fx['root'], fx['prep_path'], fx['screen_ids'], fold='screen',
                                edge_direction='forward', targets=fx['targets'],
                                preprocessing_sha256='0' * 64))
    assert calls == []
    stale = dict(fx['state'])
    stale['preprocessing_version'] = 'other'
    with pytest.raises(ValueError):
        list(screen.encode_rows(fx['root'], stale, fx['screen_ids'], fold='screen',
                                edge_direction='forward', targets=fx['targets']))
    assert calls == []


def test_encoder_refuses_validation_without_approval_record(tmp_path, monkeypatch):
    fx = encoder_fixture(tmp_path)
    calls = counting_encode_graph(monkeypatch)
    denied = [dict(allow_validation=False), dict(allow_validation=True, approval_record=None),
              dict(allow_validation=True, approval_record={'allow_validation': False}),
              dict(allow_validation=True, approval_record={'allow_validation': 'true'}),
              dict(allow_validation=False, approval_record={'allow_validation': True})]
    for kwargs in denied:
        with pytest.raises(ValueError, match='validation'):
            list(screen.encode_rows(fx['root'], fx['state'], fx['validation_ids'],
                                    fold='validation', edge_direction='forward',
                                    targets=fx['targets'], **kwargs))
    assert calls == []
    rows = list(screen.encode_rows(fx['root'], fx['state'], fx['validation_ids'],
                                   fold='validation', edge_direction='forward',
                                   targets=fx['targets'], allow_validation=True,
                                   approval_record={'allow_validation': True,
                                                    'approval_reference': 'synthetic'}))
    assert [d.sample_id for d in rows] == fx['validation_ids']
    assert calls == fx['validation_ids']


def test_encoder_never_encodes_test_rows(tmp_path, monkeypatch):
    fx = encoder_fixture(tmp_path)
    calls = counting_encode_graph(monkeypatch)
    for fold, ids, extra in (('screen', fx['screen_ids'], {}), ('dev', fx['dev_ids'], {}),
                             ('validation', fx['validation_ids'],
                              dict(allow_validation=True,
                                   approval_record={'allow_validation': True}))):
        rows = list(screen.encode_rows(fx['root'], fx['state'], ids, fold=fold,
                                       edge_direction='forward', targets=fx['targets'], **extra))
        assert [d.sample_id for d in rows] == ids
    assert not (set(calls) & set(fx['test_ids']))
    assert calls == fx['screen_ids'] + fx['dev_ids'] + fx['validation_ids']
    # a test id requested explicitly is refused, and still never encoded
    for fold in ('screen', 'validation'):
        with pytest.raises(ValueError):
            list(screen.encode_rows(fx['root'], fx['state'], fx['test_ids'], fold=fold,
                                    edge_direction='forward', targets=fx['targets'],
                                    allow_validation=True,
                                    approval_record={'allow_validation': True}))
    assert not (set(calls) & set(fx['test_ids']))
    assert screen.fold_split('screen') == 'train'
    assert screen.fold_split('dev') == 'train'
    assert screen.fold_split('validation') == 'validation'
    with pytest.raises(ValueError):
        screen.fold_split('test')
