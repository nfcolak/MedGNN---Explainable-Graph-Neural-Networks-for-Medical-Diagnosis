"""Approved three-seed XGBoost training and frozen CEI-v3 comparisons.

No flag prints a read-only plan. --execute requires a fresh ignored output root.
The existing tabular_control main is called with process-local I/O guards: test
rows are discarded as text before graph/membership JSON or target interpretation.
Only dev selects boosting rounds. Screen/validation each receive one prediction
call per model, with fitted preprocessing and projection, never a refit.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
from itertools import islice, zip_longest
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from . import cei_pilot as pilot
from . import cei_v3_screen as screen
from . import cei_v3_study as study
from . import cei_v3_validation as validation
from . import cei_v3_vs_protgnn as comparison
from ....core import contracts, tensorize, train
from ....methods.xgboost import tabular_control as tabular
from ....paths import PACKAGE_ROOT
from .cei_v3_ext.validation_scoring import (
    APPROVAL_FILENAME, approval_record_sha256, validate_approval_record,
    write_approval_record,
)
from ....core.contracts import recursive_source_hashes, sample_ids_sha256
from ....core.schema import sha256

from . import cei_v3_paths as io_paths

REPO = io_paths.REPO
V3_ROOT = io_paths.DEFAULT_RESULTS / 'core'
P_ROOT = io_paths.DEFAULT_RESULTS / 'protgnn'
MAIN = REPO
VALIDATION_ROOT = io_paths.DEFAULT_RESULTS / 'validation'
ARTIFACT = REPO / io_paths.ARTIFACT_REL
TARGETS = REPO / io_paths.TARGETS_REL
CANONICAL = REPO / 'comparison/canonical_split.json'
OUTPUT = REPO / 'comparison/standardized/clinical_runs_cei_v3_vs_xgboost_20261001'
PATHS = None


def configure(paths, output=None):
    global PATHS, MAIN, V3_ROOT, P_ROOT, VALIDATION_ROOT, ARTIFACT, TARGETS, CANONICAL, OUTPUT
    PATHS = paths
    validation.configure(paths)
    MAIN, V3_ROOT, P_ROOT = paths.data_root, paths.v3_root, paths.protgnn_root
    VALIDATION_ROOT = paths.validation_root
    ARTIFACT, TARGETS, CANONICAL = paths.artifact, paths.targets, paths.canonical
    if output is not None:
        OUTPUT = io_paths.check_output(paths, output)


def input_roots():
    return V3_ROOT, P_ROOT, VALIDATION_ROOT
CONTRASTS = [('C', 'X'), ('A', 'X'), ('B', 'X'), ('P', 'X')]
VALIDATION_HASH = '287973edfea71e36731eddaed887928b7bf0553f7a8b684187336c789b379173'
APPROVAL_REFERENCE = ('user chat 2026-09-30: xgboost icinde aynisini yap; '
                      'one validation pass per model')
IDENTITY_KEYS = (
    'artifact_graphs_sha256', 'artifact_visit_membership_sha256',
    'artifact_visit_membership_file', 'visit_membership_contract_version',
    'targets_sha256', 'target_binding_sha256', 'kept_label_indices', 'label_order',
    'sample_seed', 'edges', 'edge_direction', 'edge_payload', 'weights',
    'token_min_count', 'min_prior_visits', 'input_contract_version',
    'preprocessing_schema_version', 'evaluation_version', 'train_limit', 'dev_limit',
)
GUARD_AUDIT = {'test_graphs_deserialized': 0, 'test_memberships_deserialized': 0,
               'test_targets_interpreted': 0, 'test_graphs_encoded': 0,
               'test_rows_skipped_as_text': 0}

read = comparison.read
equal = comparison.require_equal
write = study._write_new_json


def safe_targets(path):
    """Read only non-test target identities/labels; Top-10 remains TRAIN-ranked."""
    targets, seen, patient_splits = {}, set(), {}
    with Path(path).open() as stream:
        for row in csv.DictReader(stream):
            split = row['split']
            if split == 'test':
                continue
            if split not in ('train', 'validation'):
                raise ValueError(f'Unknown non-test target split: {split!r}')
            sid, subject = row['sample_id'], row['subject_id']
            if not sid or not subject or sid in seen:
                raise ValueError('Missing/duplicate non-test target identity')
            seen.add(sid)
            if subject in patient_splits and patient_splits[subject] != split:
                raise ValueError('Patient occurs in train and validation')
            patient_splits[subject] = split
            if row['target'] and row['target'] != '-1':
                target = int(row['target'])
                if not 0 <= target < train.NUM_CLASSES:
                    raise ValueError('Target outside canonical class order')
                targets[sid] = (target, split, subject)
    if not targets:
        raise ValueError('No labelled non-test samples')
    return targets


def raw_non_test_rows(graphs_path, membership_path, limit=None):
    """Reject ambiguous split markers; skip test BEFORE either JSON is parsed."""
    with Path(graphs_path).open() as graphs, Path(membership_path).open() as memberships:
        rows = zip_longest(graphs, memberships)
        if limit is not None:
            if type(limit) is not int or limit < 0:
                raise ValueError('Invalid raw row limit')
            rows = islice(rows, limit)
        for graph_line, membership_line in rows:
            if graph_line is None or membership_line is None:
                raise ValueError('Graph/membership line counts differ')
            splits = validation.SPLIT_RE.findall(graph_line)
            if len(splits) != 1:
                raise ValueError('Cannot establish graph split before JSON parsing; test stays closed')
            if splits[0] == 'test':
                GUARD_AUDIT['test_rows_skipped_as_text'] += 1
                continue
            if splits[0] not in ('train', 'validation'):
                raise ValueError(f'Unknown graph split: {splits[0]!r}')
            ids = validation.ID_RE.findall(graph_line)
            if len(ids) != 1:
                raise ValueError('Cannot establish non-test graph identity before parsing')
            yield ids[0], splits[0], graph_line, membership_line


def safe_iterator(graphs_path, membership_path, limit=None):
    seen = set()
    for sid, split, graph_line, membership_line in raw_non_test_rows(
            graphs_path, membership_path, limit):
        graph, membership = json.loads(graph_line), json.loads(membership_line)
        equal('parsed graph split equals safe raw split', graph['split'], split)
        equal('parsed graph identity equals safe raw identity', graph['sample_id'], sid)
        contracts.validate_visit_membership_record(graph, membership)
        if sid in seen:
            raise ValueError('Duplicate non-test graph identity')
        seen.add(sid)
        yield graph, membership


def install_io_guards():
    # Every imported alias used by tabular.main/fit_features/build/fit_preprocessing
    # and assert_screen_replay is covered. Historical source files stay unchanged.
    for module in (contracts, tensorize, tabular, screen, train):
        if hasattr(module, 'iter_graphs_with_membership'):
            module.iter_graphs_with_membership = safe_iterator
    train.load_targets = safe_targets
    tabular.load_targets = safe_targets
    screen.load_targets = safe_targets


def snapshot_inputs():
    return {str(root): {p.relative_to(root).as_posix(): sha256(p)
                       for p in sorted(root.rglob('*')) if p.is_file()}
            for root in input_roots()}


def training_argv(seed, root):
    return ['--artifact', str(ARTIFACT), '--targets', str(TARGETS),
            '--canonical', str(CANONICAL), '--out', str(root / f'X_seed{seed}'),
            '--train-limit', '10000', '--dev-limit', '5000', '--sample-seed', '1234',
            '--seed', str(seed), '--top-k-labels', '10', '--edges', 'all',
            '--edge-direction', 'forward', '--weights', 'sqrt_inverse',
            '--selection-fold', 'dev', '--final-eval', 'none']


def preflight(root):
    if root.exists():
        raise FileExistsError(f'Refusing occupied output root: {root}')
    if root.parent != OUTPUT.parent or not root.name.startswith(OUTPUT.name):
        raise ValueError('Output must be approved dated root or a fresh retry suffix')
    equal('git-ignored output', subprocess.run(
        ['git', 'check-ignore', str(root) + '/'], cwd=REPO,
        capture_output=True).returncode, 0)
    install_io_guards()
    reference = read(V3_ROOT / 'A_seed1234/binding.json')
    source = recursive_source_hashes(PACKAGE_ROOT)
    for key, digest in reference['source_code'].items():
        equal(f'unchanged shared source {key}', source.get(key), digest)
    frozen = read(V3_ROOT / 'k_selection.json')
    study._validate_k_selection_record(frozen)
    equal('K*', frozen['k_selected'], 4)
    equal('K freeze bytes', sha256(V3_ROOT / 'k_selection.json'), study.k_selection_sha256(frozen))
    for path, key in ((ARTIFACT / 'graphs.jsonl', 'artifact_graphs_sha256'),
                      (ARTIFACT / 'visit_membership.jsonl', 'artifact_visit_membership_sha256'),
                      (TARGETS, 'targets_sha256')):
        equal(f'input bytes {key}', sha256(path), reference[key])
    manifest = read(ARTIFACT / 'manifest.json')
    contracts.validate_artifact_manifest(manifest)
    equal('canonical bytes', sha256(CANONICAL), read(VALIDATION_ROOT / 'decision.json')['canonical_sha256'])
    labels = read(CANONICAL)['classes']
    equal('target binding', contracts.verify_target_binding(manifest, TARGETS, labels),
          reference['target_binding_sha256'])
    targets, kept = screen.load_screen_targets(TARGETS, top_k_labels=10)
    equal('kept labels', kept, reference['kept_label_indices'])
    equal('label order', [labels[i] for i in kept], reference['label_order'])
    train_ids = train.sample_train_ids(targets, 10000, 1234)
    dev_ids = train.select_dev_ids(targets, train_ids, 5000, 1234)
    record = read(V3_ROOT / 'screen_record.json')
    screen.assert_screen_replay(record, TARGETS, train_ids, dev_ids)
    screen_ids = screen.select_screen_ids(targets, train_ids, dev_ids,
        screen_limit=record['screen_limit'], screen_seed=record['screen_seed'])
    equal('screen count', len(screen_ids), 5000)
    equal('ordered screen hash', sample_ids_sha256(screen_ids), record['screen_sample_ids_sha256'])
    ordered = {'train': [], 'dev': [], 'validation': []}
    for sid, split, _, _ in raw_non_test_rows(ARTIFACT / 'graphs.jsonl', ARTIFACT / 'visit_membership.jsonl'):
        if sid in train_ids:
            equal('train artifact split', split, 'train')
            ordered['train'].append(sid)
        elif sid in dev_ids:
            equal('dev artifact split', split, 'train')
            ordered['dev'].append(sid)
        elif sid in targets and targets[sid][1] == 'validation':
            equal('validation artifact split', split, 'validation')
            ordered['validation'].append(sid)
    for fold in ordered:
        equal(f'{fold} ordered ids', sample_ids_sha256(ordered[fold]),
              reference['split_sample_ids_sha256'][fold])
        equal(f'{fold} row count', len(ordered[fold]), reference['counts'][fold])
    equal('validation id hash', sample_ids_sha256(ordered['validation']), VALIDATION_HASH)
    equal('validation count', len(ordered['validation']), 4254)
    for directory in (V3_ROOT, P_ROOT, VALIDATION_ROOT):
        decision = read(directory / 'decision.json')
        equal(f'{directory.name}: completed', decision['status'], 'completed')
        equal(f'{directory.name}: test closed', decision['test_evaluated'], False)
    equal('screen record freeze', study.screen_record_sha256(record),
          read(V3_ROOT / 'decision.json')['screen_record_sha256'])
    return reference, targets, record, screen_ids, ordered, source


def check_training_matrix(X, y, folds, metadata, prep, reference):
    checks = {}
    for fold, code in (('train', 0), ('dev', 2), ('validation', 1)):
        mask = folds == code
        digest = sample_ids_sha256(metadata['sample_ids'][mask].tolist())
        equal(f'X before training: {fold} identities', digest, reference['split_sample_ids_sha256'][fold])
        equal(f'X before training: {fold} count', int(mask.sum()), reference['counts'][fold])
        checks[fold] = digest
    equal('fitted preprocessing state', tensorize.preprocessing_state(prep),
          read(V3_ROOT / 'A_seed1234/preprocessing.json'))
    equal('train class counts', np.bincount(y[folds == 0], minlength=10).tolist(),
          reference['class_counts_train'])
    # The GNN binding serializes its float32 torch loss weights. XGBoost's
    # DMatrix also stores weights as float32; compare at that consumed dtype.
    equal('train class weights at consumed float32 dtype',
          train.class_weights(y[folds == 0], 'sqrt_inverse', 10).astype(np.float32).tolist(),
          reference['class_weight_values'])
    if not np.isfinite(X).all():
        raise ValueError('Nonfinite training features')
    return checks


def train_worker(seed, root):
    install_io_guards()
    reference = read(V3_ROOT / 'A_seed1234/binding.json')
    original = tabular.build
    checked = {}

    def guarded_build(artifact, targets, prep, *args, **kwargs):
        if any(entry[1] == 'test' for entry in targets.values()):
            raise ValueError('Test target reached tabular build; refusing before encoding')
        result = original(artifact, targets, prep, *args, **kwargs)
        checked.update(check_training_matrix(*result, prep, reference))
        return result

    tabular.build = guarded_build
    sys.argv = [tabular.__file__] + training_argv(seed, root)
    tabular.main()
    directory = root / f'X_seed{seed}'
    write(directory / 'io_guard.json', {**GUARD_AUDIT, 'identity_verified_before_training': True,
        'split_sample_ids_sha256': checked, 'guard_source_sha256': sha256(Path(__file__))})
    return 0


def validate_trained(directory, seed, reference):
    binding, result = read(directory / 'binding.json'), read(directory / 'result.json')
    for key in IDENTITY_KEYS:
        equal(f'{directory.name}: {key}', binding[key], reference[key])
    for fold in ('train', 'dev', 'validation'):
        equal(f'{directory.name}: {fold} hash', binding['split_sample_ids_sha256'][fold],
              reference['split_sample_ids_sha256'][fold])
        equal(f'{directory.name}: {fold} count', binding['counts'][fold], reference['counts'][fold])
    equal('training seed', binding['seed'], seed)
    equal('training completed', result['status'], 'completed')
    equal('dev selected', binding['selection_fold'], 'dev')
    equal('no final evaluation', binding['final_eval'], 'none')
    equal('validation predictions during training', result['validation_evaluations'], 0)
    equal('training test closed', result['test_evaluated'], False)
    equal('preprocessing bytes', sha256(directory / 'preprocessing.json'), reference['preprocessing_sha256'])
    equal('300 rounds', binding['rounds'], 300)
    equal('round step', binding['round_step'], 25)
    equal('max depth default', binding['max_depth'], tabular.XGB_DEFAULTS['max_depth'])
    equal('learning rate default', binding['learning_rate'], tabular.XGB_DEFAULTS['learning_rate'])
    with np.load(directory / 'dev.npz', allow_pickle=False) as saved:
        equal('reproduced dev macro-F1', train.metrics(saved['y'], saved['proba'], 10)['macro_f1'],
              result['dev_metrics']['macro_f1'])
    audit = read(directory / 'io_guard.json')
    for key in ('test_graphs_deserialized', 'test_memberships_deserialized',
                'test_targets_interpreted', 'test_graphs_encoded'):
        equal(f'guard {key}', audit[key], 0)
    equal('pretraining identity guard', audit['identity_verified_before_training'], True)
    return binding, result, {'verified': True, 'identity_fields': list(IDENTITY_KEYS),
        'split_sample_ids_sha256': binding['split_sample_ids_sha256'],
        'preprocessing_matches_v3': True, 'train_class_counts_and_weights_match_v3': True,
        'guard_audit': audit}


def train_stage(seed, root, reference):
    import os
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    start = time.monotonic()
    with (root / 'driver_logs' / f'X_seed{seed}.log').open('xb') as stream:
        subprocess.run(['/usr/bin/python3', '-m', __package__ + '.cei_v3_vs_xgboost',
                        '--execute', '--train-seed', str(seed), '--output-root', str(root)] +
                       (PATHS or io_paths.resolve()).cli_args(),
                       cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    checked = validate_trained(root / f'X_seed{seed}', seed, reference)
    print(f'X_seed{seed}: dev={checked[1]["dev_metrics"]["macro_f1"]}; '
          f'rounds={checked[0]["selected_rounds"]}; seconds={time.monotonic()-start:.3f}', flush=True)
    return checked


def load_parents(screen_ids, validation_ids):
    arms = {fold: {arm: {} for arm in ('X', 'A', 'B', 'C', 'P')}
            for fold in ('screen', 'validation')}
    scores = {fold: {arm: {} for arm in arms[fold]} for fold in arms}
    reference_rows = {}
    old_screen = {arm: read(P_ROOT / 'decision.json' if arm == 'P' else V3_ROOT / 'decision.json')
                  for arm in ('A', 'B', 'C', 'P')}
    old_validation = read(VALIDATION_ROOT / 'decision.json')
    for arm in ('C', 'A', 'B', 'P'):
        for seed in study.SEEDS:
            name = f'C_K4_seed{seed}' if arm == 'C' else f'{arm}_seed{seed}'
            stage = (P_ROOT if arm == 'P' else V3_ROOT) / name
            for fold, directory, ids, loader in (
                    ('screen', stage / 'screen', screen_ids, comparison.load_screen),
                    ('validation', VALIDATION_ROOT / name, validation_ids, validation.load_saved)):
                result, rows, prediction = loader(directory, ids, reference_rows.get(fold))
                equal(f'{name}: {fold} checkpoint', result['checkpoint_sha256'], sha256(stage / 'best.pt'))
                equal(f'{name}: {fold} binding', result['binding_sha256'], sha256(stage / 'binding.json'))
                # The original core screen schema binds arrays/files but has no
                # self-hash; later P screen and validation schemas also self-hash.
                if 'result_sha256' in result:
                    equal(f'{name}: {fold} result hash', result['result_sha256'],
                          pilot._sha_json({k: v for k, v in result.items() if k != 'result_sha256'}))
                for key in ('logits', 'proba'):
                    if f'{key}_file_sha256' in result:
                        equal(f'{name}: {fold} {key} file hash', sha256(directory / f'{key}.npz'),
                              result[f'{key}_file_sha256'])
                equal(f'{name}: {fold} stored decision score', result['macro_f1'],
                      (old_screen[arm]['screen_macro_f1'] if fold == 'screen'
                       else old_validation['validation_macro_f1'])[arm][str(seed)])
                with np.load(directory / 'proba.npz', allow_pickle=False) as saved:
                    equal(f'{name}: {fold} probability score replay',
                          study.weighted_macro_f1(saved['y'], saved['proba'].argmax(1)), result['macro_f1'])
                reference_rows.setdefault(fold, rows)
                arms[fold][arm][seed], scores[fold][arm][seed] = prediction, result['macro_f1']
    return arms, scores, reference_rows


def load_x_saved(directory, fold, expected_rows):
    result = read(directory / f'{fold}_result.json')
    equal('X result hash', result['result_sha256'],
          pilot._sha_json({k: v for k, v in result.items() if k != 'result_sha256'}))
    equal('X probability file hash', sha256(directory / 'proba.npz'), result['proba_file_sha256'])
    with np.load(directory / 'proba.npz', allow_pickle=False) as saved:
        proba = saved['proba'].copy()
        rows = tuple(saved[key].copy() for key in ('y', 'subjects', 'sample_ids'))
    for key, actual, expected in zip(('y', 'subjects', 'sample_ids'), rows, expected_rows):
        equal(f'X {fold} paired {key}', np.array_equal(actual, expected), True)
    if proba.dtype != np.float32 or proba.shape != (len(rows[0]), 10) or not np.isfinite(proba).all():
        raise ValueError('Invalid X probability dtype/shape/values')
    equal('X probability array hash', comparison.array_hash(proba), result['proba_sha256'])
    equal('X macro-F1 replay', study.weighted_macro_f1(rows[0], proba.argmax(1)), result['macro_f1'])
    equal('X single inference', result['inference_passes'], 1)
    equal('X test closed', result['test_evaluated'], False)
    return result, (rows[0], proba.argmax(1), rows[1])


def score_model(directory, binding, fold, ids, targets, expected_rows, record, approval, root):
    import xgboost as xgb
    if fold not in ('screen', 'validation'):
        raise ValueError('Only screen and validation scoring allowed; test stays closed')
    model_hash = sha256(directory / 'model.ubj')
    if fold == 'validation':
        validate_approval_record(approval)
        equal('approved validation ids', sample_ids_sha256(ids), approval['validation_sample_ids_sha256'])
        equal('approved X model', approval['checkpoints'][directory.name], model_hash)
        if model_hash not in approval['checkpoint_sha256']:
            raise ValueError('X model absent from validation approval')
    wanted_split = 'train' if fold == 'screen' else 'validation'
    if len(set(ids)) != len(ids) or any(targets[sid][1] != wanted_split for sid in ids):
        raise ValueError(f'Unreproducible {fold} ids or forbidden split requested')
    out = directory / fold
    if out.exists():
        raise FileExistsError(f'Refusing repeat scoring: {out}')
    prep = tensorize.load_preprocessing(read(directory / 'preprocessing.json'))
    equal('projection feature order', read(directory / 'projection.json')['features'],
          tabular.projection_layout(prep, 'forward'))
    X, y, _, metadata = tabular.build(ARTIFACT, {sid: targets[sid] for sid in ids}, prep,
                                    return_metadata=True, edge_mode='all', edge_direction='forward')
    lookup = {sid: i for i, sid in enumerate(metadata['sample_ids'].tolist())}
    equal(f'{fold} encoded coverage', set(lookup), set(ids))
    equal(f'{fold} encoded unique count', len(lookup), len(X))
    order = np.asarray([lookup[sid] for sid in ids], dtype=np.int64)
    X = X[order]
    rows = (y[order], metadata['subjects'][order], metadata['sample_ids'][order])
    for key, actual, expected in zip(('y', 'subjects', 'sample_ids'), rows, expected_rows):
        equal(f'X {fold} encoded paired {key}', np.array_equal(actual, expected), True)
    model = xgb.Booster()
    model.load_model(directory / 'model.ubj')
    model.set_param({'nthread': 8})
    # Exactly one prediction call on this fold; reload/save checks below use arrays.
    proba = model.predict(xgb.DMatrix(X), iteration_range=(0, binding['selected_rounds'])).astype(np.float32)
    out.mkdir(exist_ok=False)
    np.savez_compressed(out / 'proba.npz', proba=proba, y=rows[0], subjects=rows[1], sample_ids=rows[2])
    result = {'arm': 'X', 'seed': binding['seed'], 'fold': fold, 'row_count': len(ids),
        'macro_f1': study.weighted_macro_f1(rows[0], proba.argmax(1)),
        'checkpoint_sha256': model_hash, 'binding_sha256': sha256(directory / 'binding.json'),
        'preprocessing_sha256': sha256(directory / 'preprocessing.json'),
        'projection_sha256': sha256(directory / 'projection.json'),
        'selected_rounds': binding['selected_rounds'], 'inference_passes': 1,
        'sample_ids_sha256': sample_ids_sha256(ids), 'row_identity_verified': True,
        'proba_sha256': comparison.array_hash(proba), 'proba_file_sha256': sha256(out / 'proba.npz'),
        'proba_path': str(out / 'proba.npz'), 'preprocessing_refitted': False,
        'screen_record_sha256': study.screen_record_sha256(record),
        'validation_evaluated': fold == 'validation', 'test_evaluated': False}
    if fold == 'validation':
        result.update(approval_record_path=str(root / APPROVAL_FILENAME),
                      approval_record_sha256=approval_record_sha256(approval),
                      validation_sample_ids_sha256=sample_ids_sha256(ids))
    result['result_sha256'] = pilot._sha_json(result)
    write(out / f'{fold}_result.json', result)
    stored, prediction = load_x_saved(out, fold, expected_rows)
    print(f'{directory.name}: {fold}_macro_f1={stored["macro_f1"]:.12f}', flush=True)
    return stored, prediction


def execute(root):
    start = time.monotonic()
    before = snapshot_inputs()
    reference, targets, record, screen_ids, ordered, source = preflight(root)
    # Branch labels are not scientific identity: preflight above binds historical
    # scientific source, immutable input bytes, K, folds and sample hashes.
    equal('source unchanged after identity preflight',
          recursive_source_hashes(PACKAGE_ROOT), source)
    equal('clean committed tree', subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=REPO, text=True), '')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'driver_logs').mkdir(exist_ok=False)
    write(root / 'input_snapshot_before.json', before)
    try:
        arms, scores, reference_rows = load_parents(screen_ids, ordered['validation'])
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(train_stage, seed, root, reference) for seed in study.SEEDS]
            trained = [future.result() for future in futures]
        checkpoints = {f'X_seed{seed}': sha256(root / f'X_seed{seed}/model.ubj') for seed in study.SEEDS}
        approval = {'allow_validation': True, 'approval_reference': APPROVAL_REFERENCE,
                    'checkpoint_sha256': list(checkpoints.values()), 'checkpoints': checkpoints,
                    'validation_sample_ids_sha256': VALIDATION_HASH,
                    'timestamp': datetime.now(timezone.utc).isoformat()}
        write_approval_record(root / APPROVAL_FILENAME, approval)
        equal('approval bytes', sha256(root / APPROVAL_FILENAME), approval_record_sha256(approval))
        identity_checks, dev_scores, selected_rounds, result_hashes = {}, {}, {}, {}
        for seed, (binding, result, checked) in zip(study.SEEDS, trained):
            name = f'X_seed{seed}'
            identity_checks[str(seed)] = checked
            dev_scores[str(seed)] = result['dev_metrics']['macro_f1']
            selected_rounds[str(seed)] = binding['selected_rounds']
            result_hashes[name] = {}
            for fold, ids in (('screen', screen_ids), ('validation', ordered['validation'])):
                saved, prediction = score_model(root / name, binding, fold, ids, targets,
                    reference_rows[fold], record, approval, root)
                arms[fold]['X'][seed], scores[fold]['X'][seed] = prediction, saved['macro_f1']
                result_hashes[name][fold] = saved['result_sha256']
        decisions = {}
        bootstrap_hashes = {}
        for fold in ('screen', 'validation'):
            deltas = study.paired_bootstrap(arms[fold], CONTRASTS, resamples=1000, seed=2026)
            np.savez_compressed(root / f'{fold}_bootstrap.npz',
                                **{f'{a}_vs_{b}': values for (a, b), values in deltas.items()})
            bootstrap_hashes[fold] = sha256(root / f'{fold}_bootstrap.npz')
            decisions[fold] = {}
            for treatment, control in CONTRASTS:
                decision = study.decide_v3(deltas, scores[fold], treatment=treatment, control=control,
                                           resamples=1000, bootstrap_seed=2026)
                decision.update(validation_evaluated=fold == 'validation', test_evaluated=False,
                    decisive=treatment == 'C',
                    statement=(f'{treatment} benefit over X demonstrated on {fold}'
                               if decision['v3_beats_control'] else
                               f'{treatment} benefit over X not demonstrated on {fold}'))
                decisions[fold][f'{treatment}_vs_X'] = decision
        equal('scientific source unchanged', recursive_source_hashes(PACKAGE_ROOT), source)
    finally:
        after = snapshot_inputs()
        unchanged = after == before
        write(root / 'input_unchanged.json', {'verified': unchanged, 'before': before, 'after': after,
            'file_counts': {key: len(value) for key, value in before.items()}})
        equal('every file under all three input roots unchanged', unchanged, True)
    report = {'status': 'completed', 'git_revision': revision, 'output_root': str(root),
        'scope': 'cei_v3_vs_xgboost_frozen_screen_and_validation', 'k_selected': 4,
        'screen_row_count': len(screen_ids), 'validation_row_count': len(ordered['validation']),
        'screen_record_sha256': study.screen_record_sha256(record),
        'validation_sample_ids_sha256': VALIDATION_HASH, 'canonical_sha256': sha256(CANONICAL),
        'screen_macro_f1': {a: {str(s): scores['screen'][a][s] for s in study.SEEDS} for a in scores['screen']},
        'validation_macro_f1': {a: {str(s): scores['validation'][a][s] for s in study.SEEDS} for a in scores['validation']},
        'seed_means': {fold: {a: float(np.mean(list(by_seed.values()))) for a, by_seed in scores[fold].items()}
                       for fold in scores},
        'xgboost_dev_macro_f1': dev_scores, 'xgboost_selected_rounds': selected_rounds,
        'identity_checks': identity_checks, 'decisions': decisions,
        'primary': {fold: decisions[fold]['C_vs_X'] for fold in decisions},
        'bootstrap_file_sha256': bootstrap_hashes, 'xgboost_result_hashes': result_hashes,
        'approval_record_path': str(root / APPROVAL_FILENAME),
        'approval_record_sha256': approval_record_sha256(approval),
        'input_roots_unchanged': True, 'input_file_counts': {k: len(v) for k, v in before.items()},
        'screen_row_identity_verified': True, 'validation_row_identity_verified': True,
        'historical_macro_f1_reproduced': True, 'validation_evaluated': True, 'test_evaluated': False,
        'test_graphs_deserialized': 0, 'test_graphs_encoded': 0,
        'selection_from_screen_or_validation': False,
        'validation_tensors_materialized_by_existing_train_runner': True,
        'temporal_clean': reference['temporal_clean'], 'temporal_limitations': reference['temporal_limitations'],
        'total_wall_seconds': time.monotonic() - start}
    report['result_sha256'] = pilot._sha_json(report)
    write(root / 'decision.json', report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output-root', type=Path, default=None)
    parser.add_argument('--train-seed', type=int, choices=study.SEEDS, help=argparse.SUPPRESS)
    io_paths.add_arguments(parser)
    args = parser.parse_args(argv)
    if args.plan and args.execute:
        parser.error('--plan and --execute are mutually exclusive')
    configure(io_paths.resolve(args), args.output_root or REPO / 'comparison/standardized/clinical_runs_cei_v3_vs_xgboost_20261001')
    root = OUTPUT
    if args.train_seed is not None:
        if not args.execute:
            raise ValueError('Internal training worker requires --execute')
        if not (root / 'input_snapshot_before.json').is_file():
            raise ValueError('Training worker requires parent execution record')
        return train_worker(args.train_seed, root)
    if not args.execute:
        report = io_paths.metadata_plan(PATHS, root, include_validation=True)
        report.update(parallel_trainings=3, booster_threads_per_training=8,
                      rounds=300, round_step=25, selection_fold='dev',
                      screen_passes_per_model=1, validation_passes_per_model=1,
                      contrasts=CONTRASTS, resamples=1000, bootstrap_seed=2026,
                      quantile_method=study.QUANTILE_METHOD,
                      training_argv={str(s): training_argv(s, root) for s in study.SEEDS})
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    return execute(root)


if __name__ == '__main__':
    raise SystemExit(main())
