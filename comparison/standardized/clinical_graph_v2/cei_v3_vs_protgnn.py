"""Approved three-seed ProtGNN comparison on the immutable CEI-GNN v3 screen.

No flag prints a read-only plan. --execute trains concurrently, verifies identity,
scores each checkpoint once, and applies existing v3 decision rules. Validation
and test scoring are forbidden. Finished training/screen directories are reused
only after verification; occupied incomplete directories are never overwritten.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from . import cei_pilot as pilot
from . import cei_v3_screen as screen
from . import cei_v3_study as study
from . import train
from .contracts import recursive_source_hashes, sample_ids_sha256
from .methods import build_method
from .schema import sha256

from . import cei_v3_paths as io_paths

REPO = io_paths.REPO
MAIN = REPO  # Compatibility alias; configured explicitly before CLI use.
V3_ROOT = io_paths.DEFAULT_RESULTS / 'core'
PILOT_ROOT = REPO / io_paths.PILOT_REL
ARTIFACT = REPO / io_paths.ARTIFACT_REL
TARGETS = REPO / io_paths.TARGETS_REL
CANONICAL = REPO / 'comparison/canonical_split.json'
OUTPUT = REPO / 'comparison/standardized/clinical_runs_cei_v3_vs_protgnn_20261001'
PATHS = None


def configure(paths, output=None):
    global PATHS, MAIN, V3_ROOT, PILOT_ROOT, ARTIFACT, TARGETS, CANONICAL, OUTPUT
    PATHS = paths
    MAIN, V3_ROOT, PILOT_ROOT = paths.data_root, paths.v3_root, paths.pilot_root
    ARTIFACT, TARGETS, CANONICAL = paths.artifact, paths.targets, paths.canonical
    if output is not None:
        OUTPUT = io_paths.check_output(paths, output)
THREADS = 6
IDENTITY_KEYS = (
    'artifact_graphs_sha256', 'artifact_visit_membership_sha256', 'targets_sha256',
    'target_binding_sha256', 'kept_label_indices', 'label_order', 'sample_seed',
    'edges', 'edge_direction', 'token_min_count', 'min_prior_visits', 'weights',
    'class_weight_values', 'input_contract_version', 'preprocessing_schema_version',
)


def read(path):
    return pilot._load_json(path, Path(path).name)


def require_equal(label, actual, expected):
    if actual != expected:
        raise ValueError(f'{label}: expected {expected!r}; actual {actual!r}')


def identity(binding, reference, label):
    for fold in ('train', 'dev'):
        require_equal(f'{label}.split_sample_ids_sha256.{fold}',
                      binding['split_sample_ids_sha256'][fold],
                      reference['split_sample_ids_sha256'][fold])
    for key in IDENTITY_KEYS:
        require_equal(f'{label}.{key}', binding[key], reference[key])
    return {'verified': True, 'fields': list(IDENTITY_KEYS),
            'split_sample_ids_sha256': {f: binding['split_sample_ids_sha256'][f]
                                        for f in ('train', 'dev')},
            'preprocessing_sha256': binding['preprocessing_sha256'],
            'preprocessing_matches_v3': binding['preprocessing_sha256'] == reference['preprocessing_sha256']}


def stages(root):
    # Source-derived pilot argv: only seed/output change, plus explicit CPU and
    # execution approval. Preserve every other flag and method-native default.
    template = pilot._stage('protgnn_control', 'protgnn', ARTIFACT, TARGETS,
                            CANONICAL, root / 'P_seed1234', study.FULL_BUDGET, 'control')
    result = []
    for seed in study.SEEDS:
        output = root / f'P_seed{seed}'
        argv = list(template.argv)
        argv[argv.index('--seed') + 1] = str(seed)
        argv[argv.index('--output') + 1] = str(output)
        argv.extend(['--device', 'cpu'])
        result.append(replace(template, name=f'P_seed{seed}', seed=seed,
                              output=str(output), argv=argv))
    return result


def frozen_inputs(reference):
    record = read(V3_ROOT / 'screen_record.json')
    frozen = read(V3_ROOT / 'k_selection.json')
    old_decision = read(V3_ROOT / 'decision.json')
    require_equal('K*', frozen['k_selected'], 4)
    require_equal('v3 completed', old_decision['status'], 'completed')
    require_equal('v3 validation evaluated', old_decision['validation_evaluated'], False)
    require_equal('v3 test evaluated', old_decision['test_evaluated'], False)
    require_equal('frozen screen record hash', study.screen_record_sha256(record),
                  old_decision['screen_record_sha256'])
    require_equal('frozen K selection hash', sha256(V3_ROOT / 'k_selection.json'),
                  old_decision['k_selection_sha256'])
    for path, key in ((ARTIFACT / 'graphs.jsonl', 'artifact_graphs_sha256'),
                      (ARTIFACT / reference['artifact_visit_membership_file'],
                       'artifact_visit_membership_sha256'), (TARGETS, 'targets_sha256')):
        require_equal(f'input bytes: {key}', sha256(path), reference[key])
    require_equal('screen artifact', record['artifact_sha256'], reference['artifact_graphs_sha256'])
    require_equal('screen preprocessing', record['preprocessing_sha256'], reference['preprocessing_sha256'])
    targets, kept = screen.load_screen_targets(TARGETS, top_k_labels=study.TOP_K_LABELS)
    require_equal('kept labels', kept, reference['kept_label_indices'])
    train_ids = train.sample_train_ids(targets, study.FULL_BUDGET[0], study.SAMPLE_SEED)
    dev_ids = train.select_dev_ids(targets, train_ids, study.FULL_BUDGET[1], study.SAMPLE_SEED)
    screen.assert_screen_replay(record, TARGETS, train_ids, dev_ids)
    ids = screen.select_screen_ids(targets, train_ids, dev_ids,
                                   screen_limit=record['screen_limit'], screen_seed=record['screen_seed'])
    require_equal('ordered frozen screen ids', sample_ids_sha256(ids), record['screen_sample_ids_sha256'])
    with np.load(V3_ROOT / 'A_seed1234/dev.npz', allow_pickle=False) as saved:
        require_equal('v3 parent dev draw', sorted(saved['sample_ids'].astype(str).tolist()), sorted(dev_ids))
        require_equal('v3 ordered dev hash', sample_ids_sha256(saved['sample_ids'].astype(str)),
                      reference['split_sample_ids_sha256']['dev'])
    return record, frozen, old_decision, targets, ids


def array_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def load_screen(directory, ids, reference_rows=None):
    result = read(directory / 'screen_result.json')
    study.assert_absence_share_replay(result, shares_path=directory / 'shares.npz',
                                     logits_path=directory / 'logits.npz')
    with np.load(directory / 'logits.npz', allow_pickle=False) as saved:
        logits = saved['logits'].copy()
        rows = tuple(saved[key].copy() for key in ('y', 'subjects', 'sample_ids'))
    with np.load(directory / 'proba.npz', allow_pickle=False) as saved:
        proba = saved['proba'].copy()
        for key, values in zip(('y', 'subjects', 'sample_ids'), rows):
            if not np.array_equal(values, saved[key]):
                raise ValueError(f'{directory}: logits/proba {key} differ')
    if rows[2].astype(str).tolist() != ids:
        raise ValueError(f'{directory}: sample_ids differ from frozen screen order')
    if reference_rows is not None:
        for key, actual, expected in zip(('y', 'subjects', 'sample_ids'), rows, reference_rows):
            if not np.array_equal(actual, expected):
                raise ValueError(f'{directory}: {key} differs row for row from v3 C_K4')
    for key, values in (('logits', logits), ('proba', proba)):
        if values.dtype != np.float32 or values.shape != (len(ids), study.NUM_CLASSES) or not np.isfinite(values).all():
            raise ValueError(f'{directory}: invalid float32 {key} shape/values')
        require_equal(f'{directory}: {key} array hash', array_hash(values), result[f'{key}_sha256'])
    require_equal(f'{directory}: row_count', result['row_count'], len(ids))
    require_equal(f'{directory}: validation evaluated', result['validation_evaluated'], False)
    require_equal(f'{directory}: test evaluated', result['test_evaluated'], False)
    score = study.weighted_macro_f1(rows[0], logits.argmax(1))
    require_equal(f'{directory}: reproduced macro_f1', score, result['macro_f1'])
    return result, rows, (rows[0], logits.argmax(1), rows[1])


def completed_stage(stage, reference, source):
    directory = Path(stage.output)
    binding, result = read(directory / 'binding.json'), read(directory / 'result.json')
    pilot.validate_pilot_binding(binding, expected_budget=study.FULL_BUDGET, expected_seed=stage.seed)
    study.check_stage_result(directory, binding, result)
    require_equal(f'{stage.name}: method', binding['method'], 'protgnn')
    require_equal(f'{stage.name}: CPU', binding['device'], 'cpu')
    require_equal(f'{stage.name}: layers', binding['layers'], 3)
    require_equal(f'{stage.name}: dropout', binding['dropout'], 0.51)
    require_equal(f'{stage.name}: source', binding['source_code'], source)
    require_equal(f'{stage.name}: preprocessing bytes', sha256(directory / 'preprocessing.json'), binding['preprocessing_sha256'])
    require_equal(f'{stage.name}: pilot method_config', binding['method_config'], read(PILOT_ROOT / 'binding.json')['method_config'])
    pilot._validate_artifacts(directory, binding, result)
    return binding, result, identity(binding, reference, stage.name)


def train_stage(stage, reference, source, root):
    directory = Path(stage.output)
    resumed = (directory / 'result.json').is_file()
    start = time.monotonic()
    if not resumed:
        if directory.exists():
            raise FileExistsError(f'Refusing occupied incomplete training directory: {directory}')
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[key] = str(THREADS)
        log = root / 'driver_logs' / f'{stage.name}_{time.time_ns()}.log'
        with log.open('xb') as stream:
            print(f'{stage.name}: training started', flush=True)
            subprocess.run(stage.argv + ['--execute'], cwd=REPO, env=env,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
    checked = completed_stage(stage, reference, source)
    print(f'{stage.name}: completed; dev_macro_f1={checked[1]["dev_metrics"]["macro_f1"]}; '
          f'elapsed={time.monotonic() - start:.3f}; resumed={resumed}', flush=True)
    return checked


def score_protgnn(stage, binding, record, frozen, targets, ids, reference_rows):
    import torch
    from torch_geometric.loader import DataLoader

    directory = Path(stage.output)
    out = directory / 'screen'
    expected = {'arm': 'P', 'seed': stage.seed,
                'checkpoint_sha256': sha256(directory / 'best.pt'),
                'binding_sha256': sha256(directory / 'binding.json'),
                'preprocessing_sha256': binding['preprocessing_sha256'],
                'k_selection_sha256': study.k_selection_sha256(frozen),
                'screen_record_sha256': study.screen_record_sha256(record)}
    if (out / 'screen_result.json').is_file():
        result, rows, predictions = load_screen(out, ids, reference_rows)
        for key, value in expected.items():
            require_equal(f'{stage.name}: saved {key}', result[key], value)
        require_equal(f'{stage.name}: result hash', result['result_sha256'],
                      pilot._sha_json({k: v for k, v in result.items() if k != 'result_sha256'}))
        for key in ('logits', 'proba'):
            require_equal(f'{stage.name}: saved {key} file', sha256(out / f'{key}.npz'), result[f'{key}_file_sha256'])
        return result, predictions
    if out.exists():
        raise FileExistsError(f'Refusing occupied incomplete screen output: {out}')
    # ProtGNN owns its preprocessing binding; the original screen record remains
    # unchanged even if that preprocessing differs from CEI's state.
    encoded = list(screen.encode_rows(ARTIFACT, directory / 'preprocessing.json', ids,
                    fold='screen', edge_direction=binding['edge_direction'], targets=targets,
                    preprocessing_sha256=binding['preprocessing_sha256']))
    by_id = {row.sample_id: row for row in encoded}
    if len(by_id) != len(encoded) or set(by_id) != set(ids):
        raise ValueError(f'{stage.name}: encoded screen identity differs from frozen ids')
    rows = [by_id[sid] for sid in ids]
    encoded_rows = (np.asarray([int(row.y.item()) for row in rows]),
                    np.asarray([str(row.subject) for row in rows]), np.asarray(ids))
    for key, actual, expected_values in zip(('y', 'subjects', 'sample_ids'), encoded_rows, reference_rows):
        if not np.array_equal(actual, expected_values):
            raise ValueError(f'{stage.name}: encoded {key} differs row for row from v3 C_K4')
    config = binding['method_config']
    architecture = config['architecture']
    args = dict(binding)
    args.update(config['effective_settings'])
    args['num_relations'] = architecture['num_relations']
    model = build_method('protgnn', num_tokens=binding['vocabulary_size'], node_dim=binding['node_dim'],
                         edge_dim=binding['edge_dim'], num_classes=binding['num_classes'], hidden=binding['hidden'],
                         layers=binding['layers'], dropout=binding['dropout'], token_dim=architecture['token_dim'],
                         num_triples=binding['num_meta_relations'], args=SimpleNamespace(**args))
    require_equal(f'{stage.name}: rebuilt method_config', model.run_config(), config)
    model.load_state_dict(torch.load(directory / 'best.pt', map_location='cpu', weights_only=True), strict=True)
    model.eval()
    chunks = []
    with torch.no_grad():
        for batch in DataLoader(rows, batch_size=binding['batch_size'], shuffle=False):
            chunks.append(model(batch, epoch=binding['selected_dev']['epoch_index']).logits.cpu().to(torch.float32))
    logits = torch.cat(chunks).numpy().astype(np.float32, copy=False)
    proba = torch.softmax(torch.from_numpy(logits), dim=1).numpy().astype(np.float32, copy=False)
    if logits.shape != (len(ids), study.NUM_CLASSES) or not np.isfinite(logits).all() or not np.isfinite(proba).all():
        raise ValueError(f'{stage.name}: invalid screen predictions')
    y, subjects, sample_ids = encoded_rows
    macro_f1 = study.weighted_macro_f1(y, logits.argmax(1))
    out.mkdir(exist_ok=False)
    for key, values in (('logits', logits), ('proba', proba)):
        np.savez_compressed(out / f'{key}.npz', **{key: values}, y=y, subjects=subjects, sample_ids=sample_ids)
    result = {**expected, 'row_count': len(ids), 'macro_f1': macro_f1,
              'logits_sha256': array_hash(logits), 'proba_sha256': array_hash(proba),
              'logits_file_sha256': sha256(out / 'logits.npz'), 'proba_file_sha256': sha256(out / 'proba.npz'),
              'logits_path': str(out / 'logits.npz'), 'proba_path': str(out / 'proba.npz'),
              'row_identity_verified': True, 'validation_evaluated': False, 'test_evaluated': False}
    result['result_sha256'] = pilot._sha_json(result)
    study._write_new_json(out / 'screen_result.json', result)
    # Read stored arrays, rather than relying on in-memory success.
    stored, _, predictions = load_screen(out, ids, reference_rows)
    print(f'{stage.name}: screen_macro_f1={macro_f1}', flush=True)
    return stored, predictions


def snapshot_v3():
    return {p.relative_to(V3_ROOT).as_posix(): sha256(p)
            for p in sorted(V3_ROOT.rglob('*')) if p.is_file()}


def execute(root, planned):
    start = time.monotonic()
    started_at_unix = time.time()
    stage_documents = json.loads(json.dumps([asdict(s) for s in planned]))
    initial_snapshot = snapshot_v3()
    reference = read(V3_ROOT / 'A_seed1234/binding.json')
    identity(read(PILOT_ROOT / 'binding.json'), reference, 'pilot preflight')
    record, frozen, old_decision, targets, ids = frozen_inputs(reference)
    source = recursive_source_hashes(Path(__file__).parent)
    # All shared scientific source bytes must match the completed v3 revision;
    # the new driver is an additional file, not a silent historical-code repair.
    for key, value in reference['source_code'].items():
        require_equal(f'v3 shared source: {key}', source.get(key), value)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    status = subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=REPO, text=True)
    require_equal('committed source working tree', status, '')
    arms, scores = {arm: {} for arm in ('P', 'A', 'B', 'C')}, {arm: {} for arm in ('P', 'A', 'B', 'C')}
    reference_rows = None
    for arm in ('C', 'A', 'B'):
        for seed in study.SEEDS:
            stage_name = f'C_K4_seed{seed}' if arm == 'C' else f'{arm}_seed{seed}'
            directory = V3_ROOT / stage_name
            binding, result = read(directory / 'binding.json'), read(directory / 'result.json')
            study.check_stage_result(directory, binding, result)
            identity(binding, reference, stage_name)
            saved, rows, prediction = load_screen(directory / 'screen', ids, reference_rows)
            require_equal(f'{stage_name}: record hash', saved['screen_record_sha256'], study.screen_record_sha256(record))
            require_equal(f'{stage_name}: checkpoint hash', saved['checkpoint_sha256'], sha256(directory / 'best.pt'))
            require_equal(f'{stage_name}: binding hash', saved['binding_sha256'], sha256(directory / 'binding.json'))
            require_equal(f'{stage_name}: stored decision score', saved['macro_f1'], old_decision['screen_macro_f1'][arm][str(seed)])
            reference_rows = rows if reference_rows is None else reference_rows
            arms[arm][seed], scores[arm][seed] = prediction, saved['macro_f1']
    if root.exists():
        if not (root / 'journal_full.json').is_file():
            raise FileExistsError(f'Refusing existing non-resumable output root: {root}')
    else:
        root.mkdir(parents=True, exist_ok=False)
    with (root / '.driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal_path = root / 'journal_full.json'
        journal = read(journal_path) if journal_path.exists() else {
            'started_at_unix': started_at_unix, 'git_revision': revision,
            'source_code': source, 'v3_snapshot': initial_snapshot,
            'screen_record_sha256': study.screen_record_sha256(record),
            'canonical_sha256': sha256(CANONICAL), 'stages': stage_documents}
        require_equal('resumed scientific source', journal['source_code'], source)
        require_equal('resumed v3 input snapshot', journal['v3_snapshot'], initial_snapshot)
        require_equal('resumed canonical', journal['canonical_sha256'], sha256(CANONICAL))
        require_equal('resumed stage argv', journal['stages'], stage_documents)
        journal.update(status='running', validation_evaluated=False, test_evaluated=False)
        pilot._write_journal(journal_path, journal)
        (root / 'driver_logs').mkdir(exist_ok=True)
        try:
            with ThreadPoolExecutor(max_workers=len(planned)) as pool:
                futures = [pool.submit(train_stage, s, reference, source, root) for s in planned]
                trained = [future.result() for future in futures]
            identity_checks, dev_scores = {}, {}
            for stage, (binding, result, checked) in zip(planned, trained):
                identity_checks[str(stage.seed)] = checked
                dev_scores[str(stage.seed)] = result['dev_metrics']['macro_f1']
                saved, prediction = score_protgnn(stage, binding, record, frozen, targets, ids, reference_rows)
                arms['P'][stage.seed], scores['P'][stage.seed] = prediction, saved['macro_f1']
            deltas = study.paired_bootstrap(arms, [('C', 'P'), ('A', 'P')], resamples=1000, seed=2026)
            primary = study.decide_v3(deltas, scores, treatment='C', control='P')
            secondary = study.decide_v3(deltas, scores, treatment='A', control='P')
            # Replace only the stock C-vs-A wording, never the registered checks.
            primary['statement'] = ('CEI-GNN v3 C beats ProtGNN on this frozen screen'
                                    if primary['v3_beats_control'] else 'CEI-GNN v3 C benefit over ProtGNN not demonstrated on this frozen screen')
            secondary.update(decisive=False, statement='secondary CEI-GNN v3 A versus ProtGNN; not an input to the primary C versus P decision')
            require_equal('v3 output root unchanged', snapshot_v3(), journal['v3_snapshot'])
            require_equal('source unchanged after execution', recursive_source_hashes(Path(__file__).parent), source)
            pilot_dev = read(PILOT_ROOT / 'result.json')['dev_metrics']['macro_f1']
            report = {'status': 'completed', 'scope': 'v3_vs_protgnn_frozen_screen',
                      'git_revision': journal['git_revision'], 'output_root': str(root), 'k_selected': 4,
                      'screen_record_sha256': study.screen_record_sha256(record),
                      'screen_record_path': str(V3_ROOT / 'screen_record.json'),
                      'screen_row_count': len(ids), 'screen_distinct_subject_count': record['distinct_subject_count'],
                      'screen_macro_f1': {arm: {str(s): scores[arm][s] for s in study.SEEDS} for arm in scores},
                      'seed_means': {arm: float(np.mean([scores[arm][s] for s in study.SEEDS])) for arm in scores},
                      'protgnn_dev_macro_f1': dev_scores, 'identity_checks': identity_checks,
                      'reproduction_seed1234': {'pilot_dev_macro_f1': pilot_dev, 'current_dev_macro_f1': dev_scores['1234'],
                                               'delta': dev_scores['1234'] - pilot_dev, 'exact_match': dev_scores['1234'] == pilot_dev},
                      'decision': primary, 'secondary_A_vs_P': secondary,
                      'v3_output_unchanged': True, 'screen_row_identity_verified': True,
                      'validation_evaluated': False, 'test_evaluated': False,
                      'validation_tensors_materialized_by_existing_train_runner': True,
                      'temporal_clean': reference['temporal_clean'],
                      'total_wall_seconds': time.time() - journal['started_at_unix']}
            path = root / 'decision.json'
            if path.exists():
                previous = read(path)
                for key in report:
                    if key != 'total_wall_seconds':
                        require_equal(f'resumed decision {key}', previous[key], report[key])
                report = previous
            else:
                study._write_new_json(path, report)
            journal.update(status='completed', finished_at_unix=time.time(), total_wall_seconds=report['total_wall_seconds'])
            pilot._write_journal(journal_path, journal)
            print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        except BaseException as error:
            journal.update(status='failed', error=str(error), finished_at_unix=time.time())
            pilot._write_journal(journal_path, journal)
            raise
    print(f'Invocation wall_seconds={time.monotonic() - start:.3f}', flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output-root', type=Path, default=None)
    io_paths.add_arguments(parser)
    args = parser.parse_args(argv)
    if args.plan and args.execute:
        parser.error('--plan and --execute are mutually exclusive')
    configure(io_paths.resolve(args), args.output_root or REPO / 'comparison/standardized/clinical_runs_cei_v3_vs_protgnn_20261001')
    planned = stages(OUTPUT)
    if not args.execute:
        template = next(stage for stage in pilot.build_plan(artifact=ARTIFACT, targets=TARGETS,
                        canonical=CANONICAL, output_root=OUTPUT) if stage.name == 'protgnn_control')
        for stage in planned:
            normalized = list(stage.argv[:-2])
            normalized[normalized.index('--seed') + 1] = '1234'
            normalized[normalized.index('--output') + 1] = template.output
            require_equal(f'{stage.name}: exact pilot argv', normalized, template.argv)
        report = io_paths.metadata_plan(PATHS, OUTPUT)
        identity(read(PILOT_ROOT / 'binding.json'), read(V3_ROOT / 'A_seed1234/binding.json'), 'pilot preflight')
        report.update(parallel_trainings=3, threads_per_training=THREADS,
                      contrasts=[['C', 'P'], ['A', 'P']], resamples=1000,
                      bootstrap_seed=2026, quantile_method=study.QUANTILE_METHOD,
                      pilot_identity_verified=True, stages=[asdict(stage) for stage in planned])
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    return execute(OUTPUT, planned)


if __name__ == '__main__':
    raise SystemExit(main())
