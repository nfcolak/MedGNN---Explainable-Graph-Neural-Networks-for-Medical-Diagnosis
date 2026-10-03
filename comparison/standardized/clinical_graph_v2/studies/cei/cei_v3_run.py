"""CLI driver for the approved, locked CEI-GNN v3 core study.

No flag prints a read-only plan; --execute full runs plan().order. There is no
smoke, extension, validation or test scoring mode. Completed outputs are checked
and resumed, never overwritten. Protocol rules remain in cei_v3_study/screen.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import cei_pilot as pilot
from . import cei_v3_study as study
from ...core.schema import sha256



def _read(path):
    return pilot._load_json(path, Path(path).name)


def _locked_plan(config):
    """Reuse plan's rules, separating its new-directory guard from resume checks.

    plan deliberately rejects occupied A/B directories (and pre-freeze C ones).
    For a resumed run, generate the exact plan in an unoccupied, never-created
    projection, then relocate ONLY its output/state path tokens to the real root.
    No stage, order, budget, option or selection rule is reconstructed here.
    """
    root = Path(config.output_root)
    if not any(root.glob('*_seed*')):
        return study.plan(config)
    projection = root / '.locked_plan_projection'
    if projection.exists():
        raise FileExistsError(f'Refusing occupied plan projection {projection}')
    template = study.plan(replace(config, output_root=str(projection)))
    paths = {template.k_selection_path: str(root / study.K_SELECTION_FILENAME)}
    paths.update({path: str(study.v3_state_path(root, k))
                  for k, path in template.v3_state_paths.items()})
    paths.update({stage.output: str(root / stage.name) for stage in template.stages})
    tokens = dict(paths)
    tokens.update({f'v3_state={old}': f'v3_state={new}' for old, new in paths.items()})
    stages = tuple(replace(stage, output=paths[stage.output],
                           v3_state=paths.get(stage.v3_state, stage.v3_state),
                           argv=tuple(tokens.get(token, token) for token in stage.argv))
                   for stage in template.stages)
    return replace(template, output_root=str(root), stages=stages,
                   v3_state_paths={k: paths[path] for k, path in template.v3_state_paths.items()},
                   k_selection_path=paths[template.k_selection_path])


def _study_binding(stage, frozen):
    path = Path(stage.output) / study.STUDY_BINDING_FILENAME
    if not path.exists():
        return study.write_study_binding(stage.output, stage, frozen)
    document = _read(path)
    expected = {
        'stage': stage.name, 'arm': stage.arm, 'seed': stage.seed, 'k': stage.k,
        'k_grid': list(study.K_GRID), 'k_selected': frozen['k_selected'],
        'k_selection_rule': study.K_SELECTION_RULE,
        'k_selection_sha256': study.k_selection_sha256(frozen),
        'binding_sha256': sha256(Path(stage.output) / 'binding.json'),
        'v3_state': stage.v3_state, 'selection_fold': 'dev', 'final_eval': 'none',
    }
    if document != expected:
        raise ValueError(f'{stage.name}: existing study_binding.json differs from the freeze')
    return document


def _completed_stage(stage, config, frozen=None):
    directory = Path(stage.output)
    binding, result = _read(directory / 'binding.json'), _read(directory / 'result.json')
    study.check_stage_result(directory, binding, result)
    successor = _study_binding(stage, frozen) if frozen is not None else None
    study.validate_v3_binding(binding, stage, k_selection=frozen, study_binding=successor)
    if binding['artifact'] != str(Path(config.artifact).resolve()) or \
            binding['targets_path'] != str(Path(config.targets).resolve()):
        raise ValueError(f'{stage.name}: input paths differ from the approved run')
    if not (directory / 'best.pt').is_file():
        raise ValueError(f'{stage.name}: best.pt is missing')
    if sha256(directory / 'preprocessing.json') != binding['preprocessing_sha256']:
        raise ValueError(f'{stage.name}: preprocessing bytes differ from the binding')
    if sha256(stage.v3_state) != binding['method_config']['v3_state_sha256']:
        raise ValueError(f'{stage.name}: state bytes differ from the binding')
    return binding, result


class Journal:
    def __init__(self, path, config, order):
        self.path = Path(path)
        identity = {
            'artifact': str(Path(config.artifact).resolve()),
            'targets': str(Path(config.targets).resolve()),
            'canonical': str(Path(config.canonical).resolve()),
            'artifact_manifest_sha256': sha256(Path(config.artifact) / 'manifest.json'),
            'targets_sha256': sha256(config.targets), 'canonical_sha256': sha256(config.canonical),
        }
        if self.path.exists():
            self.document = _read(self.path)
            if self.document.get('inputs') != identity or self.document.get('order') != list(order):
                raise ValueError('journal input identity or locked order differs; refusing resume')
        else:
            self.document = {'phase': 'full', 'order': list(order), 'inputs': identity,
                             'started_at_unix': time.time(), 'stages': {},
                             'validation_evaluated': False, 'test_evaluated': False,
                             'extensions_executed': False}
        self.document['status'] = 'running'
        self.save()

    def save(self):
        self.document['updated_at_unix'] = time.time()
        pilot._write_journal(self.path, self.document)

    def run(self, name, action):
        old = self.document['stages'].get(name, {})
        start = time.perf_counter()
        entry = {'status': 'running', 'started_at_unix': time.time()}
        self.document['stages'][name] = entry
        self.save()
        print(f'{name}: running', flush=True)
        try:
            value, resumed = action()
        except BaseException as error:
            entry.update(status='failed', wall_seconds=time.perf_counter() - start,
                         error=str(error), finished_at_unix=time.time())
            self.document['status'] = 'failed'
            self.save()
            raise
        elapsed = time.perf_counter() - start
        entry.update(status='completed', resumed=resumed,
                     wall_seconds=old.get('wall_seconds', elapsed) if resumed else elapsed,
                     last_check_seconds=elapsed, finished_at_unix=time.time())
        self.save()
        print(f'{name}: completed; wall_seconds={entry["wall_seconds"]:.3f}; '
              f'resumed={resumed}', flush=True)
        return value


def _run_stage(stage, config, frozen):
    directory = Path(stage.output)
    resumed = (directory / 'result.json').is_file()
    if not resumed:
        if directory.exists():
            raise FileExistsError(f'Refusing occupied incomplete stage output {directory}')
        logs = Path(config.output_root) / 'driver_logs'
        logs.mkdir(exist_ok=True)
        # A failed launch's log is retained; no directory or log is overwritten.
        with (logs / f'{stage.name}_{time.time_ns()}.log').open('xb') as stream:
            subprocess.run(list(stage.argv) + ['--execute'], check=True,
                           stdout=stream, stderr=subprocess.STDOUT)
    binding, _result = _completed_stage(stage, config, frozen)
    return binding, resumed


def _screen_inputs(config, stages, frozen):
    from . import cei_v3_screen as screen
    from ...core import train
    from ...core.contracts import sample_ids_sha256

    reference = None
    for stage in stages:
        binding, _ = _completed_stage(stage, config, frozen)
        identity = {key: binding[key] for key in (
            'artifact_graphs_sha256', 'artifact_visit_membership_sha256', 'targets_sha256',
            'kept_label_indices', 'preprocessing_sha256', 'split_sample_ids_sha256',
            'sample_seed', 'min_prior_visits', 'edge_direction', 'token_min_count')}
        if reference is not None and identity != reference:
            raise ValueError('screen checkpoint input/preprocessing/split identities differ')
        reference = identity
    if reference['min_prior_visits'] != 0:
        raise ValueError('locked core plan requires the unfiltered parent train sample')
    targets, _kept = screen.load_screen_targets(config.targets, top_k_labels=study.TOP_K_LABELS)
    train_ids = train.sample_train_ids(targets, study.FULL_BUDGET[0], study.SAMPLE_SEED)
    dev_ids = train.select_dev_ids(targets, train_ids, study.FULL_BUDGET[1], study.SAMPLE_SEED)
    # The stage hashes bind artifact order; the screen record binds sorted parent sets.
    for stage in stages:
        binding = _read(Path(stage.output) / 'binding.json')
        with np.load(Path(stage.output) / 'dev.npz', allow_pickle=False) as saved:
            parent_dev = saved['sample_ids'].astype(str).tolist()
        if set(parent_dev) != set(dev_ids) or len(parent_dev) != len(dev_ids) or \
                sample_ids_sha256(parent_dev) != binding['split_sample_ids_sha256']['dev']:
            raise ValueError(f'{stage.name}: dev parent draw differs from its bound predictions')
    ids = screen.select_screen_ids(targets, train_ids, dev_ids)
    record = screen.screen_record(
        config.targets, ids, train_ids, dev_ids,
        artifact_sha256=reference['artifact_graphs_sha256'],
        preprocessing_sha256=reference['preprocessing_sha256'],
        serialization_version=screen.SERIALIZATION_VERSION)
    record_path = Path(config.output_root) / 'screen_record.json'
    if record_path.exists():
        if _read(record_path) != record:
            raise ValueError('existing screen record differs from the frozen draw/bindings')
    else:
        study._write_new_json(record_path, record)
    screen.assert_screen_replay(record, config.targets, train_ids, dev_ids)
    prep_path = Path(stages[0].output) / 'preprocessing.json'

    @lru_cache(maxsize=1)
    def rows():
        encoded = list(screen.encode_rows(
            config.artifact, prep_path, ids, fold='screen', edge_direction='forward',
            targets=targets, preprocessing_sha256=reference['preprocessing_sha256']))
        by_id = {row.sample_id: row for row in encoded}
        if len(by_id) != len(encoded) or set(by_id) != set(ids):
            raise ValueError('encoded screen identity differs from the frozen draw')
        # U4 encodes artifact order; U5 requires selector order, so align by ID.
        return tuple(by_id[sid] for sid in ids)

    return record, study.ScreenEncoder(ids=tuple(ids), fold='screen', rows=rows)


def _screen_predictions(stage, config, frozen, record, encoder):
    directory = Path(stage.output)
    binding, _ = _completed_stage(stage, config, frozen)
    study.validate_v3_binding(binding, stage, screen_record=record, k_selection=frozen,
                             study_binding=_read(directory / study.STUDY_BINDING_FILENAME))
    path = directory / study.SCREEN_DIRNAME / 'screen_result.json'
    result = study.load_screen_result(path, shares_path=path.parent / 'shares.npz',
                                      logits_path=path.parent / 'logits.npz')
    expected = {'arm': stage.arm, 'seed': stage.seed, 'k': stage.k,
                'checkpoint_sha256': sha256(directory / 'best.pt'),
                'binding_sha256': sha256(directory / 'binding.json'),
                'k_selection_sha256': study.k_selection_sha256(frozen),
                'screen_record_sha256': study.screen_record_sha256(record),
                'row_count': len(encoder.ids), 'validation_evaluated': False,
                'test_evaluated': False}
    for key, value in expected.items():
        if getattr(result, key) != value:
            raise ValueError(f'{stage.name}: screen result {key} differs from its binding')
    if Path(result.logits_path) != path.parent / 'logits.npz' or \
            Path(result.proba_path) != path.parent / 'proba.npz':
        raise ValueError(f'{stage.name}: screen array paths differ')
    with np.load(result.logits_path, allow_pickle=False) as saved:
        logits, y = saved['logits'].copy(), saved['y'].copy()
        subjects, ids = saved['subjects'].astype(str), saved['sample_ids'].astype(str)
    with np.load(result.proba_path, allow_pickle=False) as saved:
        proba = saved['proba'].copy()
        if not (np.array_equal(y, saved['y']) and np.array_equal(subjects, saved['subjects'])
                and np.array_equal(ids, saved['sample_ids'])):
            raise ValueError(f'{stage.name}: screen logits/proba rows differ')
    if tuple(ids) != encoder.ids or logits.shape != (len(ids), study.NUM_CLASSES) or \
            proba.shape != logits.shape or not np.isfinite(logits).all() or \
            not np.isfinite(proba).all():
        raise ValueError(f'{stage.name}: screen array shape/order/finite contract differs')
    for values, expected_hash in ((logits, result.logits_sha256), (proba, result.proba_sha256)):
        if hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest() != expected_hash:
            raise ValueError(f'{stage.name}: screen array hash differs')
    pred = logits.argmax(1)
    if study.weighted_macro_f1(y, pred) != result.macro_f1:
        raise ValueError(f'{stage.name}: screen score does not reproduce from its saved logits')
    return result, (y, pred, subjects)


def _screen_and_decide(config, locked, frozen, journal):
    selected = [stage for stage in locked.stages if stage.k == frozen['k_selected']]
    for stage in selected:
        _study_binding(stage, frozen)
    record, encoder = _screen_inputs(config, selected, frozen)
    scores = {arm: {} for arm in study.ARMS}
    arms = {arm: {} for arm in study.ARMS}
    reference_rows = None
    for stage in selected:
        def score(stage=stage):
            out = Path(stage.output) / study.SCREEN_DIRNAME
            resumed = (out / 'screen_result.json').is_file()
            if not resumed:
                study.score_screen(study.Checkpoint(stage.output, locked.k_selection_path),
                                   record, encoder)
            return _screen_predictions(stage, config, frozen, record, encoder), resumed
        result, predictions = journal.run(f'screen:{stage.name}', score)
        scores[stage.arm][stage.seed] = result.macro_f1
        arms[stage.arm][stage.seed] = predictions
        rows = (predictions[0], predictions[2])
        if reference_rows is not None and not all(
                np.array_equal(a, b) for a, b in zip(rows, reference_rows)):
            raise ValueError('screen prediction rows are not paired across checkpoints')
        reference_rows = rows

    def decide():
        deltas = study.paired_bootstrap(arms, [('C', 'A'), ('B', 'A')])
        decision = study.decide_v3(deltas, scores)
        secondary = study.decide_v3(deltas, scores, treatment='B', control='A')
        secondary['decisive'] = False
        secondary['statement'] = 'secondary B versus A; not an input to the core C versus A decision'
        report = {
            'status': 'completed', 'study': study.STUDY_METHOD, 'scope': 'v3_core',
            'k_selected': frozen['k_selected'],
            'k_selection_sha256': study.k_selection_sha256(frozen),
            'screen_record_sha256': study.screen_record_sha256(record),
            'screen_macro_f1': {arm: {str(seed): scores[arm][seed] for seed in study.SEEDS}
                                for arm in study.ARMS},
            'seed_means': {arm: float(np.mean(list(scores[arm].values()))) for arm in study.ARMS},
            'decision': decision, 'secondary_B_vs_A': secondary,
            'validation_evaluated': False, 'test_evaluated': False, 'extensions_executed': False,
        }
        path = Path(config.output_root) / 'decision.json'
        resumed = path.exists()
        if resumed:
            if _read(path) != report:
                raise ValueError('existing decision.json differs from the saved screen predictions')
        else:
            study._write_new_json(path, report)
        return report, resumed

    journal.run('decide_v3', decide)
    return None, False


def execute_plan(config):
    locked = _locked_plan(config)
    root = Path(locked.output_root)
    root.mkdir(parents=True, exist_ok=True)
    # Hold a process-lifetime advisory lock; a second driver cannot race resumes.
    with (root / '.driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal = Journal(root / 'journal_full.json', config, locked.order)
        grid_dirs = {stage.name: stage.output for stage in locked.stages if stage.phase == 'c_grid'}
        bindings = {}
        frozen = None

        @lru_cache(maxsize=1)
        def reader():
            return study.read_v3_fit_inputs(config.artifact, config.targets)

        for name in locked.order:
            if name.startswith('fit_v3_state:'):
                k = int(name.split(':K')[1])

                def fit(k=k):
                    from ...methods.plugin_cei_gnn_v3 import load_v3_state
                    path = study.v3_state_path(root, k)
                    resumed = path.exists()
                    if not resumed:
                        study.fit_v3_state(root, k, reader=reader)
                    if load_v3_state(path).K != k:
                        raise ValueError(f'{path}: fitted state K differs from the locked plan')
                    return path, resumed

                journal.run(name, fit)
            elif name == 'k_selection':
                def freeze():
                    selection = study.select_k(bindings)
                    path = Path(locked.k_selection_path)
                    resumed = path.exists()
                    record = _read(path) if resumed else study.write_k_selection(selection, grid_dirs, path)
                    study.assert_k_selection_replay(record, grid_dirs)
                    return record, resumed

                frozen = journal.run(name, freeze)
                locked = _locked_plan(replace(config, k_selection=frozen))
            elif name == 'screen':
                journal.run(name, lambda: _screen_and_decide(config, locked, frozen, journal))
            else:
                stage = next(stage for stage in locked.stages if stage.name == name)
                bindings[name] = journal.run(name, lambda stage=stage: _run_stage(stage, config, frozen))
        journal.document['status'] = 'completed'
        journal.document['finished_at_unix'] = time.time()
        journal.save()
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for flag in ('artifact', 'targets', 'canonical', 'output-root'):
        result.add_argument(f'--{flag}', required=True)
    result.add_argument('--execute', choices=('full',))
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    root = Path(args.output_root).expanduser()
    if not root.is_absolute():
        parser().error('--output-root must be absolute')
    config = study.StudyConfig(args.artifact, args.targets, args.canonical, str(root.resolve()))
    if not args.execute:
        from .cei_v3_screen import DEFAULT_SCREEN_LIMIT, DEFAULT_SCREEN_SEED
        locked = _locked_plan(config)
        print(json.dumps({'status': 'not_executed', 'scope': 'v3_core',
                          'validation_evaluated': False, 'test_evaluated': False,
                          'screen_limit': DEFAULT_SCREEN_LIMIT, 'screen_seed': DEFAULT_SCREEN_SEED,
                          **asdict(locked)}, indent=2, sort_keys=True))
        return 0
    return execute_plan(config)


if __name__ == '__main__':
    raise SystemExit(main())
