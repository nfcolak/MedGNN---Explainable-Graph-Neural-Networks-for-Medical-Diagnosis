"""Thin CEI-v3 extension executor. Default is a byte-read-only metadata plan.

Smoke is sequential, train/dev only (256/128/2, seed 1234). Full is 12 frozen
runs followed by exploratory reuse of the inspected core screen. Full and O
validation fitting require separate scoped approval documents. No test path.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

from .. import cei_v3_study as core
from core.contracts import recursive_source_hashes
from . import study
from .arm_guards import EXTENSION_ARMS, arm_of_run_config, assert_extension_binding
from ..cei_v3_paths import PathMap, check_output, resolve as resolve_paths
from .io import digest, read_json

from core.paths import PACKAGE_ROOT, REPO_ROOT

MODULE = 'cei.studies.cei_v3_ext.run'
SMOKE_BUDGET = (256, 128, 2)
EXPLORATORY = 'exploratory reuse of an already inspected screen; not prospective confirmation'


def approval(path, scope):
    if not path:
        raise ValueError(f'{scope} requires a separate explicit --approval document')
    doc = read_json(path)
    if (doc.get('approved') is not True or doc.get('scope') != scope
            or not isinstance(doc.get('approval_reference'), str)
            or not doc['approval_reference'].strip()):
        raise ValueError(f'approval must explicitly authorize scope {scope}')
    if scope == 'cei_v3_offset_validation_fit':
        if (doc.get('allow_validation_fitting') is not True
                or 'validation_fitting' not in doc.get('authorized_operations', [])):
            raise ValueError('O requires validation FITTING approval, not one historical scoring pass')
    return doc


def plan(args, path_map):
    """Open only binding/selection metadata, never targets or any graph stream."""
    root, core_root = Path(args.output_root), Path(args.core_root)
    for path in (root, core_root, Path(args.artifact), Path(args.targets), Path(args.canonical),
                 Path(args.python)):
        if not path.is_absolute():
            raise ValueError('all input/output/interpreter paths must be explicit absolute paths')
    for path in (core_root, Path(args.artifact), Path(args.targets), Path(args.canonical),
                 Path(args.python)):
        if not path.exists():
            raise ValueError(f'input missing: {path}')
    paths = resolve_paths(argparse.Namespace(
        results_root=Path(args.path_map).parent if args.path_map else core_root.parent,
        v3_root=core_root, artifact=args.artifact, targets=args.targets,
        canonical=args.canonical, path_map=args.path_map))
    check_output(paths, root)
    if root == core_root or root in core_root.parents or core_root in root.parents:
        raise ValueError('extension root must be disjoint from immutable core root')
    if args.threads < 1 or args.memory_ceiling_gib <= 0 or args.stage_timeout_seconds <= 0:
        raise ValueError('thread, memory and time ceilings must be positive')
    frozen = read_json(core_root / 'k_selection.json')
    core._validate_k_selection_record(frozen)
    if digest(core_root / 'k_selection.json') != core.k_selection_sha256(frozen):
        raise ValueError('core freeze is not canonical immutable evidence')
    k = frozen['k_selected']
    control_binding = read_json(core_root / f'C_K{k}_seed1234' / 'binding.json')
    state_recorded = control_binding['method_config']['v3_state_path']
    if not path_map.resolve(state_recorded, frozen['selected_v3_state_sha256']).is_file():
        raise ValueError('frozen state path unavailable; provide explicit --path-map')
    # U5 refuses occupied A/B directories even when only extensions are wanted.
    # Use an uncreated projection solely for its exact protocol argv/order; only
    # output and state-open tokens are relocated, never binding documents.
    projection = root / '.read_only_plan_projection'
    if projection.exists():
        raise FileExistsError('occupied plan projection')
    base = core.plan(core.StudyConfig(args.artifact, args.targets, args.canonical,
                                     str(projection), k_selection=frozen))
    stages = []
    for stage in base.stages:
        argv = list(stage.argv)
        argv[0] = args.python
        old_state = stage.v3_state
        for i, token in enumerate(argv):
            if token == f'v3_state={old_state}':
                argv[i] = f'v3_state={path_map.resolve(state_recorded)}'
        stages.append(replace(stage, argv=tuple(argv),
                              v3_state=str(path_map.resolve(state_recorded))))
    base = replace(base, output_root=str(root), stages=tuple(stages),
                   k_selection_path=str(core_root / 'k_selection.json'))
    arms = study.FAMILY_ARMS if args.include_o else EXTENSION_ARMS
    locked = study.extension_plan(base, study.ExtensionConfig(
        frozen, arms=arms, output_root=str(root)), allow_existing=True)
    extension = [stage for stage in locked.stages if stage.phase == 'extension']
    full = []
    for stage in extension:
        argv = list(stage.argv)
        argv[argv.index('-m') + 1] = 'cei.studies.cei_v3_ext.launch'
        full.append(replace(stage, argv=tuple(argv) + ('--device', 'cpu', '--cpu-threads', str(args.threads))))
    smoke = []
    for arm in ('E2d', 'E6b', 'E6a', 'E2w'):
        stage = next(s for s in full if s.arm == arm and s.seed == 1234)
        out = root / 'smoke' / f'{arm}_seed1234'
        state = core.v3_state_path(root / 'smoke', k)
        argv = list(stage.argv)
        for flag, value in (('--output', str(out)), ('--train-limit', '256'),
                            ('--dev-limit', '128'), ('--epochs', '2')):
            argv[argv.index(flag) + 1] = value
        for i, token in enumerate(argv):
            if token.startswith('v3_state='):
                argv[i] = f'v3_state={state}'
        argv.append('--train-dev-only')
        smoke.append(replace(stage, name=f'smoke:{arm}_seed1234', output=str(out),
                             argv=tuple(argv), v3_state=str(state)))
    return frozen, full, smoke


def control_bindings_from_root(core_root, frozen, path_map, args):
    """Replay hashes and frozen policy using existing evidence only."""
    dirs = {entry['stage']: Path(core_root) / entry['stage'] for entry in frozen['stages']}
    core.assert_k_selection_replay(frozen, dirs)
    winners = [e for e in frozen['stages'] if e['k'] == frozen['k_selected']]
    directories = {e['seed']: dirs[e['stage']] for e in winners}
    reference = None
    for seed, directory in directories.items():
        binding = read_json(directory / 'binding.json')
        core.check_stage_result(directory, binding, read_json(directory / 'result.json'))
        cfg = binding['method_config']
        for key, expected in (('weights', 'sqrt_inverse'), ('train_limit', 10000),
                              ('dev_limit', 5000), ('epochs', 40), ('seed', seed)):
            if binding.get(key) != expected:
                raise ValueError(f'core control policy drift: {key}')
        for key, supplied in (('artifact', args.artifact), ('targets_path', args.targets)):
            if path_map.resolve(binding[key]).resolve() != Path(supplied).resolve():
                raise ValueError(f'core input path differs: {key}; explicit map required')
        if digest(path_map.resolve(cfg['v3_state_path'])) != frozen['selected_v3_state_sha256']:
            raise ValueError('frozen state bytes changed')
        if digest(directory / 'preprocessing.json') != binding['preprocessing_sha256']:
            raise ValueError('core preprocessing bytes changed')
        identity = {key: binding[key] for key in (
            'preprocessing_sha256', 'artifact_graphs_sha256', 'artifact_visit_membership_sha256',
            'targets_sha256', 'kept_label_indices', 'label_order', 'split_sample_ids_sha256')}
        if reference is not None and reference != identity:
            raise ValueError('core controls have different input/split identity')
        reference = identity
    controls = dict(k_selection=frozen, k_selection_sha256=core.k_selection_sha256(frozen),
                    control_binding_sha256=frozen['control_binding_sha256'],
                    control_checkpoint_sha256=[entry['checkpoint_sha256'] for entry in winners],
                    preprocessing_sha256=reference['preprocessing_sha256'],
                    directories=directories)
    from .arm_guards import _validated_controls
    _validated_controls(controls)
    return controls


def execution_identity(args, frozen):
    """Byte identity, not graph deserialization. Called only behind execute gates."""
    from core.contracts import (verify_graph_file, verify_visit_membership_file,
                             validate_artifact_manifest, verify_target_binding)
    artifact = Path(args.artifact)
    manifest = read_json(artifact / 'manifest.json')
    validate_artifact_manifest(manifest)
    verify_graph_file(artifact, manifest)
    verify_visit_membership_file(artifact, manifest)
    labels = read_json(args.canonical)['classes']
    target_binding = verify_target_binding(manifest, args.targets, labels)
    source = recursive_source_hashes(PACKAGE_ROOT)
    return dict(output_root=str(Path(args.output_root).resolve()),
                core_root=str(Path(args.core_root).resolve()), execute_mode=args.execute,
                artifact_graphs_sha256=manifest['graphs_sha256'],
                artifact_visit_membership_sha256=manifest['visit_membership_sha256'],
                targets_sha256=digest(args.targets), canonical_sha256=digest(args.canonical),
                target_binding_sha256=target_binding, source=source,
                k_selection_sha256=core.k_selection_sha256(frozen), python=args.python,
                threads=args.threads, device='cpu', path_map_sha256=(digest(args.path_map)
                    if args.path_map else None), memory_ceiling_gib=args.memory_ceiling_gib,
                stage_timeout_seconds=args.stage_timeout_seconds,
                approval_sha256=digest(args.approval) if args.approval else None,
                o_approval_sha256=digest(args.o_approval) if args.include_o else None)


def journal_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def file_inventory(path):
    path = Path(path)
    files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
    if not files:
        raise ValueError('completed operation has no output evidence')
    return {str(p): digest(p) for p in files}


class Journal:
    """Append-only; resume verifies completed stages, never salvages partial stages."""
    def __init__(self, path, identity, resume=False):
        self.path = Path(path)
        self.completed = {}
        if self.path.exists():
            if not resume:
                raise FileExistsError('occupied journal; use --resume for verified completed stages')
            records = journal_rows(self.path)
            if not records or records[0].get('identity') != identity:
                raise ValueError('resume identity drift')
            pending = set()
            for row in records[1:]:
                name = row.get('name')
                if row['event'] == 'running':
                    pending.add(name)
                elif row['event'] == 'completed':
                    pending.discard(name)
                    self.completed[name] = row
                elif row['event'] == 'failed':
                    raise ValueError('failed/partial stage cannot be resumed; preserve it and use new root')
            if pending:
                raise ValueError('partial occupied stages in journal; refusing resume')
            for row in self.completed.values():
                if file_inventory(row['output']) != row['files']:
                    raise ValueError('completed stage inventory changed; refusing resume')
                for path, expected in row['files'].items():
                    if not Path(path).is_file() or digest(path) != expected:
                        raise ValueError('completed stage evidence changed; refusing resume')
        else:
            if resume:
                raise ValueError('--resume journal is missing')
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('x') as stream:
                stream.write(json.dumps(dict(event='start', identity=identity)) + '\n')
        self.identity = identity

    def append(self, event, **fields):
        with self.path.open('a') as stream:
            stream.write(json.dumps(dict(event=event, time_unix=time.time(), **fields),
                                    sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def stage(self, name, output, action):
        if name in self.completed:
            return self.completed[name]
        if Path(output).exists():
            raise FileExistsError(f'occupied unjournaled/partial stage: {name}')
        self.append('running', name=name)
        start = time.perf_counter()
        try:
            measurements = action() or {}
            row = dict(name=name, output=str(output), files=file_inventory(output),
                       wall_seconds=time.perf_counter() - start, **measurements)
            self.append('completed', **row)
            self.completed[name] = row
            return row
        except BaseException as error:
            self.append('failed', name=name, error_type=type(error).__name__,
                        resource_refusal=(str(error) if str(error).startswith('resource refusal:') else None))
            raise


def train_stage(stage, args, frozen, controls, smoke=False):
    """Actual process timing and wait4 peak RSS; refuse limits while child runs."""
    if recursive_source_hashes(PACKAGE_ROOT) != args._expected_source:
        raise ValueError('source drift since execution identity capture')
    out = Path(stage.output)
    logs = Path(args.output_root) / 'logs'
    logs.mkdir(exist_ok=True)
    log = logs / (stage.name.replace(':', '_') + '.log')
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
               OPENBLAS_NUM_THREADS=str(args.threads), VECLIB_MAXIMUM_THREADS=str(args.threads),
               PYTHONDONTWRITEBYTECODE='1')
    ceiling = args.memory_ceiling_gib * 1024 ** 3
    start = time.perf_counter()
    refused = None
    sampled_peak = 0
    with log.open('xb') as stream:
        process = subprocess.Popen(list(stage.argv) + ['--execute'], env=env,
                                   cwd=REPO_ROOT, stdout=stream,
                                   stderr=subprocess.STDOUT)
        while True:
            pid, status, usage = os.wait4(process.pid, os.WNOHANG)
            if pid:
                process.returncode = os.waitstatus_to_exitcode(status)
                break
            try:
                rss = subprocess.run(['ps', '-o', 'rss=', '-p', str(process.pid)],
                                     capture_output=True, text=True, check=False).stdout.strip()
            except PermissionError:
                # Some macOS execution environments refuse spawning ps. Query
                # the same child's RSS through the native process API instead;
                # permission failures still propagate, never disable the ceiling.
                import psutil
                try:
                    rss = str(psutil.Process(process.pid).memory_info().rss // 1024)
                except psutil.NoSuchProcess:
                    rss = ''  # wait4 above will reap it on the next iteration.
            if rss:
                sampled_peak = max(sampled_peak, int(rss) * 1024)
            if sampled_peak > ceiling or time.perf_counter() - start > args.stage_timeout_seconds:
                refused = 'memory ceiling' if sampled_peak > ceiling else 'stage timeout'
                process.kill()
                _, status, usage = os.wait4(process.pid, 0)
                process.returncode = os.waitstatus_to_exitcode(status)
                break
            time.sleep(0.2)
    peak = max(sampled_peak, int(usage.ru_maxrss) * (1 if sys.platform == 'darwin' else 1024))
    if refused or peak > ceiling:
        raise RuntimeError(f'resource refusal: {refused or "peak memory ceiling"}; peak_bytes={peak}')
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, stage.argv)
    binding = read_json(out / 'binding.json')
    result = read_json(out / 'result.json')
    core.check_stage_result(out, binding, result)
    if binding['source_code'] != args._expected_source or recursive_source_hashes(PACKAGE_ROOT) != args._expected_source:
        raise ValueError('runner source binding drift')
    control_inputs = read_json(controls['directories'][stage.seed] / 'binding.json')
    for key in ('artifact_graphs_sha256', 'artifact_visit_membership_sha256', 'targets_sha256'):
        if binding.get(key) != control_inputs[key]:
            raise ValueError(f'input byte identity drift during training: {key}')
    if arm_of_run_config(binding['method_config']) != stage.arm:
        raise ValueError('trained extension arm drift')
    budget = SMOKE_BUDGET if smoke else core.FULL_BUDGET
    for key, expected in zip(('train_limit', 'dev_limit', 'epochs'), budget):
        if binding.get(key) != expected:
            raise ValueError('trained stage budget drift')
    if digest(out / 'preprocessing.json') != binding['method_config']['preprocessing_sha256']:
        raise ValueError('trained preprocessing/state mismatch')
    if digest(stage.v3_state) != binding['method_config']['v3_state_sha256']:
        raise ValueError('trained state bytes mismatch')
    if smoke:
        if set(binding['counts']) != {'train', 'dev'} or binding['counts'] != {'train': 256, 'dev': 128}:
            raise ValueError('smoke opened a forbidden fold or exceeded bounded row budget')
    else:
        successor = dict(stage=stage.name, extension_arm=stage.arm, seed=stage.seed, k=stage.k,
                         k_selected=frozen['k_selected'], k_grid=frozen['k_grid'],
                         k_selection_rule=frozen['rule'],
                         k_selection_sha256=core.k_selection_sha256(frozen),
                         control_binding_sha256=controls['control_binding_sha256'],
                         control_checkpoint_sha256=controls['control_checkpoint_sha256'],
                         binding_sha256=digest(out / 'binding.json'), v3_state=stage.v3_state)
        assert_extension_binding({**binding, **successor}, controls)
        from .. import cei_pilot as pilot
        pilot._validate_artifacts(out, binding, result)
        control = read_json(controls['directories'][stage.seed] / 'binding.json')
        for key in ('lr', 'weight_decay', 'dropout', 'batch_size', 'min_delta',
                    'token_min_count', 'min_prior_visits', 'patience', 'sample_seed'):
            if binding.get(key) != control.get(key):
                raise ValueError(f'training hyperparameter differs from frozen control: {key}')
        for key in ('artifact_graphs_sha256', 'artifact_visit_membership_sha256', 'targets_sha256',
                    'preprocessing_sha256', 'split_sample_ids_sha256', 'class_weight_values',
                    'label_order', 'kept_label_indices', 'class_counts_train'):
            if binding.get(key) != control.get(key):
                raise ValueError(f'training parity differs from control: {key}')
        core._write_new_json(out / 'study_binding.json', successor)
    history = result['history']
    epoch_seconds = [row['epoch_seconds'] for row in history] if smoke else [
        history[i]['seconds'] - history[i - 1]['seconds'] for i in range(1, len(history))]
    return dict(training_wall_seconds=time.perf_counter() - start,
                epoch_seconds=epoch_seconds, epochs_completed=len(history), peak_rss_bytes=peak,
                log_sha256=digest(log), validation_evaluated=False, test_evaluated=False)


def measured_budget(journal):
    """Scaling estimate, not a promised ETA; setup and early stopping remain uncertain."""
    estimates = {}
    for arm in EXTENSION_ARMS:
        row = journal.completed[f'smoke:{arm}_seed1234']
        values = row['epoch_seconds']
        if len(values) != 2 or min(values) <= 0:
            raise ValueError('smoke lacks two actual measured epoch timings')
        # Approximate combined train/dev workload ratio, explicitly not equal compute.
        scale = (10000 + 5000) / (256 + 128)
        per_epoch = sum(values) / len(values)
        setup = max(0, row['training_wall_seconds'] - sum(values))
        estimates[arm] = dict(measured_epoch_seconds=values, measured_setup_seconds=setup,
                              measured_peak_rss_bytes=row['peak_rss_bytes'],
                              estimated_three_seed_seconds=3 * (setup + 40 * per_epoch * scale))
    return dict(status='measured_scaling_estimate', runs=12, arms=estimates,
                estimated_full_training_seconds=sum(v['estimated_three_seed_seconds']
                                                     for v in estimates.values()),
                caveat='linear row scaling from 256/128/2; graph sizes, setup, memory, scoring and early stopping differ; not equal compute or a guaranteed ETA')


def execute(args, frozen, full, smoke, path_map):
    import torch
    torch.set_num_threads(args.threads)
    if args.execute == 'full':
        approved = approval(args.approval, 'cei_v3_extension_full')
        smoke_path = Path(args.smoke_journal or Path(args.output_root) / 'journal_smoke.jsonl')
        if not smoke_path.is_file() or approved.get('smoke_journal_sha256') != digest(smoke_path):
            raise ValueError('full approval must bind the actual completed measured smoke journal')
        rows = journal_rows(smoke_path)
        if not rows or rows[-1].get('event') != 'finished':
            raise ValueError('full requires a completed bounded smoke')
        power = subprocess.run(['pmset', '-g', 'batt'], capture_output=True, text=True, check=True)
        if 'AC Power' not in power.stdout:
            raise ValueError('long execution refused: AC power is required')
    if args.include_o:
        if args.execute != 'full':
            raise ValueError('O validation fitting is not a smoke operation')
        approval(args.o_approval, 'cei_v3_offset_validation_fit')
    controls = control_bindings_from_root(args.core_root, frozen, path_map, args)
    identity = execution_identity(args, frozen)
    args._expected_source = identity['source']
    reference = read_json(controls['directories'][1234] / 'binding.json')
    for key in ('artifact_graphs_sha256', 'artifact_visit_membership_sha256', 'targets_sha256'):
        if identity[key] != reference[key]:
            raise ValueError(f'current input bytes differ from frozen core: {key}')
    if args.execute == 'full':
        from .scoring import retained_screen
        if not args.include_o:
            retained_screen(args.core_root, frozen, controls, path_map)
        prior = rows[0]['identity']
        for key in ('source', 'artifact_graphs_sha256', 'artifact_visit_membership_sha256',
                    'targets_sha256', 'canonical_sha256', 'python', 'threads', 'device',
                    'k_selection_sha256', 'path_map_sha256'):
            if prior.get(key) != identity.get(key):
                raise ValueError(f'measured smoke identity drift: {key}')
        for row in rows:
            if row.get('event') == 'completed':
                if file_inventory(row['output']) != row['files']:
                    raise ValueError('measured smoke output inventory changed')
                for path, expected in row['files'].items():
                    if not Path(path).is_file() or digest(path) != expected:
                        raise ValueError('measured smoke evidence changed')
    root = Path(args.output_root)
    journal_path = root / f'journal_{args.execute}.jsonl'
    if root.exists() and not args.resume:
        if args.execute == 'smoke':
            raise FileExistsError('smoke requires a new output root')
        allowed = {'.extension.lock', 'journal_smoke.jsonl', 'smoke', 'logs', 'measured_budget.json'}
        if {p.name for p in root.iterdir()} - allowed:
            raise FileExistsError('full requires a fresh root or its completed bounded-smoke root')
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.extension.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal = Journal(journal_path, identity, args.resume)
        if args.execute == 'smoke':
            state = core.v3_state_path(root / 'smoke', frozen['k_selected'])
            def fit():
                from .io import smoke_fit_inputs
                core.fit_v3_state(root / 'smoke', frozen['k_selected'],
                                  reader=lambda: smoke_fit_inputs(args.artifact, args.targets))
            journal.stage('smoke:state', state, fit)
            for stage in smoke:
                journal.stage(stage.name, stage.output, lambda s=stage: train_stage(
                    s, args, frozen, controls, smoke=True))
            report = measured_budget(journal)
            journal.stage('smoke:budget', root / 'measured_budget.json',
                          lambda: core._write_new_json(root / 'measured_budget.json', report))
        else:
            for stage in full:
                journal.stage(stage.name, stage.output, lambda s=stage: train_stage(
                    s, args, frozen, controls))
            execute_screen(args, frozen, full, controls, path_map, journal)
        journal.append('finished', mode=args.execute, validation_evaluated=args.include_o,
                       test_evaluated=False)
    print(json.dumps(dict(status='completed', mode=args.execute, journal=str(journal_path),
                          measured_budget=(str(root / 'measured_budget.json')
                                           if args.execute == 'smoke' else None))))
    return 0


def execute_screen(args, frozen, stages, controls, path_map, journal):
    import numpy as np
    from core import train
    from .. import cei_v3_screen as screen
    from core.contracts import sample_ids_sha256
    from . import scoring

    root = Path(args.output_root)
    if args.include_o:
        execute_offsets(args, frozen, controls, None, path_map, journal)
    record, c_results, c_arrays = scoring.retained_screen(args.core_root, frozen, controls, path_map)
    targets, kept = screen.load_screen_targets(args.targets)
    train_ids = train.sample_train_ids(targets, 10000, 1234)
    dev_ids = train.select_dev_ids(targets, train_ids, 5000, 1234)
    screen.assert_screen_replay(record, args.targets, train_ids, dev_ids)
    ids = c_arrays[1234]['sample_ids'].astype(str).tolist()
    if ids != screen.select_screen_ids(targets, train_ids, dev_ids):
        raise ValueError('retained screen differs from frozen selector replay')
    if kept != record['kept_label_indices'] or sample_ids_sha256(ids) != record['screen_sample_ids_sha256']:
        raise ValueError('retained screen target contract differs')
    arrays = {'C': {seed: (v['y'], v['logits'].argmax(1), v['subjects'])
                    for seed, v in c_arrays.items()}}
    if args.include_o:
        execute_offsets(args, frozen, controls, c_results, path_map, journal)
    for stage in stages:
        output = root / 'screen' / stage.name
        journal.stage(f'screen:{stage.name}', output, lambda s=stage, out=output: (
            scoring.score_extension(s.output, out, record, c_arrays[s.seed], frozen,
                                    controls, args.artifact, targets, path_map), None)[1])
        result = read_json(output / 'screen_result.json')
        arrays.setdefault(stage.arm, {})[stage.seed] = study.load_screen_row(result)
    if args.include_o:
        for seed in core.SEEDS:
            result = read_json(root / 'offset_screen' / f'O_seed{seed}' / 'o_result.json')
            arrays.setdefault('O', {})[seed] = study.load_screen_row(result)
    contrasts = [(arm, 'C') for arm in study.FAMILY_ARMS if arm in arrays]
    deltas = core.paired_bootstrap(arrays, contrasts)
    # Compatible with the core metric owner's additional provenance return.
    if isinstance(deltas, tuple):
        deltas, scores, _provenance = deltas
    else:
        scores = {arm: {seed: core.weighted_macro_f1(y, pred)
                        for seed, (y, pred, _subjects) in by_seed.items()}
                  for arm, by_seed in arrays.items()}
    decision = study.decide_extension(deltas, scores, m=5)
    decision.update(interpretation=EXPLORATORY, validation_is_tuning_fold=args.include_o,
                    validation_evaluated=args.include_o, test_evaluated=False,
                    screen_record_sha256=core.screen_record_sha256(record),
                    k_selection_sha256=core.k_selection_sha256(frozen))
    journal.stage('decision', root / 'extension_decision.json',
                  lambda: core._write_new_json(root / 'extension_decision.json', decision))


def execute_offsets(args, frozen, controls, c_results, path_map, journal):
    """Ready-to-call O path, unreachable without explicit validation-fitting scope."""
    from .. import cei_v3_screen as screen
    from core.contracts import VISIT_MEMBERSHIP_FILENAME, sample_ids_sha256
    from . import validation_scoring as vs, offsets, scoring
    approved = approval(args.o_approval, 'cei_v3_offset_validation_fit')
    vs.validate_approval_record(approved)
    if approved['checkpoint_sha256'] != controls['control_checkpoint_sha256']:
        raise ValueError('O fitting approval does not bind the three frozen C checkpoints')
    root = Path(args.output_root)
    approval_path = root / 'o_fitting_approval.json'
    journal.stage('O:approval', approval_path,
                  lambda: vs.write_approval_record(approval_path, approved) and None)
    targets, _kept = screen.load_screen_targets(args.targets)
    # Membership metadata gives artifact order without graph deserialization.
    with (Path(args.artifact) / VISIT_MEMBERSHIP_FILENAME).open() as stream:
        ids = [sid for line in stream if (sid := json.loads(line)['sample_id']) in targets
               and targets[sid][1] == 'validation']
    if sample_ids_sha256(ids) != approved['validation_sample_ids_sha256']:
        raise ValueError('O approval validation row order differs')
    records = {}
    for seed, directory in controls['directories'].items():
        output = root / 'offset_fit' / f'C_seed{seed}'
        binding = read_json(directory / 'binding.json')
        def fit(seed=seed, directory=directory, output=output, binding=binding):
            output.mkdir(parents=True, exist_ok=False)
            # Immutable input links give the existing X3 fitter its bound parent.
            for name in ('binding.json', 'best.pt'):
                (output / name).symlink_to(directory / name)
            encoder = core.ScreenEncoder(tuple(ids), 'validation', lambda: scoring.encode_selected(
                args.artifact, directory / 'preprocessing.json', ids, targets,
                fold='validation', edge_direction='forward',
                preprocessing_sha256=binding['preprocessing_sha256']))
            validation = vs.score_validation(
                core.Checkpoint(str(directory), str(Path(args.core_root) / 'k_selection.json')),
                encoder, approved, output_dir=output / 'validation', targets=targets,
                model_factory=lambda b, p: scoring.mapped_model(b, p, path_map))
            record = offsets.offset_record(validation, approval_path)
            offsets.assert_offset_replay(record, validation, approval_path)
        journal.stage(f'O:fit:{seed}', output, fit)
        records[seed] = read_json(output / 'validation' / 'offset_record.json')
    if c_results is None:
        return records  # fitting is frozen before this invocation reads screen arrays
    # Existing O applicator verifies all three records before any output is made.
    mapped = {seed: dict(result, logits_path=str(path_map.resolve(result['logits_path'])))
              for seed, result in c_results.items()}
    output = root / 'offset_screen'
    def apply():
        study.offset_screen_rows(mapped, records, approval_record_sha256=digest(approval_path),
                                 output_dirs={seed: output / f'O_seed{seed}' for seed in core.SEEDS})
    journal.stage('O:screen', output, apply)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('core-root', 'artifact', 'targets', 'canonical', 'output-root', 'python'):
        result.add_argument('--' + name, required=True)
    result.add_argument('--execute', choices=('smoke', 'full'))
    result.add_argument('--approval', help='full approval JSON: approved=true, scope=cei_v3_extension_full, approval_reference, smoke_journal_sha256')
    result.add_argument('--include-o', action='store_true', help='disabled by default; requires distinct O fitting approval')
    result.add_argument('--o-approval', help='JSON scope=cei_v3_offset_validation_fit, allow_validation_fitting=true, authorized_operations contains validation_fitting; plus X3 bound approval fields')
    result.add_argument('--path-map', help='preservation_manifest.json or JSON mapping object (old prefix to new prefix); mappings old/new list also accepted')
    result.add_argument('--threads', type=int, default=2)
    result.add_argument('--device', choices=('cpu',), default='cpu')
    result.add_argument('--memory-ceiling-gib', type=float, default=8)
    result.add_argument('--stage-timeout-seconds', type=float, default=1800)
    result.add_argument('--smoke-journal', help='measured journal for full approval, default output-root/journal_smoke.jsonl')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--status', action='store_true', help='read journals only; no mutable verifiers')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.execute == 'full':
            approval(args.approval, 'cei_v3_extension_full')
        if args.include_o:
            approval(args.o_approval, 'cei_v3_offset_validation_fit')
        if args.status:
            if args.execute:
                raise ValueError('--status cannot execute')
            for mode in ('smoke', 'full'):
                path = Path(args.output_root) / f'journal_{mode}.jsonl'
                records = journal_rows(path) if path.exists() else []
                print(json.dumps(dict(mode=mode, status=records[-1]['event'] if records else 'not_started',
                                      completed=[r['name'] for r in records if r['event'] == 'completed'])))
            return 0
        path_map = PathMap(args.path_map)
        frozen, full, smoke = plan(args, path_map)
        if not args.execute:
            print(json.dumps(dict(status='not_executed', scope='cei_v3_extensions', m=5,
                                  arms=list(EXTENSION_ARMS), seeds=list(core.SEEDS),
                                  k_frozen=frozen['k_selected'], weights='sqrt_inverse',
                                  O_enabled=args.include_o, interpretation=EXPLORATORY,
                                  full_budget=core.FULL_BUDGET, smoke_budget=SMOKE_BUDGET,
                                  full_stages=[asdict(s) for s in full],
                                  smoke_stages=[asdict(s) for s in smoke],
                                  eta='not measured; execute bounded smoke after integration',
                                  graph_deserialization=False, validation_evaluated=False,
                                  test_evaluated=False), indent=2, sort_keys=True))
            return 0
        return execute(args, frozen, full, smoke, path_map)
    except (ValueError, FileExistsError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'refused: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
