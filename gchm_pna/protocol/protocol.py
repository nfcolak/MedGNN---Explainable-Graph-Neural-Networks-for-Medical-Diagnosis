"""Pre-registered, matched-budget protocol: is GCHM-PNA v2 the best arm?

The frozen sample10k/max6/top10 comparison cannot answer that: one seed, the epoch
picked on the same validation fold that is reported, and native defaults nobody tuned
for this task. This runner can -- and it can also answer "no".

Stages (the default is a side-effect-free dry run; writing anything needs --execute):

  pilot   wiring on real data: every arm once, 1 epoch, 300 visits, own namespace;
          its outputs never enter selection or the report
  tune    every arm gets the SAME trial budget on a patient-disjoint dev split drawn
          from unused TRAIN patients; validation is never read (--final-eval none)
  select  per arm the trial with the best dev macro-F1 (ties -> earlier trial),
          written once to selection.json and never rewritten
  final   the selected config x 3 model seeds over one fixed training sample; each run
          reads validation exactly once, at its dev-selected checkpoint
  ablate  GCHM-PNA v2 with exactly one mechanism removed, plus the untouched v1
  report  matched-binding audit, seed means, patient-cluster bootstrap of the paired
          seed-averaged differences, the pre-registered win rule, ablation deltas

Win rule (frozen in protocol_lock.json before the first real run): see WIN_RULE.
The test fold is never loaded by any stage.

Usage (repository root):
  python3 -m gchm_pna.protocol.protocol                 # dry run
  python3 -m gchm_pna.protocol.protocol --stage pilot --execute
  python3 -m gchm_pna.protocol.protocol --execute       # all stages
  python3 -m gchm_pna.protocol.protocol --status
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from core.paths import PACKAGE_ROOT, REPO_ROOT

ROOT = REPO_ROOT
PACKAGE = 'core'
PACKAGE_DIR = PACKAGE_ROOT
ARTIFACT = 'comparison/standardized/event_inputs/clinical_graph_v3_membership_max6_20260923'
TARGETS = ('comparison/standardized/event_inputs/'
           'first_recorded_lab_all_visits_v2_targets_local_v2_max6/targets.csv')
DEFAULT_OUT = 'comparison/standardized/clinical_runs_v3_gchm_v2_protocol_20260924'
IDENTITY_BEFORE = (ROOT / 'comparison' / 'standardized' / 'gchm_v2_protocol' / 'state'
                   / 'incumbent_identity_before.json')
PROTOCOL_VERSION = 'gchm_pna_v2_matched_protocol_v1'
STAGES = ('tune', 'select', 'final', 'ablate', 'report')

SETTINGS = {
    'top_k_labels': 10, 'train_limit': 10000, 'dev_limit': 5000, 'sample_seed': 1234,
    'epochs': 40, 'tune_seed': 1234, 'final_seeds': [1234, 2345, 3456],
    'xgb_rounds': 600, 'xgb_round_step': 25, 'trials_per_arm': 6,
    'bootstrap_resamples': 2000, 'bootstrap_seed': 20260924, 'interval': 0.95,
}
SCALES = {
    'full': {'train_limit': SETTINGS['train_limit'], 'dev_limit': SETTINGS['dev_limit'],
             'epochs': SETTINGS['epochs'], 'xgb_rounds': SETTINGS['xgb_rounds']},
    'pilot': {'train_limit': 300, 'dev_limit': 200, 'epochs': 1, 'xgb_rounds': 25},
}
ARMS = ('gchm_v2', 'protgnn', 'gsat', 'graphcare', 'xgboost')
CHALLENGER = 'gchm_v2'
DIRECTIONS = ('bidirectional', 'forward')
# The same three-point pattern for every GNN -- native/default settings, one dropout
# alternative, one slower and more regularised optimiser -- crossed with both edge
# views, so no arm is handicapped by the structure GCHM-PNA v2 was designed for.
GNN_HYPER = {
    'gchm_v2': [{}, {'dropout': 0.5}, {'lr': 5e-4, 'weight_decay': 1e-3}],
    'protgnn': [{}, {'dropout': 0.3}, {'lr': 1e-3, 'weight_decay': 1e-4}],
    'gsat': [{}, {'dropout': 0.5}, {'lr': 5e-4, 'weight_decay': 1e-4}],
    'graphcare': [{}, {'dropout': 0.5}, {'lr': 5e-4, 'weight_decay': 1e-4}],
}
# A reverse edge repeats its forward edge's payload, so the tabular projection gains
# only duplicate columns from the bidirectional view. XGBoost therefore spends its
# whole budget on booster settings; its round count is selected on dev in every trial.
XGB_TRIALS = [{'edge_direction': 'forward', 'max_depth': depth, 'learning_rate': rate}
              for rate in (0.05, 0.1) for depth in (4, 6, 8)]
ABLATIONS = ('flip_edge_direction', 'additive_gate', 'sum_aggregation', 'no_hub_gate')
WIN_RULE = ('gchm_v2 is the best arm iff its mean validation macro-F1 over the final '
            'seeds is the highest of all arms AND the 95% patient-cluster bootstrap '
            'interval of the seed-averaged difference (gchm_v2 - runner-up) lies above '
            'zero. Every other outcome is reported as it is.')
LABELS = {'gchm_v2': 'GCHM-PNA v2', 'protgnn': 'ProtGNN', 'gsat': 'GSAT',
          'graphcare': 'GraphCare', 'xgboost': 'XGBoost',
          'gchm_v1_reference': 'GCHM-PNA v1 (reference)',
          'gchm_v2_flip_edge_direction': 'v2, other edge view',
          'gchm_v2_additive_gate': 'v2, additive gate',
          'gchm_v2_sum_aggregation': 'v2, sum instead of PNA',
          'gchm_v2_no_hub_gate': 'v2, no hub gate'}
# Prior seconds per run for the dry-run ETA only: the 20260924 sample10k runs; v2 and
# the bidirectional view are unmeasured and assumed 1.3x v1. Live ETAs use measurements.
PRIOR_SECONDS = {'gchm_v2': 420, 'gchm_v1': 330, 'protgnn': 180, 'gsat': 215,
                 'graphcare': 150, 'xgboost': 120}
SHARED_BINDING_KEYS = (
    'artifact_graphs_sha256', 'artifact_visit_membership_sha256', 'targets_sha256',
    'target_binding_sha256', 'split_sample_ids_sha256', 'counts', 'kept_label_indices',
    'label_order', 'num_classes', 'train_limit', 'sample_seed', 'dev_limit', 'dev_policy',
    'source_code', 'preprocessing_sha256', 'evaluation_version', 'input_contract_version',
    'selection_fold', 'final_eval', 'test_evaluated')


# ----------------------------------------------------------------------- helpers
def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def atomic_write(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def finite(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def source_hashes():
    from core.contracts import recursive_source_hashes
    return recursive_source_hashes(PACKAGE_ROOT)


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def child_env(threads):
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT) + (os.pathsep + env['PYTHONPATH']
                                     if env.get('PYTHONPATH') else '')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS'):
        env[key] = str(threads)
    env.setdefault('PYTHONWARNINGS', 'ignore')
    return env


# ------------------------------------------------------------------ the ledger
def trials(arm):
    """The tuning trials of one arm, in their fixed tie-break order."""
    if arm == 'xgboost':
        return [dict(trial) for trial in XGB_TRIALS]
    return [{'edge_direction': direction, **hyper}
            for direction in DIRECTIONS for hyper in GNN_HYPER[arm]]


def assert_matched_budget():
    counts = {arm: len(trials(arm)) for arm in ARMS}
    if set(counts.values()) != {SETTINGS['trials_per_arm']}:
        raise SystemExit(f'unmatched tuning budget: {counts}')
    return counts


def cell(name, stage, arm, config, seed, final_eval, scale='full', **extra):
    return {'name': name, 'stage': stage, 'arm': arm, 'config': dict(config),
            'seed': seed, 'final_eval': final_eval, 'scale': scale, **extra}


def pilot_waves():
    seed = SETTINGS['tune_seed']
    first = [cell(f'pilot/{arm}', 'pilot', arm, trials(arm)[0], seed,
                  'validation' if arm == CHALLENGER else 'none', 'pilot')
             for arm in ARMS if arm != 'xgboost']
    first.append(cell('pilot/gchm_v2_all_switches', 'pilot', 'gchm_v2',
                      {'edge_direction': 'forward', 'modulation': 'additive',
                       'aggregation': 'sum', 'hub_gate': False}, seed, 'none', 'pilot'))
    first.append(cell('pilot/gchm_v1_reference', 'pilot', 'gchm_v1',
                      {'edge_direction': 'forward'}, seed, 'none', 'pilot'))
    # Pilot validation reads are wiring checks of the load-best/evaluate-once and
    # --match-run paths on a 1-epoch model; they never enter selection or reporting.
    second = [cell('pilot/xgboost', 'pilot', 'xgboost', trials('xgboost')[0], seed,
                   'validation', 'pilot', match=f'pilot/{CHALLENGER}')]
    return [first, second]


def tune_cells():
    return [cell(f'tune/{arm}_t{index}', 'tune', arm, config, SETTINGS['tune_seed'], 'none')
            for arm in ARMS for index, config in enumerate(trials(arm))]


def final_waves(selection):
    gnn, xgb = [], []
    for arm in ARMS:
        config = selection['arms'][arm]['config']
        for seed in SETTINGS['final_seeds']:
            name = f'final/{arm}_seed{seed}'
            if arm == 'xgboost':
                # Fail-closed peer check in tabular_control: same sample, preprocessing,
                # dev and validation rows, seed and source tree as the challenger run.
                xgb.append(cell(name, 'final', arm, config, seed, 'validation',
                                match=f'final/{CHALLENGER}_seed{seed}'))
            else:
                gnn.append(cell(name, 'final', arm, config, seed, 'validation'))
    return [gnn, xgb]


def ablation_config(base, ablation):
    config = dict(base)
    if ablation == 'flip_edge_direction':
        config['edge_direction'] = ('forward' if base['edge_direction'] == 'bidirectional'
                                    else 'bidirectional')
    elif ablation == 'additive_gate':
        config['modulation'] = 'additive'
    elif ablation == 'sum_aggregation':
        config['aggregation'] = 'sum'
    elif ablation == 'no_hub_gate':
        config['hub_gate'] = False
    else:
        raise ValueError('unknown ablation ' + ablation)
    return config


def ablate_cells(selection):
    base = selection['arms'][CHALLENGER]['config']
    cells = [cell(f'ablate/gchm_v2_{ablation}_seed{seed}', 'ablate', 'gchm_v2',
                  ablation_config(base, ablation), seed, 'validation', ablation=ablation)
             for ablation in ABLATIONS for seed in SETTINGS['final_seeds']]
    cells += [cell(f'ablate/gchm_v1_reference_seed{seed}', 'ablate', 'gchm_v1',
                   {'edge_direction': 'forward'}, seed, 'validation', ablation='v1_reference')
              for seed in SETTINGS['final_seeds']]
    return cells


def group_label(spec):
    if spec['stage'] == 'final':
        return spec['arm']
    if spec['arm'] == 'gchm_v1':
        return 'gchm_v1_reference'
    return 'gchm_v2_' + spec['ablation']


# ------------------------------------------------------------ commands + checks
def command(root, spec):
    scale, config, arm = SCALES[spec['scale']], spec['config'], spec['arm']
    shared = ['--artifact', ARTIFACT, '--targets', TARGETS,
              '--top-k-labels', str(SETTINGS['top_k_labels']),
              '--train-limit', str(scale['train_limit']),
              '--seed', str(spec['seed']), '--sample-seed', str(SETTINGS['sample_seed']),
              '--selection-fold', 'dev', '--dev-limit', str(scale['dev_limit']),
              '--final-eval', spec['final_eval'],
              '--edge-direction', config['edge_direction']]
    out = str(root / spec['name'])
    if arm == 'xgboost':
        argv = [sys.executable, '-m', PACKAGE + '.methods.xgboost.tabular_control', '--out', out, *shared,
                '--rounds', str(scale['xgb_rounds']),
                '--round-step', str(min(SETTINGS['xgb_round_step'], scale['xgb_rounds'])),
                '--max-depth', str(config['max_depth']),
                '--learning-rate', repr(config['learning_rate'])]
        if spec.get('match'):
            argv += ['--match-run', str(root / spec['match'])]
        return argv
    argv = [sys.executable, '-m', PACKAGE + '.core.train', '--output', out, *shared,
            '--epochs', str(scale['epochs']), '--device', 'cpu']
    if arm in ('gchm_v2', 'gchm_v1'):
        argv += ['--method', 'clinical_gnn', '--conv', 'gchm_v2' if arm == 'gchm_v2' else 'gchm']
    else:
        argv += ['--method', arm]
    for key in ('dropout', 'lr', 'weight_decay'):
        if key in config:
            argv += ['--' + key.replace('_', '-'), repr(config[key])]
    for key in ('modulation', 'aggregation'):
        if key in config:
            argv += ['--' + key, config[key]]
    if config.get('hub_gate') is False:
        argv.append('--no-hub-gate')
    argv.append('--execute')
    return argv


def expected_binding(spec):
    scale, config, arm = SCALES[spec['scale']], spec['config'], spec['arm']
    expected = {'seed': spec['seed'], 'sample_seed': SETTINGS['sample_seed'],
                'train_limit': scale['train_limit'], 'dev_limit': scale['dev_limit'],
                'selection_fold': 'dev', 'final_eval': spec['final_eval'],
                'edge_direction': config['edge_direction'], 'test_evaluated': False,
                'num_classes': SETTINGS['top_k_labels']}
    if arm == 'xgboost':
        expected.update(max_depth=config['max_depth'], rounds=scale['xgb_rounds'],
                        learning_rate=config['learning_rate'])
        return expected
    expected['epochs'] = scale['epochs']
    expected['method'] = 'clinical_gnn' if arm in ('gchm_v2', 'gchm_v1') else arm
    if arm in ('gchm_v2', 'gchm_v1'):
        expected['conv'] = 'gchm_v2' if arm == 'gchm_v2' else 'gchm'
    for key in ('dropout', 'lr', 'weight_decay', 'modulation', 'aggregation'):
        if key in config:
            expected[key] = config[key]
    if arm == 'gchm_v2':
        expected['hub_gate'] = config.get('hub_gate', True)
    return expected


def validate_cell(root, spec):
    """Problems with one cell's outputs; an empty list means complete and on-spec."""
    out = root / spec['name']
    path = out / 'result.json'
    if not path.exists():
        return ['result.json missing']
    try:
        result = json.loads(path.read_text())
    except ValueError:
        return ['result.json unreadable']
    problems = []
    if result.get('status') != 'completed':
        problems.append('status is not completed')
    binding = result.get('binding') or {}
    for key, value in expected_binding(spec).items():
        if binding.get(key) != value:
            problems.append(f'binding.{key}={binding.get(key)!r}, expected {value!r}')
    if not finite((result.get('dev_metrics') or {}).get('macro_f1')):
        problems.append('dev macro_f1 missing')
    if spec['final_eval'] == 'none':
        if result.get('validation_evaluations') != 0 or (out / 'validation.npz').exists():
            problems.append('validation was read by a run that must not read it')
        return problems
    if result.get('validation_evaluations') != 1:
        problems.append('validation was not read exactly once')
    if not finite((result.get('metrics') or {}).get('macro_f1')):
        problems.append('validation macro_f1 missing')
    saved = out / 'validation.npz'
    if not saved.exists():
        problems.append('validation.npz missing')
    elif spec['arm'] != 'xgboost':
        with np.load(saved) as arrays:
            digest = hashlib.sha256(np.ascontiguousarray(arrays['proba']).tobytes()).hexdigest()
        if (binding.get('selected_validation') or {}).get('prediction_sha256') != digest:
            problems.append('validation predictions differ from their bound digest')
    return problems


# --------------------------------------------------------------- state machine
class RunnerLock:
    """One live runner per output root; a dead runner's lock is taken over."""

    def __init__(self, root):
        self.path = root / '.runner.lock'

    def __enter__(self):
        if self.path.exists():
            try:
                pid = int(self.path.read_text().strip() or 0)
            except ValueError:
                pid = 0
            if pid and pid != os.getpid() and pid_alive(pid):
                raise SystemExit(f'another protocol runner (pid {pid}) is active on this --out')
            self.path.unlink()
        handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(handle, 'w') as stream:
            stream.write(str(os.getpid()))
        return self

    def __exit__(self, *exc):
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()


class Ledger:
    """Atomic cell states plus an append-only event log, outside every result cell."""

    def __init__(self, root):
        self.root = root
        self.path = root / 'state.json'
        self.events = root / 'events.jsonl'
        self.lock = threading.Lock()
        self.state = (json.loads(self.path.read_text()) if self.path.exists() else
                      {'protocol_version': PROTOCOL_VERSION, 'cells': {}, 'preflight': None})

    def cell(self, name):
        return dict(self.state['cells'].get(name, {}))

    def update(self, name, status, **extra):
        with self.lock:
            entry = self.state['cells'].setdefault(name, {})
            entry.update(status=status, updated_at=now(), **extra)
            self._save({'cell': name, 'status': status, **extra})

    def note(self, key, value):
        with self.lock:
            self.state[key] = value
            self._save({key: value})

    def recover_interrupted(self):
        for name, entry in list(self.state['cells'].items()):
            if entry.get('status') == 'running':
                self.update(name, 'interrupted',
                            note='the previous runner exited while this cell was running')

    def _save(self, event):
        atomic_write(self.path, self.state)
        with self.events.open('a') as stream:
            stream.write(json.dumps({'time': now(), **event}, sort_keys=True) + '\n')


def archive(root, spec):
    """Move an incomplete or invalid output aside; completed work is never overwritten."""
    source = root / spec['name']
    target = root / 'invalid' / (spec['name'].replace('/', '__') + '_'
                                 + datetime.now().strftime('%Y%m%dT%H%M%S'))
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, target)
    return target


def run_cell(root, ledger, spec, args):
    name, out = spec['name'], root / spec['name']
    entry = ledger.cell(name)
    if out.exists() and not validate_cell(root, spec):
        if entry.get('status') != 'completed':
            ledger.update(name, 'completed', note='validated existing output')
        return 'skipped'
    if entry.get('status') == 'failed' and not args.retry_failed:
        print(f'  skip {name}: failed earlier ({entry.get("problems", ["?"])[0]}); '
              'use --retry-failed', flush=True)
        return 'failed'
    if spec.get('match') and validate_cell(root, match_spec(root, spec)):
        ledger.update(name, 'failed', problems=['peer run is not complete: ' + spec['match']])
        return 'failed'
    if out.exists():
        ledger.update(name, entry.get('status', 'interrupted'),
                      archived_to=str(archive(root, spec)))
    argv = command(root, spec)
    log = root / 'logs' / (name.replace('/', '__') + '.log')
    started = time.monotonic()
    ledger.update(name, 'running', argv=argv, log=str(log), started_at=now())
    with log.open('a') as stream:
        stream.write(f'=== {now()} {shlex.join(argv)}\n')
        stream.flush()
        code = subprocess.run(argv, cwd=ROOT, env=child_env(args.threads), stdout=stream,
                              stderr=subprocess.STDOUT).returncode
    duration = round(time.monotonic() - started, 1)
    problems = validate_cell(root, spec) if code == 0 else [f'exit code {code}']
    if not problems:
        ledger.update(name, 'completed', returncode=code, duration_s=duration)
        return 'completed'
    status = 'interrupted' if code < 0 or code == 130 else 'failed'
    ledger.update(name, status, returncode=code, duration_s=duration, problems=problems)
    return status


def match_spec(root, spec):
    """The completed peer an XGBoost cell is matched against (same stage and seed)."""
    peer_name = spec['match']
    arm, seed = CHALLENGER, spec['seed']
    if spec['stage'] == 'pilot':
        waves = pilot_waves()
        return next(c for wave in waves for c in wave if c['name'] == peer_name)
    selection = json.loads((root / 'selection.json').read_text())
    return cell(peer_name, 'final', arm, selection['arms'][arm]['config'], seed, 'validation')


def run_waves(root, ledger, waves, args):
    """Run each wave's cells in parallel; waves run in order. True if all completed."""
    cells = [spec for wave in waves for spec in wave]
    total, finished, measured = len(cells), [0], []
    progress_lock = threading.Lock()

    def work(spec):
        try:
            outcome = run_cell(root, ledger, spec, args)
        except Exception as error:  # a runner bug must fail the cell, not the matrix
            ledger.update(spec['name'], 'failed', problems=[f'runner error: {error!r}'])
            outcome = 'failed'
        with progress_lock:
            finished[0] += 1
            duration = ledger.cell(spec['name']).get('duration_s')
            if outcome in ('completed', 'failed', 'interrupted') and duration:
                measured.append(float(duration))
            remaining = total - finished[0]
            eta = (f'~{math.ceil(np.mean(measured) * remaining / args.jobs / 60)} min left'
                   if measured and remaining else '')
            print(f'[{finished[0]}/{total}] {spec["name"]}: {outcome} {eta}', flush=True)
        return outcome

    outcomes = []
    for wave in waves:
        if not wave:
            continue
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            outcomes.extend(pool.map(work, wave))
    return all(outcome in ('completed', 'skipped') for outcome in outcomes)


# ------------------------------------------------------------ gates + selection
def lock_payload():
    return {'protocol_version': PROTOCOL_VERSION, 'artifact': ARTIFACT, 'targets': TARGETS,
            'settings': SETTINGS, 'arms': list(ARMS), 'challenger': CHALLENGER,
            'trials': {arm: trials(arm) for arm in ARMS}, 'ablations': list(ABLATIONS),
            'win_rule': WIN_RULE, 'source_code': source_hashes()}


def ensure_lock(root):
    """Freeze protocol and source tree at the first real run; refuse any later drift."""
    path = root / 'protocol_lock.json'
    payload = lock_payload()
    if path.exists():
        locked = json.loads(path.read_text())
        drift = sorted(key for key in payload if locked.get(key) != payload[key])
        if drift:
            raise SystemExit('protocol_lock.json differs in ' + ', '.join(drift) + '. The '
                             'protocol and source tree are frozen; use a fresh --out.')
        return locked
    atomic_write(path, {**payload, 'locked_at': now()})
    return payload


def preflight(root, ledger, args):
    """Untrained mechanism checks and the incumbent-identity proof, once per source tree."""
    digest = sha256_json(source_hashes())
    record = ledger.state.get('preflight') or {}
    if record.get('passed') and record.get('source_code_sha256') == digest:
        return
    checks = [('mechanism_check', [sys.executable, '-m', PACKAGE + '.core.mechanism_check']),
              ('incumbent_identity',
               [sys.executable, '-m', PACKAGE + '.methods.gchm_pna.protocol.identity_snapshot',
                '--compare', str(IDENTITY_BEFORE)])]
    for name, argv in checks:
        log = root / 'logs' / f'preflight__{name}.log'
        print(f'preflight: {name} ...', flush=True)
        with log.open('a') as stream:
            stream.write(f'=== {now()} {shlex.join(argv)}\n')
            stream.flush()
            code = subprocess.run(argv, cwd=ROOT, env=child_env(args.threads), stdout=stream,
                                  stderr=subprocess.STDOUT).returncode
        if code != 0:
            ledger.note('preflight', {'passed': False, 'failed_check': name, 'at': now()})
            raise SystemExit(f'preflight {name} failed (exit {code}); see {log}')
    ledger.note('preflight', {'passed': True, 'source_code_sha256': digest, 'at': now()})


def select(root, ledger):
    path = root / 'selection.json'
    if path.exists():
        return json.loads(path.read_text())
    specs = {spec['name']: spec for spec in tune_cells()}
    rows, missing = defaultdict(list), []
    for name, spec in specs.items():
        problems = validate_cell(root, spec)
        if problems:
            missing.append(f'{name}: {problems[0]}')
            continue
        result = json.loads((root / name / 'result.json').read_text())
        rows[spec['arm']].append({'trial': int(name.rsplit('_t', 1)[1]),
                                  'config': spec['config'],
                                  'dev_macro_f1': result['dev_metrics']['macro_f1'],
                                  'selected_epoch': result.get('selected_epoch'),
                                  'selected_rounds': result.get('selected_rounds')})
    if missing:
        raise SystemExit('selection needs every tuning trial:\n  ' + '\n  '.join(missing))
    selection = {'protocol_version': PROTOCOL_VERSION, 'selected_at': now(),
                 'rule': 'highest dev macro_f1; ties -> earlier trial', 'arms': {}}
    for arm in ARMS:
        ordered = sorted(rows[arm], key=lambda row: row['trial'])
        best = max(ordered, key=lambda row: (row['dev_macro_f1'], -row['trial']))
        selection['arms'][arm] = {'selected_trial': best['trial'], 'config': best['config'],
                                  'dev_macro_f1': best['dev_macro_f1'], 'trials': ordered}
    atomic_write(path, selection)
    ledger.note('selection_sha256', sha256_file(path))
    return selection


def load_selection(root, ledger):
    path = root / 'selection.json'
    if not path.exists():
        raise SystemExit('selection.json missing; run --stage select first')
    recorded = ledger.state.get('selection_sha256')
    if recorded and recorded != sha256_file(path):
        raise SystemExit('selection.json changed after it was frozen')
    return json.loads(path.read_text())


# -------------------------------------------------------------------- the report
def load_run(root, spec):
    out = root / spec['name']
    result = json.loads((out / 'result.json').read_text())
    with np.load(out / 'validation.npz') as saved:
        arrays = {key: np.asarray(saved[key]) for key in ('proba', 'y', 'subjects', 'sample_ids')}
    arrays['subjects'] = arrays['subjects'].astype(str)
    arrays['sample_ids'] = arrays['sample_ids'].astype(str)
    return {'spec': spec, 'result': result, 'binding': result['binding'], **arrays}


def binding_audit(runs, locked_sources):
    names = sorted(runs)
    reference = runs[names[0]]
    problems = []
    if reference['binding'].get('source_code') != locked_sources:
        problems.append('source tree of the runs differs from protocol_lock.json')
    for name in names:
        run = runs[name]
        for key in SHARED_BINDING_KEYS:
            if run['binding'].get(key) != reference['binding'].get(key):
                problems.append(f'{name}: {key} differs from {names[0]}')
        for key in ('sample_ids', 'y', 'subjects'):
            if not np.array_equal(run[key], reference[key]):
                problems.append(f'{name}: validation {key} differ from {names[0]}')
        if run['binding'].get('test_evaluated') is not False:
            problems.append(f'{name}: the test fold was evaluated')
    if problems:
        raise SystemExit('matched-binding audit failed:\n  ' + '\n  '.join(problems))
    return {'runs': len(names), 'reference': names[0], 'shared_keys': list(SHARED_BINDING_KEYS),
            'validation_reads_per_run': 1, 'test_evaluated': False}


def macro_f1_curves(weights, y, pred, num_classes):
    """Macro-F1 under each row of visit weights (sklearn semantics, zero_division=0)."""
    eye = np.eye(num_classes)
    truth, predicted = eye[y], eye[pred]
    true_positive = weights @ (truth * predicted)
    support = weights @ truth + weights @ predicted
    f1 = np.divide(2.0 * true_positive, support, out=np.zeros_like(true_positive),
                   where=support > 0)
    return f1.mean(axis=1)


def patient_weights(subjects, resamples, seed):
    patients, index = np.unique(subjects, return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(patients), np.full(len(patients), 1.0 / len(patients)),
                            size=resamples)
    return draws[:, index].astype(np.float64), len(patients)


def interval(values):
    tail = (1.0 - SETTINGS['interval']) / 2.0 * 100.0
    return [float(np.percentile(values, tail)), float(np.percentile(values, 100.0 - tail))]


def build_report(root, ledger, selection):
    finals = [spec for wave in final_waves(selection) for spec in wave]
    missing = [f"{spec['name']}: {problems[0]}" for spec in finals
               if (problems := validate_cell(root, spec))]
    if missing:
        raise SystemExit('the report needs every final run:\n  ' + '\n  '.join(missing))
    ablations = ablate_cells(selection)
    ablations_complete = not any(validate_cell(root, spec) for spec in ablations)
    specs = finals + (ablations if ablations_complete else [])
    runs = {spec['name']: load_run(root, spec) for spec in specs}
    locked = json.loads((root / 'protocol_lock.json').read_text())
    audit = binding_audit(runs, locked['source_code'])

    num_classes = SETTINGS['top_k_labels']
    first = runs[sorted(runs)[0]]
    weights, patients = patient_weights(first['subjects'], SETTINGS['bootstrap_resamples'],
                                        SETTINGS['bootstrap_seed'])
    ones = np.ones((1, len(first['y'])))
    groups = defaultdict(list)
    for name, run in runs.items():
        pred = run['proba'].argmax(1)
        point = float(macro_f1_curves(ones, run['y'], pred, num_classes)[0])
        recorded = run['result']['metrics']['macro_f1']
        if abs(point - recorded) > 1e-6:
            raise SystemExit(f'{name}: recomputed macro-F1 {point} != recorded {recorded}')
        run['point'] = point
        run['curve'] = macro_f1_curves(weights, run['y'], pred, num_classes)
        groups[group_label(run['spec'])].append(name)

    arms = {}
    for label, names in groups.items():
        names = sorted(names, key=lambda n: runs[n]['spec']['seed'])
        points = [runs[n]['point'] for n in names]
        curve = np.mean([runs[n]['curve'] for n in names], axis=0)
        metrics = [runs[n]['result']['metrics'] for n in names]
        binding = runs[names[0]]['binding']
        arms[label] = {
            'label': LABELS.get(label, label), 'runs': names,
            'seeds': [runs[n]['spec']['seed'] for n in names],
            'macro_f1_per_seed': points, 'mean_macro_f1': float(np.mean(points)),
            'seed_sd': float(np.std(points, ddof=1)) if len(points) > 1 else None,
            'macro_f1_interval': interval(curve),
            'mean_accuracy': float(np.mean([m['accuracy'] for m in metrics])),
            'mean_balanced_acc': float(np.mean([m['balanced_acc'] for m in metrics])),
            'mean_top3_acc': float(np.mean([m['top3_acc'] for m in metrics])),
            'mean_patient_equal_macro_f1': float(np.mean(
                [runs[n]['result']['patient_equal']['macro_f1'] for n in names])),
            'parameter_count': binding.get('parameter_count'),
            'config': runs[names[0]]['spec']['config'],
            'selected_epochs_or_rounds': [runs[n]['result'].get('selected_rounds')
                                          or runs[n]['result'].get('selected_epoch')
                                          for n in names],
            'curve': curve,
        }

    ranking = sorted(ARMS, key=lambda arm: arms[arm]['mean_macro_f1'], reverse=True)
    challenger = arms[CHALLENGER]
    comparisons = {}
    for label in arms:
        if label == CHALLENGER:
            continue
        difference = challenger['curve'] - arms[label]['curve']
        comparisons[label] = {
            'delta_mean': challenger['mean_macro_f1'] - arms[label]['mean_macro_f1'],
            'interval': interval(difference),
            'bootstrap_share_not_positive': float((difference <= 0).mean())}
    leader = ranking[0]
    if leader != CHALLENGER:
        verdict = {'label': 'not best', 'versus': leader, 'sentence': (
            f"{LABELS[leader]} leads ({arms[leader]['mean_macro_f1']:.4f} vs "
            f"{challenger['mean_macro_f1']:.4f}); v2 - leader "
            f"{comparisons[leader]['delta_mean']:+.4f} "
            f"[{comparisons[leader]['interval'][0]:+.4f}, "
            f"{comparisons[leader]['interval'][1]:+.4f}].")}
    else:
        runner_up = ranking[1]
        low, high = comparisons[runner_up]['interval']
        won = low > 0
        verdict = {'label': 'best' if won else 'tied at the top', 'versus': runner_up,
                   'sentence': (
                       f"GCHM-PNA v2 has the highest mean ({challenger['mean_macro_f1']:.4f}); "
                       f"v2 - {LABELS[runner_up]} "
                       f"{comparisons[runner_up]['delta_mean']:+.4f} [{low:+.4f}, {high:+.4f}]"
                       + ('.' if won else '; the interval includes zero, so this is not '
                          'a proven win.'))}
    report = {
        'protocol_version': PROTOCOL_VERSION, 'generated_at': now(), 'win_rule': WIN_RULE,
        'verdict': verdict, 'ranking': ranking, 'comparisons_v2_minus_arm': comparisons,
        'arms': {label: {k: v for k, v in entry.items() if k != 'curve'}
                 for label, entry in arms.items()},
        'ablations_included': ablations_complete,
        'validation': {'visits': int(len(first['y'])), 'patients': int(patients)},
        'bootstrap': {'unit': 'patient', 'resamples': SETTINGS['bootstrap_resamples'],
                      'seed': SETTINGS['bootstrap_seed'], 'interval': SETTINGS['interval'],
                      'statistic': 'difference of seed-averaged macro-F1, same resample'},
        'audit': audit,
        'selection_sha256': sha256_file(root / 'selection.json'),
        'protocol_lock_sha256': sha256_file(root / 'protocol_lock.json'),
        'caveats': [
            'One training sample (sample_seed 1234, 10,000 visits); seeds vary '
            'initialisation, batch order and booster sampling only.',
            'GCHM-PNA v2 default regularisation was chosen after seeing v1\'s historical '
            'validation curve; every selection inside this protocol used the dev split.',
            'Validation is read once per run and never used for selection, but it is not '
            'the held-out test fold, which stays closed until separately authorised.',
            'Top-10 labels only; not comparable with 30-class runs.',
            'XGBoost is tuned on the forward view: reverse edges add only duplicate '
            'columns to its projection.',
        ],
    }
    atomic_write(root / 'report.json', report)
    (root / 'report.md').write_text(render_markdown(report))
    ledger.note('report_sha256', sha256_file(root / 'report.json'))
    return report


def render_markdown(report):
    arms, comparisons = report['arms'], report['comparisons_v2_minus_arm']

    def row(rank, label):
        entry = arms[label]
        delta = comparisons.get(label)
        delta_text = ('—' if delta is None else
                      f"{delta['delta_mean']:+.4f} [{delta['interval'][0]:+.4f}, "
                      f"{delta['interval'][1]:+.4f}]")
        sd = '—' if entry['seed_sd'] is None else f"{entry['seed_sd']:.4f}"
        params = entry['parameter_count'] if entry['parameter_count'] is not None else '—'
        low, high = entry['macro_f1_interval']
        return (f"| {rank} | {entry['label']} | {entry['mean_macro_f1']:.4f} | {sd} | "
                f"{low:.4f}–{high:.4f} | {delta_text} | {params} | "
                f"`{json.dumps(entry['config'], sort_keys=True)}` |")

    lines = ['# GCHM-PNA v2 — matched protocol report', '',
             f"**Verdict: {report['verdict']['label']}.** {report['verdict']['sentence']}", '',
             f"Validation: {report['validation']['visits']} visits, "
             f"{report['validation']['patients']} patients; "
             f"{len(SETTINGS['final_seeds'])} seeds per arm; validation read once per run; "
             'test fold never loaded.', '', f"Win rule: {report['win_rule']}", '',
             '| Rank | Arm | Mean macro-F1 | Seed SD | 95% CI | v2 − arm [95% CI] | Params | '
             'Selected config |', '|---|---|---|---|---|---|---|---|']
    lines += [row(rank, label) for rank, label in enumerate(report['ranking'], 1)]
    extra = [label for label in arms if label not in report['ranking']]
    if extra:
        lines += ['', '## Ablations and reference (v2 − variant)', '',
                  '| Variant | Mean macro-F1 | Seed SD | 95% CI | v2 − variant [95% CI] | '
                  'Params | Config |', '|---|---|---|---|---|---|---|']
        lines += [row('', label).replace('|  |', '|', 1) for label in sorted(extra)]
    lines += ['', '## Caveats', ''] + [f'- {caveat}' for caveat in report['caveats']]
    lines += ['', f"Audit: {report['audit']['runs']} runs share every binding in "
              f"`SHARED_BINDING_KEYS`; selection sha256 `{report['selection_sha256'][:12]}`; "
              f"lock sha256 `{report['protocol_lock_sha256'][:12]}`.", '']
    return '\n'.join(lines)


# ------------------------------------------------------------------ entrypoints
def planned(stage, root):
    if stage == 'pilot':
        return [spec for wave in pilot_waves() for spec in wave]
    if stage == 'tune':
        return tune_cells()
    selection_path = root / 'selection.json'
    if stage in ('final', 'ablate') and selection_path.exists():
        selection = json.loads(selection_path.read_text())
        return ([spec for wave in final_waves(selection) for spec in wave]
                if stage == 'final' else ablate_cells(selection))
    return []


def dry_run(root, stage):
    budget = assert_matched_budget()
    stages = ('pilot',) if stage == 'pilot' else (STAGES if stage == 'all' else (stage,))
    print(f'DRY RUN (nothing written). root={root}')
    print(f'tuning budget per arm: {budget}')
    prior = 0.0
    for name in stages:
        cells = planned(name, root)
        if name in ('final', 'ablate') and not cells:
            count = (len(ARMS) * len(SETTINGS['final_seeds']) if name == 'final'
                     else (len(ABLATIONS) + 1) * len(SETTINGS['final_seeds']))
            arms = (list(ARMS) if name == 'final' else ['gchm_v2'] * len(ABLATIONS) + ['gchm_v1'])
            prior += sum(PRIOR_SECONDS[arm] for arm in arms) * len(SETTINGS['final_seeds'])
            print(f'\n[{name}] {count} runs; configs come from selection.json after tuning')
            continue
        if name in ('select', 'report'):
            print(f'\n[{name}] reads completed outputs only; no training')
            continue
        print(f'\n[{name}] {len(cells)} runs')
        for spec in cells:
            if name != 'pilot':
                prior += PRIOR_SECONDS[spec['arm']]
            print(f"  {spec['name']}: {shlex.join(command(root, spec)[1:])}")
    if prior:
        print(f'\nprior ETA (historical per-run seconds, v2 assumed 1.3x v1): '
              f'~{prior / 3600:.1f} h sequential, ~{prior / 3600 / 3 * 1.2:.1f} h at 3 jobs')


def show_status(root):
    path = root / 'state.json'
    if not path.exists():
        print(f'no state yet at {root}')
        return
    state = json.loads(path.read_text())
    cells = state.get('cells', {})
    counts = Counter(entry.get('status') for entry in cells.values())
    print(f'root={root}\nstatus counts: {dict(counts)}')
    print(f"preflight: {state.get('preflight')}")
    for name, entry in sorted(cells.items()):
        if entry.get('status') in ('running', 'failed', 'interrupted'):
            detail = entry.get('problems', [''])[0] if entry.get('problems') else ''
            print(f"  {entry['status']:>11}  {name}  {entry.get('started_at', '')} {detail}")
    durations = [entry['duration_s'] for entry in cells.values()
                 if entry.get('status') == 'completed' and entry.get('duration_s')]
    total = len(tune_cells()) + len(ARMS) * len(SETTINGS['final_seeds']) + \
        (len(ABLATIONS) + 1) * len(SETTINGS['final_seeds'])
    done = sum(1 for name, entry in cells.items()
               if entry.get('status') == 'completed' and not name.startswith('pilot/'))
    if durations:
        left = (total - done) * float(np.mean(durations)) / 3 / 60
        print(f'protocol runs completed: {done}/{total}; measured ETA at 3 jobs ~{left:.0f} min')


def execute(root, args):
    assert_matched_budget()
    root.mkdir(parents=True, exist_ok=True)
    (root / 'logs').mkdir(exist_ok=True)
    with RunnerLock(root):
        ledger = Ledger(root)
        ledger.recover_interrupted()
        preflight(root, ledger, args)
        if args.stage == 'pilot':
            ok = run_waves(root, ledger, pilot_waves(), args)
            print('pilot ' + ('passed' if ok else 'FAILED; see logs/'), flush=True)
            return 0 if ok else 1
        locked = ensure_lock(root)
        print(f"protocol locked: {len(locked['source_code'])} source files, "
              f"{SETTINGS['trials_per_arm']} trials per arm", flush=True)
        for stage in (STAGES if args.stage == 'all' else (args.stage,)):
            print(f'== stage {stage} ({now()})', flush=True)
            if stage == 'tune':
                if not run_waves(root, ledger, [tune_cells()], args):
                    print('tuning incomplete; fix or --retry-failed, then rerun', flush=True)
                    return 1
            elif stage == 'select':
                selection = select(root, ledger)
                for arm, entry in selection['arms'].items():
                    print(f"  {arm}: trial {entry['selected_trial']} "
                          f"dev macro-F1 {entry['dev_macro_f1']:.4f} {entry['config']}")
            elif stage == 'final':
                if not run_waves(root, ledger, final_waves(load_selection(root, ledger)), args):
                    print('final runs incomplete; fix or --retry-failed, then rerun', flush=True)
                    return 1
            elif stage == 'ablate':
                if not run_waves(root, ledger, [ablate_cells(load_selection(root, ledger))], args):
                    print('ablations incomplete; the report will omit them', flush=True)
            elif stage == 'report':
                report = build_report(root, ledger, load_selection(root, ledger))
                print(f"verdict: {report['verdict']['label']} — {report['verdict']['sentence']}")
                print(f"report: {root / 'report.md'}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--stage', choices=('pilot', 'all') + STAGES, default='all')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--jobs', type=int, default=3)
    parser.add_argument('--threads', type=int, default=6, help='OMP threads per job')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args(argv)
    if args.jobs < 1 or args.threads < 1:
        parser.error('--jobs and --threads must be positive')
    root = Path(args.out)
    root = root if root.is_absolute() else ROOT / root
    if args.status:
        show_status(root)
        return 0
    if not args.execute:
        dry_run(root, args.stage)
        return 0
    return execute(root, args)


if __name__ == '__main__':
    sys.exit(main())
