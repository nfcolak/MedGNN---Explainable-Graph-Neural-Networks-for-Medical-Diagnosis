"""Item 5 — validation-tuned per-class logit offsets (EXT spec §4, §9 X3).

Arm O = a frozen arm-C checkpoint whose prediction is `argmax_c (z_c + delta_c)` on its stored
raw float32 logits; `delta` is fitted on the validation fold (opened for this item only by the
approval record) by coordinate ascent over the integer grid i in {-20..20}, delta = i / 10.
The offsets change the decision rule only: not the model, its checkpoint or its decomposition.

Behaviour is added step by step under TDD (red: offset fitter / offset freeze replay /
offset screen application).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..cei_v3_study import NUM_CLASSES, weighted_macro_f1   # bound metric (E13, v3 §12.20)

GRID = range(-20, 21)
SWEEPS = 5
DELTA_SCALE = 10          # delta = i / DELTA_SCALE
METRIC_NAME = 'weighted_macro_f1_unrounded'
VALIDATION_FOLD = 'validation'
OPTIMIZER_RULE = ('coordinate ascent on the integer grid i in {-20..20} (delta = i/10), start '
                  'i = 0 for every class, exactly 5 sweeps over classes in label order 0..9, '
                  'each coordinate set to the grid value maximising the unrounded validation '
                  'weighted_macro_f1 with the others fixed; ties -> smallest |i|, then the '
                  'smaller (more negative) i; no early stop, no restart, no other grid')


def _check_logits_and_labels(logits, y, *, num_classes=NUM_CLASSES):
    logits = np.asarray(logits)
    y = np.asarray(y)
    if logits.ndim != 2 or logits.shape[1] != num_classes:
        raise ValueError(f'logits must have shape [n, {num_classes}], got {logits.shape}')
    if y.ndim != 1 or y.shape[0] != logits.shape[0] or logits.shape[0] == 0:
        raise ValueError('y must be a nonempty 1-D label vector aligned with the logits rows')
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError('y must be integer labels')
    if (y < 0).any() or (y >= num_classes).any():
        raise ValueError(f'labels must lie in 0..{num_classes - 1}')
    if not np.isfinite(logits).all():
        raise ValueError('logits must be finite')
    return logits, y.astype(np.int64)


def _check_delta(delta_int, *, num_classes=NUM_CLASSES, grid=GRID):
    delta = np.asarray(delta_int)
    if delta.shape != (num_classes,):
        raise ValueError(f'offsets must have shape [{num_classes}], got {delta.shape}')
    if delta.dtype == bool or not np.issubdtype(delta.dtype, np.integer):
        raise ValueError('offsets are bound as integers i (delta = i / 10); got a non-integer dtype')
    low, high = min(grid), max(grid)
    if (delta < low).any() or (delta > high).any():
        raise ValueError(f'offsets must lie on the grid {low}..{high}')
    return delta.astype(np.int64)


def apply_offsets(logits, delta_int) -> np.ndarray:
    """Prediction `argmax_c (z_c + i_c / 10)` on raw logits (EXT §4.1, §4.3).

    The sum is formed in float64 from the float32 logits and the integer offsets, so the
    decision is exactly reproducible from the stored integers; `log(softmax)` is never used.
    """
    logits = np.asarray(logits)
    if logits.ndim != 2 or logits.shape[1] != NUM_CLASSES:
        raise ValueError(f'logits must have shape [n, {NUM_CLASSES}], got {logits.shape}')
    delta = _check_delta(delta_int)
    shifted = logits.astype(np.float64) + delta.astype(np.float64) / DELTA_SCALE
    return shifted.argmax(1).astype(np.int64)


def _looks_like_log_softmax(logits) -> bool:
    z = logits.astype(np.float64)
    if (z > 0).any():
        return False
    log_norm = np.log(np.exp(z).sum(1))
    return bool(np.abs(log_norm).max() < 1e-4)


def _check_grid(grid):
    values = [int(v) for v in grid]
    if any(int(v) != v for v in grid) or not values:
        raise ValueError('grid must be a nonempty sequence of integers')
    if len(set(values)) != len(values):
        raise ValueError('grid values must be distinct')
    if 0 not in values:
        raise ValueError('grid must contain the start point i = 0 (EXT §4.2)')
    return values


def fit_offsets(logits, y, *, grid=GRID, sweeps=SWEEPS, fold=VALIDATION_FOLD,
                row_splits=None, trace: Optional[list] = None) -> np.ndarray:
    """Coordinate-ascent offsets on validation logits (EXT §4.2, E13).

    `logits` must be the raw float32 validation logits, `y` the aligned labels; `fold` and
    `row_splits` (when given) must be `'validation'` (the fitter accepts validation rows
    only). Returns `i` as `int64[10]`; `trace`, when a list is passed, receives one dict
    `{sweep, label, i, score}` per coordinate visit (5 x 10 entries).
    """
    if fold != VALIDATION_FOLD:
        raise ValueError(f"offsets are fitted on the validation fold only, not {fold!r}")
    logits, y = _check_logits_and_labels(logits, y)
    if logits.dtype != np.float32:
        raise ValueError(f'logits must be the stored raw float32 logits, got dtype {logits.dtype}')
    if _looks_like_log_softmax(logits):
        raise ValueError('logits look like log-softmax outputs (every row normalises to 1); '
                         'offsets are fitted on raw logits only (EXT §4.3)')
    if row_splits is not None:
        splits = [str(s) for s in row_splits]
        if len(splits) != logits.shape[0]:
            raise ValueError('row_splits must be aligned with the logits rows')
        foreign = sorted({s for s in splits if s != VALIDATION_FOLD})
        if foreign:
            raise ValueError(f'offsets are fitted on validation rows only; rows carry '
                             f'{foreign}')
    values = _check_grid(grid)
    if isinstance(sweeps, bool) or int(sweeps) < 1:
        raise ValueError('sweeps must be a positive integer')
    current = np.zeros(NUM_CLASSES, dtype=np.int64)
    for sweep in range(int(sweeps)):
        for label in range(NUM_CLASSES):
            best_key, best_i = None, None
            for i in values:
                trial = current.copy()
                trial[label] = i
                score = float(weighted_macro_f1(y, apply_offsets(logits, trial)))
                key = (score, -abs(i), -i)          # max score; smallest |i|; more negative i
                if best_key is None or key > best_key:
                    best_key, best_i = key, i
            current[label] = best_i
            if trace is not None:
                trace.append({'sweep': sweep, 'label': label, 'i': int(best_i),
                              'score': best_key[0]})
    return current


# ------------------------------------------------------------- freeze record

OFFSET_RECORD_VERSION = 'cei_v3_item5_offsets_v1'
OFFSET_RECORD_FILENAME = 'offset_record.json'
TUNING_SCORE_CAVEAT = ('validation macro-F1 before/after the offsets is a tuning score on the '
                       'fold the offsets were fitted on, not a result (EXT §4.2, §4.4)')
_RECORD_KEYS = ('version', 'arm', 'control_arm', 'stage', 'seed', 'k', 'delta_int', 'delta',
                'grid', 'sweeps', 'delta_scale', 'metric', 'optimizer_rule',
                'checkpoint_sha256', 'binding_sha256', 'k_selection_sha256',
                'validation_sample_ids_sha256', 'validation_row_count',
                'validation_logits_sha256', 'validation_logits_path', 'approval_record_sha256',
                'validation_result_sha256', 'tuning_macro_f1_before', 'tuning_macro_f1_after',
                'tuning_score_caveat', 'validation_evaluated', 'test_evaluated',
                'screen_read_before_freeze')


def _canonical_bytes(document) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + '\n').encode('utf-8')


def _file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def offset_record_sha256(record) -> str:
    """SHA-256 of the frozen offset record (equals the SHA-256 of the written file bytes)."""
    return hashlib.sha256(_canonical_bytes(record)).hexdigest()


def _load_scored_validation(validation_dir, approval_path):
    """Read the X3 validation scoring output and check every hash it binds against disk.

    Returns `(stage_dir, result, approval, approval_hash, logits, y, sample_ids, logits_hash)`.
    """
    from . import validation_scoring as vs
    from ....core.contracts import sample_ids_sha256

    validation_dir = Path(validation_dir)
    stage_dir = validation_dir.parent
    result_path = validation_dir / vs.VALIDATION_RESULT_FILENAME
    logits_path = validation_dir / 'logits.npz'
    for path in (result_path, logits_path):
        if not path.is_file():
            raise ValueError(f'validation scoring output lacks {path.name}: {validation_dir}')
    result = json.loads(result_path.read_bytes().decode('utf-8'))
    if not isinstance(result, dict) or result.get('fold') != VALIDATION_FOLD:
        raise ValueError('validation_result.json is not an X3 validation scoring result')
    if result.get('arm') != 'C':
        raise ValueError(f"offsets are fitted on arm C only; result arm {result.get('arm')!r}")
    if result.get('metric') != METRIC_NAME:
        raise ValueError(f'validation result binds metric {result.get("metric")!r}, '
                         f'not {METRIC_NAME!r}')
    approval, approval_hash = vs.load_approval_record(approval_path)
    if approval_hash != result.get('approval_record_sha256'):
        raise ValueError('approval record differs from the one bound at validation scoring')
    checkpoint_path, binding_path = stage_dir / 'best.pt', stage_dir / 'binding.json'
    for path in (checkpoint_path, binding_path):
        if not path.is_file():
            raise ValueError(f'checkpoint directory lacks {path.name}: {stage_dir}')
    checkpoint_sha = _file_sha256(checkpoint_path)
    if checkpoint_sha != result.get('checkpoint_sha256'):
        raise ValueError('checkpoint best.pt differs from the one scored on validation')
    if checkpoint_sha not in approval.get('checkpoint_sha256', []):
        raise ValueError('approval record does not list this checkpoint SHA-256')
    if _file_sha256(binding_path) != result.get('binding_sha256'):
        raise ValueError('binding.json differs from the one scored on validation')
    with np.load(logits_path, allow_pickle=False) as saved:
        if set(saved.files) != {'logits', 'y', 'subjects', 'sample_ids'}:
            raise ValueError('logits.npz does not carry logits, y, subjects, sample_ids')
        logits = np.ascontiguousarray(saved['logits'])
        y = np.asarray(saved['y'])
        sample_ids = [str(s) for s in saved['sample_ids']]
    logits_hash = hashlib.sha256(logits.tobytes()).hexdigest()
    if logits.dtype != np.float32 or logits_hash != result.get('logits_sha256'):
        raise ValueError('validation logits.npz differs from the hashed scoring output')
    ids_hash = sample_ids_sha256(sample_ids)
    if (ids_hash != result.get('validation_sample_ids_sha256')
            or ids_hash != approval.get('validation_sample_ids_sha256')
            or len(sample_ids) != result.get('row_count')):
        raise ValueError('validation sample ids differ from the scoring result / approval record')
    return stage_dir, result, approval, approval_hash, logits, y, sample_ids, logits_hash


def _offset_document(result, approval_hash, logits, y, delta, *, logits_hash, result_hash,
                     logits_path) -> dict:
    delta_int = [int(v) for v in delta]
    return {
        'version': OFFSET_RECORD_VERSION, 'arm': 'O', 'control_arm': 'C',
        'stage': str(result['stage']), 'seed': int(result['seed']), 'k': int(result['k']),
        'delta_int': delta_int, 'delta': [v / DELTA_SCALE for v in delta_int],
        'grid': list(GRID), 'sweeps': SWEEPS, 'delta_scale': DELTA_SCALE,
        'metric': METRIC_NAME, 'optimizer_rule': OPTIMIZER_RULE,
        'checkpoint_sha256': result['checkpoint_sha256'],
        'binding_sha256': result['binding_sha256'],
        'k_selection_sha256': result['k_selection_sha256'],
        'validation_sample_ids_sha256': result['validation_sample_ids_sha256'],
        'validation_row_count': int(result['row_count']),
        'validation_logits_sha256': logits_hash,
        'validation_logits_path': str(logits_path),
        'approval_record_sha256': approval_hash,
        'validation_result_sha256': result_hash,
        'tuning_macro_f1_before': float(weighted_macro_f1(y, logits.argmax(1))),
        'tuning_macro_f1_after': float(weighted_macro_f1(y, apply_offsets(logits, delta))),
        'tuning_score_caveat': TUNING_SCORE_CAVEAT,
        'validation_evaluated': True, 'test_evaluated': False,
        'screen_read_before_freeze': False,
    }


def offset_record(validation_dir, approval_path) -> dict:
    """Fit delta for one scored C checkpoint and freeze it (EXT §4.2 "Freeze").

    `validation_dir` is the directory `score_validation` returned; `approval_path` the
    approval record file bound at scoring time. Every hash is checked against disk before
    the fit; the record is written once to `<validation_dir>/offset_record.json` (never
    overwriting), so its SHA-256 is bound before any screen read.
    """
    from . import validation_scoring as vs

    validation_dir = Path(validation_dir)
    out_path = validation_dir / OFFSET_RECORD_FILENAME
    if out_path.exists():
        raise FileExistsError(f'Refusing occupied offset record {out_path}')
    (_stage_dir, result, _approval, approval_hash, logits, y, _ids,
     logits_hash) = _load_scored_validation(validation_dir, approval_path)
    result_hash = _file_sha256(validation_dir / vs.VALIDATION_RESULT_FILENAME)
    if result.get('tuning_macro_f1') != float(weighted_macro_f1(y, logits.argmax(1))):
        raise ValueError('validation result tuning_macro_f1 differs from the stored logits')
    delta = fit_offsets(logits, y)
    document = _offset_document(result, approval_hash, logits, y, delta, logits_hash=logits_hash,
                                result_hash=result_hash, logits_path=validation_dir / 'logits.npz')
    with out_path.open('xb') as stream:
        stream.write(_canonical_bytes(document))
    return document


def assert_offset_replay(record, validation_dir, approval_path) -> None:
    """Recompute delta from the stored validation logits; raise ValueError on any drift.

    Checks the record's version, grid, sweeps, metric, rule, integer delta and every bound
    hash (checkpoint, binding, K freeze, validation ids, validation logits, approval record,
    validation result), then refits and compares delta and both tuning scores exactly.
    Pure: `record` is not modified.
    """
    from . import validation_scoring as vs

    if not isinstance(record, dict):
        raise ValueError('offset record must be a dict')
    missing = [key for key in _RECORD_KEYS if key not in record]
    if missing:
        raise ValueError(f'offset record lacks bound fields: {missing}')
    if record['version'] != OFFSET_RECORD_VERSION:
        raise ValueError(f'offset record version {record["version"]!r} is not '
                         f'{OFFSET_RECORD_VERSION!r}')
    if record['arm'] != 'O' or record['control_arm'] != 'C':
        raise ValueError('offset record arm must be O on control arm C')
    if list(record['grid']) != list(GRID):
        raise ValueError('offset record grid differs from the frozen grid -20..20')
    if record['sweeps'] != SWEEPS:
        raise ValueError(f'offset record sweeps {record["sweeps"]!r} differs from the fixed {SWEEPS}')
    if record['delta_scale'] != DELTA_SCALE:
        raise ValueError('offset record delta_scale differs from 10')
    if record['metric'] != METRIC_NAME:
        raise ValueError(f'offset record metric {record["metric"]!r} is not {METRIC_NAME!r}')
    if record['optimizer_rule'] != OPTIMIZER_RULE:
        raise ValueError('offset record optimizer rule differs from the pre-registered rule')
    delta_int = record['delta_int']
    if (not isinstance(delta_int, list) or len(delta_int) != NUM_CLASSES
            or any(isinstance(v, bool) or not isinstance(v, int) for v in delta_int)
            or any(v not in GRID for v in delta_int)):
        raise ValueError('offset record delta_int must be ten grid integers')
    if list(record['delta']) != [v / DELTA_SCALE for v in delta_int]:
        raise ValueError('offset record delta differs from delta_int / 10')
    validation_dir = Path(validation_dir)
    (_stage_dir, result, _approval, approval_hash, logits, y, _ids,
     logits_hash) = _load_scored_validation(validation_dir, approval_path)
    result_hash = _file_sha256(validation_dir / vs.VALIDATION_RESULT_FILENAME)
    for key, expected, label in (
            ('checkpoint_sha256', result['checkpoint_sha256'], 'checkpoint'),
            ('binding_sha256', result['binding_sha256'], 'binding'),
            ('k_selection_sha256', result['k_selection_sha256'], 'k_selection'),
            ('validation_sample_ids_sha256', result['validation_sample_ids_sha256'],
             'validation sample ids'),
            ('validation_logits_sha256', logits_hash, 'validation logits'),
            ('approval_record_sha256', approval_hash, 'approval record'),
            ('validation_result_sha256', result_hash, 'validation_result')):
        if record[key] != expected:
            raise ValueError(f'offset record {label} hash ({key}) differs from disk')
    if (record['stage'], record['seed'], record['k'], record['validation_row_count']) != (
            result['stage'], result['seed'], result['k'], result['row_count']):
        raise ValueError('offset record stage/seed/k/row_count differ from the validation result')
    replay = fit_offsets(logits, y)
    if replay.tolist() != delta_int:
        raise ValueError(f'replayed delta_int {replay.tolist()} differs from the frozen '
                         f'{delta_int}')
    before = float(weighted_macro_f1(y, logits.argmax(1)))
    after = float(weighted_macro_f1(y, apply_offsets(logits, replay)))
    if record['tuning_macro_f1_before'] != before or record['tuning_macro_f1_after'] != after:
        raise ValueError('offset record tuning_macro_f1 before/after differ from the '
                         'recomputed scores')
    if record['validation_evaluated'] is not True or record['test_evaluated'] is not False:
        raise ValueError('offset record fold flags differ from the validation-only scoring')
    return None


# --------------------------------------------------------- screen application

OFFSET_SCREEN_DIRNAME = 'O'
OFFSET_SCREEN_RESULT_FILENAME = 'o_result.json'
_SCREEN_RESULT_KEYS = ('arm', 'seed', 'k', 'checkpoint_sha256', 'binding_sha256',
                       'k_selection_sha256', 'screen_record_sha256', 'row_count', 'macro_f1',
                       'logits_path', 'logits_sha256')


def _screen_result_dict(screen_result) -> dict:
    if isinstance(screen_result, dict):
        return dict(screen_result)
    if hasattr(screen_result, '__dataclass_fields__'):
        return dict(vars(screen_result))
    raise ValueError('screen_result must be a U5 ScreenResult or its screen_result.json dict')


def _validate_offset_record_fields(record) -> list:
    """Static checks of a frozen delta record (no disk); returns delta_int."""
    if not isinstance(record, dict):
        raise ValueError('delta must be given as the frozen offset record (dict), never as a '
                         'raw vector; an unbound delta is refused (EXT §4.3)')
    missing = [key for key in _RECORD_KEYS if key not in record]
    if missing:
        raise ValueError(f'offset record lacks bound fields: {missing}; an unbound delta is refused')
    if record['version'] != OFFSET_RECORD_VERSION:
        raise ValueError(f'offset record version {record["version"]!r} is not '
                         f'{OFFSET_RECORD_VERSION!r}')
    if record['arm'] != 'O' or record['control_arm'] != 'C':
        raise ValueError('offset record arm must be O on control arm C')
    if list(record['grid']) != list(GRID) or record['sweeps'] != SWEEPS \
            or record['delta_scale'] != DELTA_SCALE or record['metric'] != METRIC_NAME \
            or record['optimizer_rule'] != OPTIMIZER_RULE:
        raise ValueError('offset record grid/sweeps/scale/metric/rule differ from the '
                         'pre-registered protocol')
    delta_int = record['delta_int']
    if (not isinstance(delta_int, list) or len(delta_int) != NUM_CLASSES
            or any(isinstance(v, bool) or not isinstance(v, int) for v in delta_int)
            or any(v not in GRID for v in delta_int)):
        raise ValueError('offset record delta_int must be ten grid integers')
    if list(record['delta']) != [v / DELTA_SCALE for v in delta_int]:
        raise ValueError('offset record delta differs from delta_int / 10')
    for key in ('checkpoint_sha256', 'binding_sha256', 'k_selection_sha256',
                'approval_record_sha256', 'validation_logits_sha256'):
        value = record[key]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f'offset record {key} is not a hex digest; delta is unbound')
    return delta_int


def score_offset_screen(screen_result, offset_record, *, approval_record_sha256,
                        output_dir=None) -> dict:
    """Arm O = C's frozen delta applied to C's stored raw float32 screen logits (EXT §4.3).

    No re-inference: predictions are `argmax(z + delta)` on `logits.npz` written by the U5
    screen scorer. Refuses a C screen result without a logits hash or path, stored logits
    whose bytes differ from that hash, a non-C result, an unbound / raw / off-grid delta,
    and a delta record whose checkpoint, binding, K-freeze or approval-record hash differs
    from the screen result / the given approval hash. Writes `o_pred.npz` (pred, y,
    subjects, sample_ids) and `o_result.json` into `<screen_dir>/O/` (or `output_dir`),
    never overwriting.
    """
    result = _screen_result_dict(screen_result)
    missing = [key for key in _SCREEN_RESULT_KEYS if key not in result]
    if missing:
        raise ValueError(f'C screen result lacks bound fields: {missing}')
    if result['arm'] != 'C':
        raise ValueError(f"offsets apply to arm C screen logits only, got arm {result['arm']!r}")
    logits_hash = result['logits_sha256']
    if not isinstance(logits_hash, str) or len(logits_hash) != 64:
        raise ValueError('C screen result carries no logits hash; O is never scored from '
                         'proba or unhashed logits (EXT E7)')
    delta_int = _validate_offset_record_fields(offset_record)
    if offset_record['checkpoint_sha256'] != result['checkpoint_sha256']:
        raise ValueError('offset record checkpoint SHA-256 differs from the screened checkpoint')
    if offset_record['binding_sha256'] != result['binding_sha256']:
        raise ValueError('offset record binding SHA-256 differs from the screened checkpoint')
    if offset_record['k_selection_sha256'] != result['k_selection_sha256']:
        raise ValueError('offset record k_selection SHA-256 differs from the screen result')
    if (offset_record['seed'], offset_record['k']) != (result['seed'], result['k']):
        raise ValueError('offset record seed/K differ from the screened checkpoint')
    if not isinstance(approval_record_sha256, str) or \
            offset_record['approval_record_sha256'] != approval_record_sha256:
        raise ValueError('offset record approval-record SHA-256 differs from the bound approval')
    logits_path = Path(result['logits_path'])
    if not logits_path.is_file():
        raise ValueError(f'C screen logits.npz missing at {logits_path}')
    out = Path(output_dir) if output_dir is not None else logits_path.parent / OFFSET_SCREEN_DIRNAME
    if out.exists():
        raise FileExistsError(f'Refusing occupied O output {out}')
    with np.load(logits_path, allow_pickle=False) as saved:
        if set(saved.files) != {'logits', 'y', 'subjects', 'sample_ids'}:
            raise ValueError('screen logits.npz does not carry logits, y, subjects, sample_ids')
        logits = np.ascontiguousarray(saved['logits'])
        y = np.asarray(saved['y'])
        subjects = np.asarray(saved['subjects']).astype(str)
        sample_ids = np.asarray(saved['sample_ids']).astype(str)
    if logits.dtype != np.float32:
        raise ValueError('stored screen logits are not raw float32')
    if hashlib.sha256(logits.tobytes()).hexdigest() != logits_hash:
        raise ValueError('stored screen logits differ from the hash in the C screen result')
    if logits.shape != (int(result['row_count']), NUM_CLASSES):
        raise ValueError(f'stored screen logits shape {logits.shape} differs from the screen '
                         f"result row_count {result['row_count']}")
    delta = np.asarray(delta_int, dtype=np.int64)
    control_pred = logits.argmax(1).astype(np.int64)
    control_f1 = float(weighted_macro_f1(y, control_pred))
    if control_f1 != float(result['macro_f1']):
        raise ValueError('C macro-F1 recomputed from the stored logits differs from the screen result')
    pred = apply_offsets(logits, delta)
    out.mkdir(parents=True, exist_ok=False)
    pred_path = out / 'o_pred.npz'
    np.savez_compressed(pred_path, pred=pred, y=y, subjects=subjects, sample_ids=sample_ids)
    document = {
        'arm': 'O', 'control_arm': 'C', 'seed': int(result['seed']), 'k': int(result['k']),
        'row_count': int(result['row_count']),
        'macro_f1': float(weighted_macro_f1(y, pred)), 'control_macro_f1': control_f1,
        'metric': 'weighted_macro_f1', 'delta_int': list(delta_int),
        'delta': [v / DELTA_SCALE for v in delta_int],
        'pred_path': str(pred_path),
        'pred_sha256': hashlib.sha256(np.ascontiguousarray(pred).tobytes()).hexdigest(),
        'control_pred_sha256': hashlib.sha256(np.ascontiguousarray(control_pred).tobytes()).hexdigest(),
        'changed_row_count': int((pred != control_pred).sum()),
        'checkpoint_sha256': result['checkpoint_sha256'],
        'binding_sha256': result['binding_sha256'],
        'k_selection_sha256': result['k_selection_sha256'],
        'screen_record_sha256': result['screen_record_sha256'],
        'screen_logits_path': str(logits_path), 'screen_logits_sha256': logits_hash,
        'offset_record_sha256': offset_record_sha256(offset_record),
        'approval_record_sha256': approval_record_sha256,
        'reinference': False, 'validation_evaluated': False, 'test_evaluated': False,
        'statement_scope': ('decision-rule change on C: validation-tuned class thresholds; '
                            'not a model change (EXT §4.1, §4.3)'),
    }
    with (out / OFFSET_SCREEN_RESULT_FILENAME).open('xb') as stream:
        stream.write(_canonical_bytes(document))
    return document
