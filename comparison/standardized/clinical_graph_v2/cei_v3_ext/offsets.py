"""Item 5 — validation-tuned per-class logit offsets (EXT spec §4, §9 X3).

Arm O = a frozen arm-C checkpoint whose prediction is `argmax_c (z_c + delta_c)` on its stored
raw float32 logits; `delta` is fitted on the validation fold (opened for this item only by the
approval record) by coordinate ascent over the integer grid i in {-20..20}, delta = i / 10.
The offsets change the decision rule only: not the model, its checkpoint or its decomposition.

Behaviour is added step by step under TDD (red: offset fitter / offset freeze replay /
offset screen application).
"""
from __future__ import annotations

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
