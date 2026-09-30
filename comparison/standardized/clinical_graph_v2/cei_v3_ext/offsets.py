"""Item 5 — validation-tuned per-class logit offsets (EXT spec §4, §9 X3).

Arm O = a frozen arm-C checkpoint whose prediction is `argmax_c (z_c + delta_c)` on its stored
raw float32 logits; `delta` is fitted on the validation fold (opened for this item only by the
approval record) by coordinate ascent over the integer grid i in {-20..20}, delta = i / 10.

Stub: behaviour is added step by step under TDD (red: offset fitter / offset freeze replay /
offset screen application).
"""
from __future__ import annotations

import numpy as np

from ..cei_v3_study import NUM_CLASSES, weighted_macro_f1  # noqa: F401  (bound metric, E13)

GRID = range(-20, 21)
SWEEPS = 5
DELTA_SCALE = 10          # delta = i / DELTA_SCALE
METRIC_NAME = 'weighted_macro_f1_unrounded'
VALIDATION_FOLD = 'validation'


def apply_offsets(logits, delta_int) -> np.ndarray:
    """Prediction `argmax(z + delta)` on raw logits (stub)."""
    logits = np.asarray(logits)
    return np.zeros(logits.shape[0], dtype=np.int64)


def fit_offsets(logits, y, *, grid=GRID, sweeps=SWEEPS, fold=VALIDATION_FOLD,
                row_splits=None, trace=None) -> np.ndarray:
    """Coordinate-ascent offsets on validation logits (stub: returns zeros)."""
    return np.zeros(NUM_CLASSES, dtype=np.int64)
