"""CEI-GNN v3 unit U1: fixed train-fitted piecewise-linear encoding (PLE) state and basis.

Pure numpy/torch on arrays the caller passes in; reads no files.
Spec: v3 design §4.2 as amended by §12 (F1, F2, F14, F15, F16) and extensions spec §9 U1.

Knot fitting (F1/F2/F16): for each item with >= ``min_values`` finite float32 values, the
K+1 quantiles at ``j/K`` are taken over ALL values with multiplicity via
``numpy.quantile(method='linear')``, cast to float32, then equal knots are collapsed
(lowest quantile index canonical). Fewer than 3 distinct knots -> the item is inactive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

QUANTILE_METHOD = 'linear'
MIN_ACTIVE_KNOTS = 3
TRANSFORM_ZSCORE = 'zscore'
TRANSFORM_SIGNED_LOG = 'signed_log'
TRANSFORMS = (TRANSFORM_ZSCORE, TRANSFORM_SIGNED_LOG)
STATE_VERSION = 1


@dataclass
class KnotTable:
    """Frozen per-item knot table. ``items`` maps item key -> row dict with keys
    ``knots`` (list[float], float32-exact, strictly increasing), ``effective_knots``,
    ``active``, ``count``, ``transform``."""
    K: int
    items: Dict[str, dict] = field(default_factory=dict)
    method: str = QUANTILE_METHOD
    min_values: int = 20
    token_min_count: int = 20
    below_threshold: Dict[str, int] = field(default_factory=dict)

    def rows(self) -> Dict[str, int]:
        """Item key -> row index in ``tensor()`` (sorted item keys)."""
        return {key: i for i, key in enumerate(sorted(self.items))}

    def state(self) -> dict:
        return {}

    def sha256(self) -> str:
        return ''

    @classmethod
    def load(cls, state: dict) -> 'KnotTable':
        return cls(K=0)

    def tensor(self) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """(knots float32[items, K+1] padded with +inf, active bool[items], transform list)."""
        keys = sorted(self.items)
        knots = torch.full((len(keys), self.K + 1), float('inf'), dtype=torch.float32)
        active = torch.zeros((len(keys),), dtype=torch.bool)
        transform = []
        for i, key in enumerate(keys):
            row = self.items[key]
            values = torch.as_tensor(np.asarray(row['knots'], dtype=np.float32))
            knots[i, :values.numel()] = values
            active[i] = bool(row['active'])
            transform.append(str(row['transform']))
        return knots, active, transform


def _fit_row(values: np.ndarray, K: int, transform: str) -> dict:
    probs = np.arange(K + 1, dtype=np.float64) / float(K)
    quantiles = np.quantile(values.astype(np.float64), probs, method=QUANTILE_METHOD)
    quantiles32 = quantiles.astype(np.float32)
    if not np.all(np.isfinite(quantiles32)):
        raise ValueError('non-finite knot after float32 cast')
    collapsed: List[np.float32] = []
    for q in quantiles32:  # keeps the lowest quantile index of each tie as canonical
        if not collapsed or q != collapsed[-1]:
            collapsed.append(q)
    knots = np.asarray(collapsed, dtype=np.float32)
    if knots.size > 1 and not np.all(np.diff(knots) > 0):
        raise ValueError('knots not strictly increasing after collapse')
    return {
        'knots': [float(k) for k in knots],
        'effective_knots': int(knots.size),
        'active': bool(knots.size >= MIN_ACTIVE_KNOTS),
        'count': int(values.size),
        'transform': transform,
    }


def fit_knots(values_by_item: Dict[str, np.ndarray], K: int, *, min_values: int = 20,
              transform_by_item: Optional[Dict[str, str]] = None,
              token_min_count: int = 20) -> KnotTable:
    """Fit the knot table from per-node float32 values supplied by the caller (F14).

    ``transform_by_item`` records which items carry signed-log values instead of z-scores
    (F15); every item defaults to ``'zscore'``. Non-finite values are dropped before counting."""
    if not isinstance(K, int) or isinstance(K, bool) or K < 1:
        raise ValueError(f'K must be a positive integer, got {K!r}')
    if min_values < 1:
        raise ValueError('min_values must be >= 1')
    transform_by_item = dict(transform_by_item or {})
    unknown = set(transform_by_item) - set(values_by_item)
    if unknown:
        raise ValueError(f'transform_by_item names unknown items: {sorted(unknown)}')
    table = KnotTable(K=K, min_values=min_values, token_min_count=token_min_count)
    for key in sorted(values_by_item):
        raw = np.asarray(values_by_item[key])
        if raw.dtype != np.float32:
            raise ValueError(f'{key}: values must be float32, got {raw.dtype}')
        values = raw[np.isfinite(raw)]
        transform = transform_by_item.get(key, TRANSFORM_ZSCORE)
        if transform not in TRANSFORMS:
            raise ValueError(f'{key}: unknown transform {transform!r}')
        if values.size < min_values:
            table.below_threshold[key] = int(values.size)
            continue
        table.items[key] = _fit_row(values, K, transform)
    return table


def ple_basis(values: torch.Tensor, has_value: torch.Tensor, knot_row: torch.Tensor,
              knots: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    return torch.zeros((values.shape[0], knots.shape[1]), dtype=torch.float32)
