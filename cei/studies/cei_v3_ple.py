"""CEI-GNN v3 unit U1: fixed train-fitted piecewise-linear encoding (PLE) state and basis.

Pure numpy/torch on arrays the caller passes in; reads no files.
Spec: v3 design §4.2 as amended by §12 (F1, F2, F14, F15, F16) and extensions spec §9 U1.

Knot fitting (F1/F2/F16): for each item with >= ``min_values`` finite float32 values, the
K+1 quantiles at ``j/K`` are taken over ALL values with multiplicity via
``numpy.quantile(method='linear')``, cast to float32, then equal knots are collapsed
(lowest quantile index canonical). Fewer than 3 distinct knots -> the item is inactive.
"""
from __future__ import annotations

import hashlib
import json
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
        """Canonical JSON-serialisable state (no NaN/inf; knots are float32-exact floats)."""
        items = {}
        for key in sorted(self.items):
            row = self.items[key]
            items[key] = {
                'knots': [float(np.float32(k)) for k in row['knots']],
                'effective_knots': int(row['effective_knots']),
                'active': bool(row['active']),
                'count': int(row['count']),
                'transform': str(row['transform']),
            }
        return {
            'state_version': STATE_VERSION,
            'method': self.method,
            'K': int(self.K),
            'min_values': int(self.min_values),
            'token_min_count': int(self.token_min_count),
            'items': items,
            'below_threshold': {k: int(v) for k, v in sorted(self.below_threshold.items())},
        }

    def sha256(self) -> str:
        payload = json.dumps(self.state(), sort_keys=True, separators=(',', ':'), allow_nan=False)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    @classmethod
    def load(cls, state: dict) -> 'KnotTable':
        """Rebuild from ``state()`` output; every bound field and row invariant is asserted."""
        required = ('state_version', 'method', 'K', 'min_values', 'token_min_count', 'items')
        missing = [k for k in required if k not in state]
        if missing:
            raise ValueError(f'knot state missing fields: {missing}')
        if state['state_version'] != STATE_VERSION:
            raise ValueError(f'unsupported knot state_version {state["state_version"]!r}')
        if state['method'] != QUANTILE_METHOD:
            raise ValueError(f'unsupported quantile method {state["method"]!r}')
        K = state['K']
        if not isinstance(K, int) or isinstance(K, bool) or K < 1:
            raise ValueError(f'invalid K {K!r}')
        items: Dict[str, dict] = {}
        for key, row in state['items'].items():
            knots = np.asarray(row['knots'], dtype=np.float64)
            if knots.ndim != 1 or knots.size < 1 or knots.size > K + 1:
                raise ValueError(f'{key}: knot count {knots.size} not in 1..{K + 1}')
            if not np.all(np.isfinite(knots)):
                raise ValueError(f'{key}: non-finite knot')
            knots32 = knots.astype(np.float32)
            if not np.array_equal(knots32.astype(np.float64), knots):
                raise ValueError(f'{key}: knots are not float32-exact')
            if knots32.size > 1 and not np.all(np.diff(knots32) > 0):
                raise ValueError(f'{key}: knots not strictly increasing in float32')
            if int(row['effective_knots']) != knots32.size:
                raise ValueError(f'{key}: effective_knots does not match knot list')
            if bool(row['active']) != (knots32.size >= MIN_ACTIVE_KNOTS):
                raise ValueError(f'{key}: active flag inconsistent with knot count')
            if row['transform'] not in TRANSFORMS:
                raise ValueError(f'{key}: unknown transform {row["transform"]!r}')
            items[key] = {
                'knots': [float(k) for k in knots32],
                'effective_knots': int(knots32.size),
                'active': bool(row['active']),
                'count': int(row['count']),
                'transform': str(row['transform']),
            }
        return cls(K=K, items=items, method=state['method'], min_values=int(state['min_values']),
                   token_min_count=int(state['token_min_count']),
                   below_threshold={k: int(v) for k, v in state.get('below_threshold', {}).items()})

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


def _validate_knots(knots: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Return effective knot count per row; raise on any invalid active row (F2)."""
    if knots.dim() != 2 or knots.dtype != torch.float32:
        raise ValueError('knots must be float32[items, K+1]')
    if active.shape != (knots.shape[0],) or active.dtype != torch.bool:
        raise ValueError('active must be bool[items]')
    padded = torch.isposinf(knots)
    counts = (~padded).sum(dim=1)
    if knots.shape[0] == 0:
        return counts
    # padding must be a suffix of +inf: no finite knot after the first padding column
    cols = torch.arange(knots.shape[1]).unsqueeze(0)
    if bool((padded & (cols < counts.unsqueeze(1))).any()):
        raise ValueError('knot padding must be a trailing +inf suffix')
    act = active
    if bool(act.any()):
        rows = knots[act]
        n = counts[act]
        if not bool(torch.isfinite(rows[cols.expand_as(rows) < n.unsqueeze(1)]).all()):
            raise ValueError('non-finite knot in an active row')
        if bool((n < MIN_ACTIVE_KNOTS).any()):
            raise ValueError('active row with fewer than 3 knots')
        widths = rows[:, 1:] - rows[:, :-1]
        valid = (cols[:, 1:].expand_as(widths) < n.unsqueeze(1))
        if not bool((widths[valid] > 0).all()):
            raise ValueError('active knot row is not strictly increasing (zero-width interval)')
    return counts


def ple_basis(values: torch.Tensor, has_value: torch.Tensor, knot_row: torch.Tensor,
              knots: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear hat basis float32[N, K+1].

    Interior: linear interpolation between the two adjacent knots; outside the fitted
    range: the endpoint basis (clamped, no extrapolation). Zero rows for ``has_value == 0``,
    an inactive row, or ``knot_row == -1`` (no table). Raises on a zero-width interval, a
    non-finite active knot, or an out-of-range row index; asserts the result is finite."""
    counts = _validate_knots(knots, active)
    n_items, width = knots.shape
    values = values.to(torch.float32).reshape(-1)
    has_value = has_value.to(torch.float32).reshape(-1)
    knot_row = knot_row.to(torch.long).reshape(-1)
    n = values.shape[0]
    if has_value.shape[0] != n or knot_row.shape[0] != n:
        raise ValueError('values, has_value and knot_row must share length N')
    out = torch.zeros((n, width), dtype=torch.float32, device=knots.device)
    if n == 0:
        return out
    if bool((knot_row < -1).any()) or bool((knot_row >= n_items).any()):
        raise ValueError('knot_row out of range')
    use = (knot_row >= 0) & (has_value > 0)
    if n_items > 0:
        use = use & active[knot_row.clamp(min=0)]
    idx = use.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return out
    rows = knot_row[idx]
    v = values[idx]
    if not bool(torch.isfinite(v).all()):
        raise ValueError('non-finite value with has_value=1')
    k = knots[rows]                                   # [n, K+1]
    last = counts[rows] - 1                           # index of last real knot
    lo = k[:, 0]
    hi = k.gather(1, last.unsqueeze(1)).squeeze(1)
    v = torch.minimum(torch.maximum(v, lo), hi)       # clamp to the fitted range
    # right index: number of knots <= v, in [1, last]; left = right - 1
    right = (k <= v.unsqueeze(1)).sum(dim=1).clamp(max=last)
    right = torch.maximum(right, torch.ones_like(right))
    left = right - 1
    k_left = k.gather(1, left.unsqueeze(1)).squeeze(1)
    k_right = k.gather(1, right.unsqueeze(1)).squeeze(1)
    w = (v - k_left) / (k_right - k_left)
    w = w.clamp(0.0, 1.0)
    basis = torch.zeros((idx.numel(), width), dtype=torch.float32, device=knots.device)
    basis.scatter_(1, left.unsqueeze(1), (1.0 - w).unsqueeze(1))
    basis.scatter_add_(1, right.unsqueeze(1), w.unsqueeze(1))
    out[idx] = basis
    if not bool(torch.isfinite(out).all()):
        raise ValueError('non-finite PLE basis')
    return out
