"""CEI-GNN v3 unit U1: fixed train-fitted piecewise-linear encoding (PLE) state and basis.

Pure numpy/torch on arrays the caller passes in; reads no files.
Spec: v3 design §4.2 as amended by §12 (F1, F2, F14, F15, F16) and extensions spec §9 U1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch

QUANTILE_METHOD = 'linear'
MIN_ACTIVE_KNOTS = 3
TRANSFORM_ZSCORE = 'zscore'
TRANSFORM_SIGNED_LOG = 'signed_log'
STATE_VERSION = 1


@dataclass
class KnotTable:
    K: int
    items: Dict[str, dict] = field(default_factory=dict)
    method: str = QUANTILE_METHOD
    min_values: int = 20
    token_min_count: int = 20
    below_threshold: Dict[str, int] = field(default_factory=dict)

    def rows(self) -> Dict[str, int]:
        return {}

    def state(self) -> dict:
        return {}

    def sha256(self) -> str:
        return ''

    @classmethod
    def load(cls, state: dict) -> 'KnotTable':
        return cls(K=0)

    def tensor(self) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        return torch.zeros((0, self.K + 1), dtype=torch.float32), torch.zeros((0,), dtype=torch.bool), []


def fit_knots(values_by_item: Dict[str, np.ndarray], K: int, *, min_values: int = 20,
              transform_by_item: Dict[str, str] = None, token_min_count: int = 20) -> KnotTable:
    return KnotTable(K=K, min_values=min_values, token_min_count=token_min_count)


def ple_basis(values: torch.Tensor, has_value: torch.Tensor, knot_row: torch.Tensor,
              knots: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    return torch.zeros((values.shape[0], knots.shape[1]), dtype=torch.float32)
