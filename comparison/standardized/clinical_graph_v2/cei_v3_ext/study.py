"""CEI-GNN v3 extension study (EXT spec §9 X6): combined run plan lock, family bounds and
the family-corrected extension decision.

Five extension arms (E2w, E2d, O, E6a, E6b) are contrasted against v3 arm C on the screen
fold (EXT §6, §7, §8). This module only returns plans (command lines, output paths, bound
hashes) and applies decision arithmetic to arrays the caller supplies; it never executes a
stage, opens a data file or scores a fold. Behaviour is added step by step under TDD
(red: extension plan lock / family bounds / extension decision).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .. import cei_v3_study as study
from .arm_guards import CONTROL_ARM, EXTENSION_ARMS

FAMILY_M = 5
OFFSET_ARM = 'O'
FAMILY_ARMS = ('E2w', 'E2d', OFFSET_ARM, 'E6a', 'E6b')   # EXT §7, in the §6 table order
FAMILY_CONTRASTS = tuple((arm, CONTROL_ARM) for arm in FAMILY_ARMS)
TRAINED_EXTENSION_ARMS = EXTENSION_ARMS                   # E2w, E2d, E6a, E6b (X14)
SCREEN_ARMS = ('A', 'B', 'C') + TRAINED_EXTENSION_ARMS + (OFFSET_ARM,)   # 24 screen rows


@dataclass(frozen=True)
class ExtensionConfig:
    """Extension plan inputs (EXT §6–§8). `k_selection` is the frozen `k_selection.json`
    content the v3 plan was built with; the O hashes may be added once delta is frozen."""
    k_selection: dict
    arms: Tuple[str, ...] = FAMILY_ARMS
    m: int = FAMILY_M
    seeds: Tuple[int, ...] = study.SEEDS
    selection_fold: str = 'dev'
    dev_limit: int = 5000
    final_eval: str = 'none'
    output_root: Optional[str] = None
    approval_record_sha256: Optional[str] = None
    offset_record_sha256: Optional[Dict[int, str]] = None


@dataclass(frozen=True)
class ScreenRow:
    name: str
    arm: str
    seed: int
    stage: str
    source: str
    output: str
    offset_record_sha256: Optional[str] = None
    approval_record_sha256: Optional[str] = None


@dataclass(frozen=True)
class ExtensionPlan(study.Plan):
    arms: Tuple[str, ...] = ()
    arms_not_run: Tuple[str, ...] = ()
    m: int = 0
    contrasts: Tuple[Tuple[str, str], ...] = ()
    control_stages: Tuple[str, ...] = ()
    control_binding_sha256: Optional[Tuple[str, ...]] = None
    screen_rows: Tuple[ScreenRow, ...] = ()
    approval_record_sha256: Optional[str] = None
    offset_record_sha256: Optional[Dict[int, str]] = None
    validation_evaluated: bool = False
    test_evaluated: bool = False


def extension_plan(v3_plan, config) -> study.Plan:
    """Locked combined plan (EXT §8) on top of a frozen v3 plan. Stub: empty plan."""
    return ExtensionPlan(output_root=str(getattr(v3_plan, 'output_root', '')), v3_state_paths={},
                         stages=(), k_selection_path='', k_selection_sha256=None, order=())
