"""CEI-GNN v3 study core (unit U5): plan lock, v3 state fit, K selection, screen scoring,
paired bootstrap and the pre-registered decision rule.

Spec: v3 design §6 and §11.2 as amended by §12 (F4–F7, F17, F18, F20, item 21), extensions
spec §4 (E4, E7) and §9 U5. Reuses U1 (`cei_v3_ple`), U2 (`cei_v3_absence`), U3
(`methods.plugin_cei_gnn_v3`) and U4 (`cei_v3_screen`); `cei_v2_study.py` stays untouched
and its `weighted_macro_f1` / `paired_bootstrap` are re-implemented here so v2 files stay
byte-identical (F20).

Stub: behaviour is added step by step under TDD (red: plan lock / k selection /
bootstrap and screen result).
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

STUDY_METHOD = 'cei_gnn_v3'
ARMS = ('A', 'B', 'C')
SEEDS = (1234, 2025, 7)
SAMPLE_SEED = 1234
K_GRID = (4, 8, 16)
FULL_BUDGET = (10000, 5000, 40)   # train rows, dev rows, epochs
PATIENCE = 40
TOP_K_LABELS = 10
NUM_CLASSES = 10
MANDATORY_FLAGS = ('--selection-fold', 'dev', '--dev-limit', '5000', '--final-eval', 'none')
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 2026
QUANTILE_METHOD = 'linear'
K_SELECTION_VERSION = 'cei_v3_k_selection_v1'
K_SELECTION_RULE = ('arm C only; seeds 1234, 2025, 7; statistic = unweighted mean over the three '
                    'seeds of selected_dev.metric_value (OLD-dev macro-F1 of the dev-selected '
                    'checkpoint, 6-decimal rounded by the runner); winner = argmax over K in '
                    '{4, 8, 16}; tie (equal 6-decimal seed-mean) -> smaller K; no re-run, no '
                    'extra seed, no secondary metric')
V3_STATE_DIRNAME = 'v3_state'
K_SELECTION_FILENAME = 'k_selection.json'
STUDY_BINDING_FILENAME = 'study_binding.json'
SCREEN_DIRNAME = 'screen'


# ------------------------------------------------------------------ plan lock


@dataclass(frozen=True)
class StudyConfig:
    artifact: str
    targets: str
    canonical: str
    output_root: str
    k_grid: Tuple[int, ...] = K_GRID
    seeds: Tuple[int, ...] = SEEDS
    budget: Tuple[int, int, int] = FULL_BUDGET
    selection_fold: str = 'dev'
    final_eval: str = 'none'
    arms: Tuple[str, ...] = ARMS
    k_selection: Optional[dict] = None   # frozen k_selection.json content, once written


@dataclass(frozen=True)
class Stage:
    name: str
    phase: str          # 'c_grid' (before the K freeze) or 'post_freeze' (A/B)
    arm: str
    k: Optional[int]
    seed: int
    output: str
    argv: Tuple[str, ...]
    v3_state: str


@dataclass(frozen=True)
class Plan:
    output_root: str
    v3_state_paths: Dict[int, str]
    stages: Tuple[Stage, ...]
    k_selection_path: str
    k_selection_sha256: Optional[str]
    order: Tuple[str, ...]


def v3_state_path(output_root, k) -> Path:
    return Path(output_root) / V3_STATE_DIRNAME / f'K{int(k)}.json'


def plan(config) -> Plan:
    return Plan(output_root='', v3_state_paths={}, stages=(), k_selection_path='',
                k_selection_sha256=None, order=())


def validate_v3_binding(binding, stage, *, screen_record=None, k_selection=None,
                        study_binding=None) -> None:
    return None


def check_stage_result(stage_dir, binding, result) -> None:
    return None


def write_study_binding(stage_dir, stage, k_selection) -> dict:
    return {}


# --------------------------------------------------------------- v3 state fit


@dataclass
class V3FitInputs:
    values_by_item: Dict[str, np.ndarray]
    transform_by_item: Dict[str, str]
    identity_graph_counts: Dict[str, int]
    vocabulary: Dict[str, int]
    vocabulary_tokens: Tuple[str, ...]
    token_min_count: int
    preprocessing_sha256: str
    node_feature_layout: Tuple[str, ...]
    train_count: int = 0


def read_v3_fit_inputs(artifact, targets_path, *, train_limit=FULL_BUDGET[0],
                       sample_seed=SAMPLE_SEED, token_min_count=20,
                       top_k_labels=TOP_K_LABELS) -> V3FitInputs:
    raise RuntimeError('not implemented in the stub')


def fit_v3_state(output_root, k, *, reader, min_values=20, min_graphs=20) -> Path:
    return Path(output_root)


# --------------------------------------------------------------- K selection


@dataclass(frozen=True)
class KSelection:
    k_grid: Tuple[int, ...]
    seeds: Tuple[int, ...]
    arm: str
    statistic: Dict[Tuple[int, int], float]
    seed_means: Dict[int, float]
    k_selected: int
    tie_rule_applied: bool
    knot_table_sha256_by_k: Dict[int, str]
    v3_state_sha256_by_k: Dict[int, str]
    dev_sample_ids_sha256: str
    rule: str = K_SELECTION_RULE


def select_k(bindings) -> KSelection:
    return KSelection(k_grid=(), seeds=(), arm='', statistic={}, seed_means={}, k_selected=0,
                      tie_rule_applied=False, knot_table_sha256_by_k={}, v3_state_sha256_by_k={},
                      dev_sample_ids_sha256='')


def k_selection_sha256(record) -> str:
    return ''


def write_k_selection(selection, stage_dirs, path) -> dict:
    return {}


def assert_k_selection_replay(record, stage_dirs) -> None:
    return None


# ----------------------------------------------------------- metrics, bootstrap


def weighted_macro_f1(y, pred, weights=None, *, num_classes=NUM_CLASSES) -> float:
    return 0.0


def paired_bootstrap(arms, contrasts, *, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    return {}


def decide_v3(deltas, scores, *, treatment='C', control='A', seeds=SEEDS) -> dict:
    return {}


def absence_share(contributions, graph_count) -> np.ndarray:
    return np.zeros(int(graph_count))


# -------------------------------------------------------------- screen scoring


@dataclass(frozen=True)
class Checkpoint:
    stage_dir: str
    k_selection_path: str


@dataclass(frozen=True)
class ScreenEncoder:
    ids: Tuple[str, ...]
    fold: str
    rows: Callable[[], Iterable]


@dataclass(frozen=True)
class ScreenResult:
    arm: str
    seed: int
    k: int
    checkpoint_sha256: str
    binding_sha256: str
    k_selection_sha256: str
    screen_record_sha256: str
    row_count: int
    macro_f1: float
    logits_path: str
    logits_sha256: str
    proba_path: str
    proba_sha256: str
    absence_share_mean: Optional[float]
    validation_evaluated: bool = False
    test_evaluated: bool = False


def screen_record_sha256(record) -> str:
    return ''


def score_screen(checkpoint, record, encoder, *, output_dir=None, model_factory=None,
                 batch_size=128) -> ScreenResult:
    return ScreenResult(arm='', seed=0, k=0, checkpoint_sha256='', binding_sha256='',
                        k_selection_sha256='', screen_record_sha256='', row_count=0,
                        macro_f1=0.0, logits_path='', logits_sha256='', proba_path='',
                        proba_sha256='', absence_share_mean=None)
