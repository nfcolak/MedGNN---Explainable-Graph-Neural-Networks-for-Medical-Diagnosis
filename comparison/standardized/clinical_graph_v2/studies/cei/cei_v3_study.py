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
ABSENCE_AGGREGATION_LEGACY = 'cei_v3_absence_row_mean_legacy_v1'
ABSENCE_AGGREGATION_VERSION = 'cei_v3_absence_row_and_equal_patient_mean_v2'


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


def _canonical_file_bytes(document) -> bytes:
    """The exact bytes every study JSON file is written with (train.py line 737 layout)."""
    return (json.dumps(document, indent=2, sort_keys=True) + '\n').encode('utf-8')


def _write_new_json(path, document) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:   # 'x': never overwrite an existing study file
        stream.write(_canonical_file_bytes(document))


def _load_json(path, label) -> dict:
    from . import cei_pilot as pilot
    return pilot._load_json(path, label)


def _stage_argv(*, artifact, targets, canonical, output, seed, arm, k, v3_state) -> tuple:
    train_limit, dev_limit, epochs = FULL_BUDGET
    return (
        sys.executable, '-m', 'comparison.standardized.clinical_graph_v2.core.train',
        '--artifact', str(artifact), '--targets', str(targets),
        '--canonical', str(canonical), '--output', str(output),
        '--method', STUDY_METHOD,
        '--train-limit', str(train_limit), '--dev-limit', str(dev_limit),
        '--sample-seed', str(SAMPLE_SEED), '--seed', str(seed),
        '--top-k-labels', str(TOP_K_LABELS), '--edges', 'all',
        '--edge-direction', 'forward', '--weights', 'sqrt_inverse',
        '--selection-fold', 'dev', '--final-eval', 'none',
        '--epochs', str(epochs), '--patience', str(PATIENCE),
        '--method-option', f'arm={arm}',
        '--method-option', f'v3_state={v3_state}',
        '--method-option', f'k={int(k)}',
    )


def _validate_k_selection_record(record) -> None:
    if not isinstance(record, dict):
        raise ValueError('k_selection must be the frozen k_selection.json content (dict)')
    if record.get('version') != K_SELECTION_VERSION:
        raise ValueError(f'k_selection version {record.get("version")!r} is not '
                         f'{K_SELECTION_VERSION!r}')
    if tuple(record.get('k_grid', ())) != K_GRID:
        raise ValueError(f'k_selection grid {record.get("k_grid")!r} differs from the frozen '
                         f'grid {list(K_GRID)}')
    k_selected = record.get('k_selected')
    if isinstance(k_selected, bool) or k_selected not in K_GRID:
        raise ValueError(f'frozen K {k_selected!r} is outside the grid {list(K_GRID)}')
    if record.get('rule') != K_SELECTION_RULE:
        raise ValueError('k_selection rule differs from the pre-registered selection rule '
                         '(v3 §11.2 items 2–4)')


def plan(config) -> Plan:
    """Locked v3 study plan (v3 §11.2 item 11, §12.17; EXT §4 E4).

    Order: fit the three v3 state files, nine C stages over the K grid, the K freeze
    record, then A and B at the frozen K, then screen scoring. A/B stages are only
    materialised (argv, K, state path) once `config.k_selection` holds the frozen record.
    Nothing is executed and no input file is opened.
    """
    from . import cei_pilot as pilot

    if tuple(config.k_grid) != K_GRID:
        raise ValueError(f'K grid {tuple(config.k_grid)} differs from the frozen grid {K_GRID} '
                         '(v3 §11.2 item 1); no candidate may be added or removed')
    if tuple(config.seeds) != SEEDS:
        raise ValueError(f'seeds {tuple(config.seeds)} differ from the pre-registered seeds {SEEDS}')
    if tuple(config.arms) != ARMS:
        raise ValueError(f'arms {tuple(config.arms)} differ from the pre-registered arms {ARMS}; '
                         'no arm may be added or removed after registration')
    if tuple(config.budget) != FULL_BUDGET:
        raise ValueError(f'budget {tuple(config.budget)} differs from the approved budget '
                         f'{FULL_BUDGET} (train rows, dev rows, epochs)')
    if config.selection_fold != 'dev':
        raise ValueError(f"selection_fold must be 'dev', got {config.selection_fold!r}")
    if config.final_eval != 'none':
        raise ValueError(f"final_eval must be 'none' (EXT §4 E4), got {config.final_eval!r}; "
                         'train.py would otherwise score validation once per stage')
    artifact = pilot._absolute_existing(config.artifact, 'artifact', directory=True)
    targets = pilot._absolute_existing(config.targets, 'targets')
    canonical = pilot._absolute_existing(config.canonical, 'canonical')
    root = Path(config.output_root).expanduser()
    if not root.is_absolute():
        raise ValueError(f'output_root path must be absolute: {config.output_root}')
    root = root.resolve()
    frozen = config.k_selection
    if frozen is not None:
        _validate_k_selection_record(frozen)
    state_paths = {k: str(v3_state_path(root, k)) for k in K_GRID}
    stages: List[Stage] = []
    for k in K_GRID:
        for seed in SEEDS:
            name = f'C_K{k}_seed{seed}'
            output = root / name
            if frozen is None and output.exists():
                raise FileExistsError(f'Refusing occupied stage output {output}')
            stages.append(Stage(
                name=name, phase='c_grid', arm='C', k=k, seed=seed, output=str(output),
                argv=_stage_argv(artifact=artifact, targets=targets, canonical=canonical,
                                 output=output, seed=seed, arm='C', k=k,
                                 v3_state=state_paths[k]),
                v3_state=state_paths[k]))
    for arm in ('A', 'B'):
        for seed in SEEDS:
            name = f'{arm}_seed{seed}'
            output = root / name
            if output.exists():
                raise FileExistsError(f'Refusing occupied stage output {output}')
            if frozen is None:
                stages.append(Stage(name=name, phase='post_freeze', arm=arm, k=None, seed=seed,
                                    output=str(output), argv=(), v3_state=''))
                continue
            k = int(frozen['k_selected'])
            stages.append(Stage(
                name=name, phase='post_freeze', arm=arm, k=k, seed=seed, output=str(output),
                argv=_stage_argv(artifact=artifact, targets=targets, canonical=canonical,
                                 output=output, seed=seed, arm=arm, k=k,
                                 v3_state=state_paths[k]),
                v3_state=state_paths[k]))
    order = (tuple(f'fit_v3_state:K{k}' for k in K_GRID)
             + tuple(stage.name for stage in stages[:9])
             + ('k_selection',)
             + tuple(stage.name for stage in stages[9:])
             + ('screen',))
    return Plan(output_root=str(root), v3_state_paths=state_paths, stages=tuple(stages),
                k_selection_path=str(root / K_SELECTION_FILENAME),
                k_selection_sha256=None if frozen is None else k_selection_sha256(frozen),
                order=order)


_ARM_FLAGS = {'A': (False, False), 'B': (True, False), 'C': (True, True)}  # (ple, absence)
_SPLIT_KEYS = frozenset(('train', 'dev', 'validation'))
_SELECTED_DEV_KEYS = frozenset(('epoch', 'epoch_index', 'metric', 'metric_value',
                                'prediction_sha256', 'sample_ids_sha256'))


def _policy_check(binding, expected) -> None:
    for key, value in expected.items():
        if binding.get(key) != value:
            raise ValueError(f'study policy mismatch for {key}: expected {value!r}, '
                             f'got {binding.get(key)!r}')


def validate_v3_binding(binding, stage, *, screen_record=None, k_selection=None,
                        study_binding=None) -> None:
    """Validate one stage `binding.json` against the locked plan and the arm table.

    Refuses arm drift (arm, K, state path, active blocks, depth, block, width, direction),
    protocol drift (budget, seeds, folds, final_eval), any validation/test trace, screen
    reuse for selection (dev hash equal to the screen hash, or a screen split recorded), and,
    for post-freeze stages, a missing/mismatching `k_selection_sha256` or K (F17).
    """
    from . import cei_pilot as pilot

    if not isinstance(binding, dict):
        raise ValueError('binding must be a dict')
    for field_name in pilot._REQUIRED_BINDINGS:
        if field_name not in binding or binding[field_name] is None:
            raise ValueError(f'binding missing required field: {field_name}')
    train_limit, dev_limit, epochs = FULL_BUDGET
    _policy_check(binding, {
        'method': STUDY_METHOD, 'train_limit': train_limit, 'dev_limit': dev_limit,
        'epochs': epochs, 'patience': PATIENCE, 'seed': stage.seed, 'sample_seed': SAMPLE_SEED,
        'selection_fold': 'dev', 'final_eval': 'none', 'test_evaluated': False,
        'weights': 'sqrt_inverse', 'edges': 'all', 'edge_direction': 'forward',
        'top_k_labels': TOP_K_LABELS, 'num_classes': NUM_CLASSES, 'hidden': 128,
    })
    if 'selected_validation' in binding:
        raise ValueError('binding carries selected_validation: validation was scored')
    splits = binding.get('split_sample_ids_sha256')
    counts = binding.get('counts', {})
    if not isinstance(splits, dict) or set(splits) != _SPLIT_KEYS or set(counts) - _SPLIT_KEYS:
        raise ValueError('binding split keys differ from [dev, train, validation]; the screen '
                         'and test folds must never appear in a stage binding')
    selected = binding.get('selected_dev')
    if not isinstance(selected, dict) or set(selected) != _SELECTED_DEV_KEYS:
        raise ValueError('binding selected_dev is missing or differs from the six-field contract')
    if selected['metric'] != 'macro_f1' or selected['sample_ids_sha256'] != splits['dev']:
        raise ValueError('binding selected_dev must be the dev macro_f1 checkpoint proof')
    if screen_record is not None:
        screen_hash = screen_record.get('screen_sample_ids_sha256')
        if screen_hash in (splits['dev'], splits['train'], selected['sample_ids_sha256']):
            raise ValueError('screen fold reused for selection or training: a stage split hash '
                             'equals the screen sample-id hash')
    config = binding.get('method_config')
    if not isinstance(config, dict) or config.get('method') != STUDY_METHOD:
        raise ValueError('binding method_config is not a cei_gnn_v3 run_config')
    settings = config.get('effective_settings')
    if not isinstance(settings, dict):
        raise ValueError('binding method_config lacks effective_settings')
    if config.get('arm') != stage.arm or settings.get('arm') != stage.arm:
        raise ValueError(f'arm drift: binding arm {config.get("arm")!r}/{settings.get("arm")!r} '
                         f'differs from the planned arm {stage.arm!r}')
    architecture = config.get('architecture', {})
    if not (config.get('k') == stage.k == settings.get('k') == architecture.get('k')):
        raise ValueError(f'k drift: binding k {config.get("k")!r} differs from the planned '
                         f'K {stage.k!r}')
    if settings.get('v3_state') != stage.v3_state or config.get('v3_state_path') != stage.v3_state:
        raise ValueError('v3_state drift: binding v3_state path differs from the planned state file')
    ple_active, absence_active = _ARM_FLAGS[stage.arm]
    if config.get('ple_active') != ple_active:
        raise ValueError(f'ple_active {config.get("ple_active")!r} differs from arm {stage.arm}')
    if config.get('absence_active') != absence_active:
        raise ValueError(f'absence_active {config.get("absence_active")!r} differs from arm '
                         f'{stage.arm}')
    for key, value in (('encoder_depth', 1), ('comorbid_block', 0), ('hidden', 128),
                       ('edge_direction', 'forward'), ('pair_mode', 'additive')):
        if config.get(key) != value:
            raise ValueError(f'arm drift: method_config {key} {config.get(key)!r} != {value!r}')
    if config.get('preprocessing_sha256') != binding['preprocessing_sha256']:
        raise ValueError('preprocessing_sha256 of the v3 state differs from the stage binding '
                         '(v3 §12.18 post-run check)')
    if stage.phase == 'post_freeze':
        if k_selection is None:
            raise ValueError('post-freeze stage needs the frozen k_selection record and its '
                             'k_selection_sha256')
        _validate_k_selection_record(k_selection)
        if study_binding is None:
            raise ValueError(f'{stage.name}: no study binding carrying k_selection_sha256; A/B '
                             'stages must bind the K freeze (v3 §12.17)')
        expected_hash = k_selection_sha256(k_selection)
        if study_binding.get('k_selection_sha256') != expected_hash:
            raise ValueError('study binding k_selection_sha256 differs from the frozen record')
        if not (study_binding.get('k_selected') == k_selection['k_selected'] == stage.k):
            raise ValueError(f'frozen K mismatch: study binding K {study_binding.get("k_selected")!r}'
                             f', record K {k_selection["k_selected"]!r}, stage K {stage.k!r}')
        binding_path = Path(stage.output) / 'binding.json'
        if not binding_path.is_file():
            raise ValueError(f'{stage.name}: binding.json is missing at {binding_path}')
        digest = hashlib.sha256(binding_path.read_bytes()).hexdigest()
        if study_binding.get('binding_sha256') != digest:
            raise ValueError('study binding binding_sha256 differs from the binding.json on disk')
        if json.loads(binding_path.read_bytes().decode('utf-8')) != binding:
            raise ValueError('binding differs from binding.json on disk')
    return None


def check_stage_result(stage_dir, binding, result) -> None:
    """Refuse any stage whose result carries validation/test traces (EXT §6, E4)."""
    stage_dir = Path(stage_dir)
    if result.get('status') != 'completed':
        raise ValueError('stage result is not a completed run')
    if result.get('binding') != binding:
        raise ValueError('stage result binding differs from binding.json')
    if binding.get('selection_fold') != 'dev' or binding.get('final_eval') != 'none':
        raise ValueError("stage binding must be selection_fold='dev', final_eval='none'")
    if 'selected_validation' in binding:
        raise ValueError('stage binding carries selected_validation: validation was scored')
    if result.get('metrics') is not None or result.get('per_class') is not None:
        raise ValueError('stage result carries validation metrics; final_eval must be none')
    if not isinstance(result.get('dev_metrics'), dict):
        raise ValueError('stage result lacks dev_metrics')
    if result.get('validation_evaluations', 0) != 0:
        raise ValueError('stage result records a validation evaluation')
    if result.get('test_evaluated', False) is not False or binding.get('test_evaluated') is not False:
        raise ValueError('stage evaluated the test fold')
    for name in ('validation.npz', 'test.npz'):
        if (stage_dir / name).exists():
            raise ValueError(f'stage output contains {name}; validation/test were scored')
    return None


def write_study_binding(stage_dir, stage, k_selection) -> dict:
    """Bind an A/B stage to the K freeze (F17): k_grid, k_selected, rule, freeze hash."""
    _validate_k_selection_record(k_selection)
    stage_dir = Path(stage_dir)
    binding_path = stage_dir / 'binding.json'
    if not binding_path.is_file():
        raise ValueError(f'{stage.name}: binding.json is missing at {binding_path}')
    document = {
        'stage': stage.name, 'arm': stage.arm, 'seed': int(stage.seed), 'k': stage.k,
        'k_grid': list(K_GRID), 'k_selected': int(k_selection['k_selected']),
        'k_selection_rule': K_SELECTION_RULE,
        'k_selection_sha256': k_selection_sha256(k_selection),
        'binding_sha256': hashlib.sha256(binding_path.read_bytes()).hexdigest(),
        'v3_state': stage.v3_state, 'selection_fold': 'dev', 'final_eval': 'none',
    }
    _write_new_json(stage_dir / STUDY_BINDING_FILENAME, document)
    return document


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
    """Default reader (NOT exercised by tests; opens the artifact, G3 only).

    Re-runs `load_targets -> select_top_labels -> sample_train_ids -> fit_preprocessing`
    exactly as `train.py` lines 593–618 do (F18), then streams the TRAIN-sample graphs once
    more to collect, per measurement/vital identity, (i) the per-node values exactly as
    `Scaler.fit` sees them and `Scaler.transform` emits them (float32 z-score or signed-log
    fallback, F14/F15) and (ii) the number of distinct graphs containing the identity (F10).
    Dev, screen, validation and test rows are never read as graphs.
    """
    from ...core import train as train_module
    from ...core.contracts import (
        VISIT_MEMBERSHIP_FILENAME, iter_graphs_with_membership)
    from ...core.tensorize import (
        fit_preprocessing, node_token, preprocessing_state)

    artifact = Path(artifact)
    graphs_path = artifact / 'graphs.jsonl'
    membership_path = artifact / VISIT_MEMBERSHIP_FILENAME
    targets = train_module.load_targets(targets_path)
    targets, _kept, _dropped = train_module.select_top_labels(targets, top_k_labels)
    train_ids = train_module.sample_train_ids(targets, train_limit, sample_seed)
    prep = fit_preprocessing(graphs_path, train_ids, token_min_count,
                             membership_path=membership_path)
    state = preprocessing_state(prep)
    preprocessing_sha256 = hashlib.sha256(_canonical_file_bytes(state)).hexdigest()
    scaler, vocabulary = prep['scaler'], prep['vocabulary']
    values: Dict[str, List[float]] = {}
    graphs_per_item: Dict[str, int] = {}
    for graph, _membership in iter_graphs_with_membership(graphs_path, membership_path):
        if graph['sample_id'] not in train_ids:
            continue
        seen = set()
        for node in graph['nodes']:
            if node['kind'] not in ('measurement', 'vital'):
                continue
            token = node_token(node)
            seen.add(token)
            value, has_value = scaler.transform(token, node.get('value'))
            if has_value:
                values.setdefault(token, []).append(value)
        for token in seen:
            graphs_per_item[token] = graphs_per_item.get(token, 0) + 1
    transform = {token: ('zscore' if token in scaler.stats else 'signed_log')
                 for token in values}
    return V3FitInputs(
        values_by_item={token: np.asarray(v, dtype=np.float32) for token, v in values.items()},
        transform_by_item=transform, identity_graph_counts=graphs_per_item,
        vocabulary=dict(vocabulary.index), vocabulary_tokens=tuple(vocabulary.tokens),
        token_min_count=int(prep['token_min_count']), preprocessing_sha256=preprocessing_sha256,
        node_feature_layout=tuple(state['node_feature_layout']), train_count=len(train_ids))


def fit_v3_state(output_root, k, *, reader, min_values=20, min_graphs=20) -> Path:
    """Fit and write `<output_root>/v3_state/K<k>.json` (F18) from `reader()`'s inputs.

    The document is exactly `plugin_cei_gnn_v3.build_v3_state(...)`; the file is written once
    (never overwritten) with the canonical study layout so `load_v3_state` hashes it stably.
    """
    from .cei_v3_absence import fit_universe
    from .cei_v3_ple import fit_knots
    from ...methods.plugin_cei_gnn_v3 import (
        build_v3_state, load_v3_state)

    if isinstance(k, bool) or k not in K_GRID:
        raise ValueError(f'K {k!r} is outside the frozen grid {list(K_GRID)} (v3 §11.2 item 1)')
    path = v3_state_path(output_root, k)
    if path.exists():
        raise FileExistsError(f'Refusing occupied v3 state file {path}')
    inputs = reader()
    table = fit_knots(inputs.values_by_item, int(k), min_values=min_values,
                      transform_by_item=inputs.transform_by_item,
                      token_min_count=int(inputs.token_min_count))
    universe = fit_universe(inputs.identity_graph_counts, inputs.vocabulary,
                            min_graphs=min_graphs, token_min_count=int(inputs.token_min_count))
    document = build_v3_state(K=int(k), knot_table=table, universe=universe,
                              vocabulary_tokens=inputs.vocabulary_tokens,
                              vocabulary_min_count=inputs.token_min_count,
                              preprocessing_sha256=inputs.preprocessing_sha256,
                              node_feature_layout=inputs.node_feature_layout)
    _write_new_json(path, document)
    load_v3_state(path)   # the adapter must accept exactly what was written
    return path


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


def _grid_stage_names() -> List[str]:
    return [f'C_K{k}_seed{seed}' for k in K_GRID for seed in SEEDS]


def _seed_mean(values) -> float:
    return float(np.mean([float(v) for v in values]))


def _argmax_with_tie_rule(seed_means) -> Tuple[int, bool]:
    """Winner over the grid; equality of the runner's 6-decimal metric -> smaller K."""
    rounded = {k: round(seed_means[k], 6) for k in K_GRID}
    best = max(rounded.values())
    winners = [k for k in K_GRID if rounded[k] == best]   # K_GRID is ascending
    return winners[0], len(winners) > 1


def select_k(bindings) -> KSelection:
    """K selection statistic and winner from the nine C `binding.json` dicts (v3 §11.2).

    `bindings` maps stage name (`C_K<k>_seed<seed>`) -> binding. Exactly the nine grid
    stages are required; each must be a conforming arm-C stage at its K, share the dev
    sample-id hash, and agree on the knot-table / v3-state hash of its K.
    """
    if not isinstance(bindings, dict):
        raise ValueError('bindings must map stage name -> binding dict')
    expected = _grid_stage_names()
    missing = [name for name in expected if name not in bindings]
    if missing:
        raise ValueError(f'K selection needs all nine C grid stages; missing: {missing}')
    extra = sorted(set(bindings) - set(expected))
    if extra:
        raise ValueError(f'K selection accepts the nine C grid stages only; extra: {extra}')
    statistic: Dict[Tuple[int, int], float] = {}
    knot_hashes: Dict[int, str] = {}
    state_hashes: Dict[int, str] = {}
    dev_hash = None
    for k in K_GRID:
        for seed in SEEDS:
            name = f'C_K{k}_seed{seed}'
            binding = bindings[name]
            stage = Stage(name=name, phase='c_grid', arm='C', k=k, seed=seed, output='',
                          argv=(), v3_state=binding.get('method_config', {})
                          .get('effective_settings', {}).get('v3_state', ''))
            try:
                validate_v3_binding(binding, stage)
            except ValueError as error:
                raise ValueError(f'{name}: {error}') from error
            if not stage.v3_state.endswith(f'/K{k}.json'):
                raise ValueError(f'{name}: v3_state path {stage.v3_state!r} is not the K{k} state')
            config = binding['method_config']
            for key, store in (('knot_table_sha256', knot_hashes), ('v3_state_sha256', state_hashes)):
                value = config.get(key)
                if not isinstance(value, str) or len(value) != 64:
                    raise ValueError(f'{name}: method_config lacks a hex {key}')
                if store.setdefault(k, value) != value:
                    raise ValueError(f'{name}: {key} differs between the seeds of K{k}; the '
                                     'three stages did not bind the same state file')
            binding_dev = binding['split_sample_ids_sha256']['dev']
            if dev_hash is None:
                dev_hash = binding_dev
            elif binding_dev != dev_hash:
                raise ValueError(f'{name}: dev sample-id hash differs across grid stages; K '
                                 'selection needs one shared OLD-dev split')
            value = binding['selected_dev']['metric_value']
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f'{name}: selected_dev.metric_value is not a finite number')
            statistic[(k, seed)] = float(value)
    seed_means = {k: _seed_mean(statistic[(k, seed)] for seed in SEEDS) for k in K_GRID}
    k_selected, tie = _argmax_with_tie_rule(seed_means)
    return KSelection(k_grid=K_GRID, seeds=SEEDS, arm='C', statistic=statistic,
                      seed_means=seed_means, k_selected=k_selected, tie_rule_applied=tie,
                      knot_table_sha256_by_k=knot_hashes, v3_state_sha256_by_k=state_hashes,
                      dev_sample_ids_sha256=str(dev_hash))


def k_selection_sha256(record) -> str:
    """SHA-256 of the `k_selection.json` bytes (`write_k_selection` writes exactly them)."""
    return hashlib.sha256(_canonical_file_bytes(record)).hexdigest()


def _file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stage_entries(selection, stage_dirs) -> List[dict]:
    entries = []
    for k in K_GRID:
        for seed in SEEDS:
            name = f'C_K{k}_seed{seed}'
            if name not in stage_dirs:
                raise ValueError(f'stage directory missing for {name}')
            stage_dir = Path(stage_dirs[name])
            binding_path, checkpoint = stage_dir / 'binding.json', stage_dir / 'best.pt'
            if not binding_path.is_file() or not checkpoint.is_file():
                raise ValueError(f'{name}: binding.json or best.pt missing in {stage_dir}')
            entries.append({
                'stage': name, 'k': int(k), 'seed': int(seed), 'output': str(stage_dir),
                'checkpoint_sha256': _file_sha256(checkpoint),
                'binding_sha256': _file_sha256(binding_path),
                'metric': 'macro_f1', 'metric_value': float(selection.statistic[(k, seed)])})
    return entries


def _k_selection_document(selection, entries) -> dict:
    return {
        'version': K_SELECTION_VERSION,
        'k_grid': [int(k) for k in K_GRID], 'seeds': [int(s) for s in SEEDS], 'arm': 'C',
        'rule': K_SELECTION_RULE,
        'statistic': 'unweighted mean over seeds of selected_dev.metric_value',
        'tie_rule': 'equal 6-decimal seed-mean -> smaller K',
        'stages': entries,
        'seed_means': {str(k): float(selection.seed_means[k]) for k in K_GRID},
        'seed_means_rounded': {str(k): round(float(selection.seed_means[k]), 6) for k in K_GRID},
        'k_selected': int(selection.k_selected),
        'tie_rule_applied': bool(selection.tie_rule_applied),
        'knot_table_sha256': {str(k): selection.knot_table_sha256_by_k[k] for k in K_GRID},
        'v3_state_sha256': {str(k): selection.v3_state_sha256_by_k[k] for k in K_GRID},
        'selected_knot_table_sha256': selection.knot_table_sha256_by_k[selection.k_selected],
        'selected_v3_state_sha256': selection.v3_state_sha256_by_k[selection.k_selected],
        'dev_sample_ids_sha256': selection.dev_sample_ids_sha256,
        'control_binding_sha256': [entry['binding_sha256'] for entry in entries
                                   if entry['k'] == selection.k_selected],
        'k_selection_completed_before_screen': True,
        'screen_record_sha256': None,   # successor record fills this (F17)
    }


def write_k_selection(selection, stage_dirs, path) -> dict:
    """Write the K-freeze record once (v3 §11.2 item 7, §12.17, §12.21).

    Every stage directory's `binding.json` must reproduce the selection (its metric value
    and arm/K), so the record can never disagree with the bindings it was computed from.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing occupied K-freeze record {path}')
    entries = _stage_entries(selection, stage_dirs)
    bindings = {entry['stage']: _load_json(Path(entry['output']) / 'binding.json', 'binding.json')
                for entry in entries}
    try:
        recomputed = select_k(bindings)
    except ValueError as error:
        raise ValueError(f'binding.json files on disk do not reproduce the K selection: '
                         f'{error}') from error
    if recomputed != selection:
        raise ValueError('binding.json files on disk do not reproduce the given K selection')
    document = _k_selection_document(selection, entries)
    _write_new_json(path, document)
    return document


def assert_k_selection_replay(record, stage_dirs) -> None:
    """Recompute the K selection from the stage bindings on disk; raise on any drift."""
    if not isinstance(record, dict):
        raise ValueError('k_selection record must be a dict')
    _validate_k_selection_record(record)
    stages = record.get('stages')
    if not isinstance(stages, list) or [s.get('stage') for s in stages] != _grid_stage_names():
        raise ValueError('k_selection stages differ from the nine C grid stages')
    bindings = {}
    for entry in stages:
        name = entry['stage']
        stage_dir = Path(stage_dirs[name]) if name in stage_dirs else Path(entry['output'])
        binding_path, checkpoint = stage_dir / 'binding.json', stage_dir / 'best.pt'
        if not binding_path.is_file() or not checkpoint.is_file():
            raise ValueError(f'{name}: binding.json or best.pt missing in {stage_dir}')
        if entry.get('checkpoint_sha256') != _file_sha256(checkpoint):
            raise ValueError(f'{name}: checkpoint_sha256 differs from best.pt on disk')
        if entry.get('binding_sha256') != _file_sha256(binding_path):
            raise ValueError(f'{name}: binding_sha256 differs from binding.json on disk')
        bindings[name] = _load_json(binding_path, 'binding.json')
    selection = select_k(bindings)
    for entry in stages:
        expected = selection.statistic[(entry['k'], entry['seed'])]
        if entry.get('metric') != 'macro_f1' or entry.get('metric_value') != expected:
            raise ValueError(f"{entry['stage']}: metric_value differs from the binding on disk")
    if record.get('k_selected') != selection.k_selected:
        raise ValueError(f'k_selection winner {record.get("k_selected")!r} differs from the '
                         f'recomputed argmax {selection.k_selected}')
    if record.get('tie_rule_applied') is not selection.tie_rule_applied:
        raise ValueError('k_selection tie_rule_applied differs from the recomputed selection')
    expected_document = _k_selection_document(selection, _stage_entries(selection, {
        name: (Path(stage_dirs[name]) if name in stage_dirs else Path(entry['output']))
        for name, entry in ((s['stage'], s) for s in stages)}))
    for key in ('seeds', 'arm', 'statistic', 'tie_rule', 'seed_means', 'seed_means_rounded',
                'knot_table_sha256', 'v3_state_sha256', 'selected_knot_table_sha256',
                'selected_v3_state_sha256', 'dev_sample_ids_sha256', 'control_binding_sha256',
                'k_selection_completed_before_screen'):
        if record.get(key) != expected_document[key]:
            raise ValueError(f'k_selection {key} differs from the recomputed selection')
    return None


# ----------------------------------------------------------- metrics, bootstrap


def weighted_macro_f1(y, pred, weights=None, *, num_classes=NUM_CLASSES) -> float:
    """Fixed-label macro-F1 with optional row weights; re-implements
    `cei_v2_study.weighted_macro_f1` (lines 652–661) bit for bit (v3 §12.20 F20)."""
    y, pred = np.asarray(y, dtype=int), np.asarray(pred, dtype=int)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    confusion = np.bincount(y * num_classes + pred, weights=w,
                            minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    tp = np.diag(confusion)
    fp, fn = confusion.sum(0) - tp, confusion.sum(1) - tp
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros(num_classes), where=denominator > 0)
    return float(f1.mean())


def _paired_rows(arms, contrasts, seeds):
    """Check every arm-seed carries the same (y, subjects) rows; return (y, subjects, preds)."""
    if not isinstance(arms, dict) or not arms:
        raise ValueError('arms must map arm name -> {seed: (y, pred, subjects)}')
    for first, second in contrasts:
        for name in (first, second):
            if name not in arms:
                raise ValueError(f'contrast names an unknown arm {name!r}')
    reference = None
    predictions = {}
    for arm_name, by_seed in arms.items():
        for seed in seeds:
            if seed not in by_seed:
                raise ValueError(f'arm {arm_name!r} lacks seed {seed}')
            y, pred, subjects = by_seed[seed]
            y = np.asarray(y, dtype=int)
            pred = np.asarray(pred, dtype=int)
            subjects = np.asarray(subjects).astype(str)
            if y.ndim != 1 or pred.shape != y.shape or subjects.shape != y.shape or not len(y):
                raise ValueError(f'arm {arm_name!r} seed {seed}: y, pred, subjects must be aligned '
                                 'nonempty 1-D arrays')
            if reference is None:
                reference = (y, subjects)
            elif not (np.array_equal(y, reference[0]) and np.array_equal(subjects, reference[1])):
                raise ValueError(f'arm {arm_name!r} seed {seed}: rows are not paired with the '
                                 'other arm-seeds (y or subjects differ)')
            predictions[(arm_name, seed)] = pred
    return reference[0], reference[1], predictions


def paired_bootstrap(arms, contrasts, *, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED,
                     seeds=SEEDS, num_classes=NUM_CLASSES):
    """Paired patient-cluster bootstrap of seed-averaged macro-F1 deltas (v3 §6.2 cond. 3).

    `arms[arm][seed] = (y, pred, subjects)`; every arm-seed must carry identical `y` and
    `subjects`. Returns `{contrast: float64[resamples]}` of `mean_seeds(F1_first − F1_second)`
    per resample, drawn exactly as `cei_v2_study.paired_bootstrap` (lines 681–694): distinct
    patients with replacement, one draw shared by every arm and seed. Bound rules (linear
    quantile here, order statistics in the extension family) are applied by the caller.
    """
    contrasts = [tuple(contrast) for contrast in contrasts]
    if any(len(contrast) != 2 for contrast in contrasts):
        raise ValueError('each contrast must be a (treatment, control) pair')
    y, subjects, predictions = _paired_rows(arms, contrasts, seeds)
    patients, inverse = np.unique(subjects, return_inverse=True)
    rng = np.random.default_rng(seed)
    samples = {contrast: [] for contrast in contrasts}
    for _ in range(int(resamples)):
        drawn = rng.integers(0, len(patients), len(patients))
        weights = np.bincount(drawn, minlength=len(patients))[inverse].astype(float)
        scores = {key: weighted_macro_f1(y, pred, weights, num_classes=num_classes)
                  for key, pred in predictions.items()}
        for first, second in contrasts:
            samples[(first, second)].append(float(np.mean(
                [scores[(first, s)] - scores[(second, s)] for s in seeds])))
    return {contrast: np.asarray(values, dtype=np.float64) for contrast, values in samples.items()}


def decide_v3(deltas, scores, *, treatment='C', control='A', seeds=SEEDS,
              resamples=BOOTSTRAP_RESAMPLES, bootstrap_seed=BOOTSTRAP_SEED) -> dict:
    """The pre-registered C-vs-A rule (v3 §6.2, F20). `scores[arm][seed]` = screen macro-F1;
    `deltas[(treatment, control)]` = the 1,000 averaged bootstrap deltas."""
    key = (treatment, control)
    if key not in deltas:
        raise ValueError(f'deltas lack the contrast {key}')
    values = np.asarray(deltas[key], dtype=np.float64)
    if values.ndim != 1 or values.shape[0] != resamples:
        raise ValueError(f'the decision requires exactly {resamples} averaged deltas, '
                         f'got shape {values.shape}')
    if not np.isfinite(values).all():
        raise ValueError('bootstrap deltas must be finite')
    for arm in key:
        if arm not in scores:
            raise ValueError(f'scores lack arm {arm!r}')
        for seed in seeds:
            if seed not in scores[arm]:
                raise ValueError(f'scores for arm {arm!r} lack seed {seed}')
    per_seed = {str(seed): {arm: float(scores[arm][seed]) for arm in key} for seed in seeds}
    means = {arm: float(np.mean([per_seed[str(seed)][arm] for seed in seeds])) for arm in key}
    lower = float(np.quantile(values, 0.025, method=QUANTILE_METHOD))
    upper = float(np.quantile(values, 0.975, method=QUANTILE_METHOD))
    checks = {
        f'{treatment.lower()}_beats_{control.lower()}_each_seed': all(
            per_seed[str(seed)][treatment] > per_seed[str(seed)][control] for seed in seeds),
        f'{treatment.lower()}_mean_above_{control.lower()}': means[treatment] > means[control],
        f'{treatment.lower()}_minus_{control.lower()}_lower_bound_above_zero': lower > 0,
    }
    wins = all(checks.values())
    return {
        'contrast': [treatment, control], 'metric': 'weighted_macro_f1',
        'per_seed': per_seed, 'seed_means': means,
        'point_delta': means[treatment] - means[control],
        'interval_95': [lower, upper], 'quantile_method': QUANTILE_METHOD,
        'resamples': int(resamples), 'bootstrap_seed': int(bootstrap_seed),
        'checks': checks, 'v3_beats_control': wins,
        'statement': ('v3 beats the v2 additive control on this screen' if wins
                      else 'benefit not demonstrated on this screen'),
        'validation_evaluated': False, 'test_evaluated': False,
    }


def absence_share(parts, *, batch_index, edge_index, graph_count) -> np.ndarray:
    """Secondary metric (v3 §6.2): per graph, mean over classes of
    `sum|absence| / sum|node + edge + pair + absence|`, 0 where the denominator is 0."""
    import torch

    graph_count = int(graph_count)
    batch_index = torch.as_tensor(batch_index, dtype=torch.long).view(-1)
    classes = int(parts['node_contributions'].size(1))

    def per_graph(values, owner):
        total = torch.zeros((graph_count, classes), dtype=torch.float64)
        if values.numel():
            total.index_add_(0, owner.long(), values.detach().abs().to(torch.float64))
        return total

    node = per_graph(parts['node_contributions'], batch_index)
    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    edge = per_graph(parts['edge_contributions'], batch_index[edge_index[0]]
                     if edge_index.numel() else edge_index.new_zeros((0,)))
    pairs = parts['pairs']
    pair = per_graph(parts['pair_contributions'], batch_index[pairs[0]]
                     if pairs.numel() else pairs.new_zeros((0,)))
    absence = per_graph(parts['absence_contributions'], parts['absence_items'][0])
    denominator = node + edge + pair + absence
    share = torch.where(denominator > 0, absence / denominator.clamp(min=1e-300),
                        torch.zeros_like(denominator))
    return share.mean(dim=1).numpy()


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
    """Historical JSON loads unchanged via ``ScreenResult(**old_json)``.

    ``absence_share_mean`` ALWAYS means the row (visit) mean, including in new
    results; it is retained for legacy consumers, never relabelled patient mean.
    Missing new fields default to None and the legacy aggregation version: no
    equal-patient statistic or shares proof is inferred from a historical mean.
    New absence-active results report both means and bind the exact shares.npz
    FILE bytes (not only the share array) with ``shares_sha256``. Absence-inactive
    arms keep all means/proof fields None. All shares files remain private outputs.
    """
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
    absence_share_row_mean: Optional[float] = None
    absence_share_patient_mean: Optional[float] = None
    absence_share_aggregation_version: str = ABSENCE_AGGREGATION_LEGACY
    shares_path: Optional[str] = None
    shares_sha256: Optional[str] = None


def absence_share_means(shares, subjects) -> Tuple[float, float]:
    """Return visit mean and equal-patient mean, without reading data or writing.

    Each subject first receives the mean of its visits; these subject means then
    receive equal weight. The per-visit class-share arithmetic is unchanged.
    """
    shares = np.asarray(shares, dtype=np.float64)
    subjects = np.asarray(subjects)
    if (shares.ndim != 1 or not len(shares) or subjects.shape != shares.shape
            or subjects.dtype.kind not in ('U', 'S')):
        raise ValueError('absence shares and subjects must be aligned nonempty 1-D arrays '
                         'with string subject identities')
    if not np.isfinite(shares).all() or np.any((shares < 0) | (shares > 1)):
        raise ValueError('absence shares must be finite values in [0, 1]')
    subjects = subjects.astype(str)
    if np.any(subjects == ''):
        raise ValueError('absence share subject identities must be nonempty')
    patients, inverse = np.unique(subjects, return_inverse=True)
    sums = np.bincount(inverse, weights=shares, minlength=len(patients))
    counts = np.bincount(inverse, minlength=len(patients))
    return float(shares.mean()), float((sums / counts).mean())


def assert_absence_share_replay(result, *, shares_path=None, logits_path=None) -> None:
    """Validate an optional shares proof, never generating or repairing evidence.

    Accepts a ScreenResult or its JSON dict. Legacy results without proof/new
    aggregates remain readable without opening arrays. A present proof requires
    complete v2 metadata, a matching FILE hash, aligned identities with hashed
    logits, and exact reproduction of both aggregates and the legacy row alias.
    Explicit path overrides are external I/O mappings for relocated immutable
    outputs; embedded provenance and hashes are never rewritten. This is a replay
    check, not a plan-mode operation (plan/help must not call array readers).
    """
    if isinstance(result, ScreenResult):
        result = asdict(result)
    if not isinstance(result, dict):
        raise ValueError('screen result must be a ScreenResult or JSON dict')
    version = result.get('absence_share_aggregation_version', ABSENCE_AGGREGATION_LEGACY)
    if version not in (ABSENCE_AGGREGATION_LEGACY, ABSENCE_AGGREGATION_VERSION):
        raise ValueError('unknown absence share aggregation version')
    fields = ('absence_share_row_mean', 'absence_share_patient_mean',
              'shares_path', 'shares_sha256')
    if all(result.get(key) is None for key in fields):
        if version == ABSENCE_AGGREGATION_LEGACY:
            return None
        if result.get('arm') in ('A', 'B') and result.get('absence_share_mean') is None:
            return None
        raise ValueError('new absence share result lacks its shares proof and aggregates')
    if version != ABSENCE_AGGREGATION_VERSION or any(result.get(key) is None for key in fields):
        raise ValueError('incomplete absence share proof or aggregation metadata')
    proof_path = Path(shares_path if shares_path is not None else result['shares_path'])
    digest = result['shares_sha256']
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('shares_sha256 must be a lowercase SHA-256 hex digest')
    if not proof_path.is_file() or _file_sha256(proof_path) != digest:
        raise ValueError('shares proof file missing or shares_sha256 differs')
    with np.load(proof_path, allow_pickle=False) as saved:
        if set(saved.files) != {'absence_share', 'subjects', 'sample_ids'}:
            raise ValueError('shares proof keys differ from the absence proof contract')
        shares, subjects, ids = (saved[key].copy()
                                for key in ('absence_share', 'subjects', 'sample_ids'))
    if shares.dtype != np.float64:
        raise ValueError('shares proof must preserve the float64 per-visit absence shares')
    row_mean, patient_mean = absence_share_means(shares, subjects)
    if (ids.shape != shares.shape or ids.dtype.kind not in ('U', 'S')
            or len(shares) != result.get('row_count') or np.any(ids.astype(str) == '')
            or len(np.unique(ids)) != len(ids)):
        raise ValueError('shares proof row count or sample identities differ')
    for key, expected in (('absence_share_mean', row_mean),
                          ('absence_share_row_mean', row_mean),
                          ('absence_share_patient_mean', patient_mean)):
        if result.get(key) != expected:
            raise ValueError(f'{key} differs from the shares proof aggregation')
    source_path = logits_path if logits_path is not None else result.get('logits_path')
    if source_path is None:
        raise ValueError('shares replay requires the bound logits path')
    with np.load(source_path, allow_pickle=False) as saved:
        logits = saved['logits']
        if (logits.dtype != np.float32 or logits.ndim != 2 or logits.shape[0] != len(shares)
                or not np.isfinite(logits).all()
                or not np.array_equal(subjects, saved['subjects'])
                or not np.array_equal(ids, saved['sample_ids'])):
            raise ValueError('shares proof identities differ from the bound logits rows')
        if hashlib.sha256(np.ascontiguousarray(logits).tobytes()).hexdigest() != result.get('logits_sha256'):
            raise ValueError('shares replay logits differ from the bound logits hash')
    return None


def load_screen_result(path, *, shares_path=None, logits_path=None) -> ScreenResult:
    """Read historical/new JSON and validate any shares proof, read-only.

    For relocated files, callers explicitly supply mapped array paths; do not edit
    historical JSON. A historical row mean alone cannot reproduce a patient mean.
    """
    result = ScreenResult(**_load_json(path, 'screen_result.json'))
    assert_absence_share_replay(result, shares_path=shares_path, logits_path=logits_path)
    return result


def screen_record_sha256(record) -> str:
    """Canonical SHA-256 of a U4 screen record (sorted compact JSON)."""
    payload = json.dumps(record, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _default_model_factory(binding, checkpoint_path):
    """Rebuild the bound cei_gnn_v3 adapter and load its dev-selected checkpoint."""
    import torch
    from types import SimpleNamespace
    from ...methods import build_method

    config = binding['method_config']
    architecture = config['architecture']
    args = dict(binding)
    args['method_options'] = dict(config['effective_settings'])
    model = build_method(
        STUDY_METHOD, num_tokens=binding['vocabulary_size'], node_dim=binding['node_dim'],
        edge_dim=binding['edge_dim'], num_classes=binding['num_classes'],
        hidden=binding['hidden'], layers=binding['layers'], dropout=binding['dropout'],
        token_dim=architecture['token_dim'], num_triples=binding['num_meta_relations'],
        args=SimpleNamespace(**args))
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True),
                          strict=True)
    if sum(p.numel() for p in model.parameters()) != binding['parameter_count']:
        raise ValueError('rebuilt parameter count differs from the checkpoint binding')
    return model


def _stage_of_binding(stage_dir, binding, study_binding) -> Stage:
    config = binding['method_config']
    return Stage(name=str(study_binding['stage']), phase='post_freeze', arm=str(config['arm']),
                 k=int(config['k']), seed=int(binding['seed']), output=str(stage_dir), argv=(),
                 v3_state=str(config['effective_settings']['v3_state']))


def score_screen(checkpoint, record, encoder, *, output_dir=None, model_factory=None,
                 batch_size=128) -> ScreenResult:
    """Score one frozen checkpoint once on the screen fold (v3 §5, §6.1, §11.2 item 8, EXT E7).

    Refuses, before any screen row is read: a missing K-freeze record; a checkpoint
    without a study binding carrying `k_selection_sha256`, or with a hash or K that differs
    from the freeze record; a binding/result with validation or test traces; an encoder
    fold other than `'screen'`; a screen record whose fold, row count, preprocessing hash or
    id hash disagree with the encoder ids or the checkpoint binding. Writes raw float32
    `logits.npz` and `proba.npz` (both hashed), plus a file-hashed `shares.npz`
    for absence-active arm C and `screen_result.json`, into `<stage_dir>/screen/`
    (or `output_dir`), never overwriting. New absence results explicitly carry
    the visit mean, equal-patient mean and aggregation version; the legacy mean
    remains a visit mean. No decisive metric or decision rule changes.
    """
    import torch
    from torch_geometric.loader import DataLoader

    stage_dir = Path(checkpoint.stage_dir)
    freeze_path = Path(checkpoint.k_selection_path)
    if not freeze_path.is_file():
        raise ValueError(f'no K-freeze record at {freeze_path}; the screen is never read '
                         'before K is frozen (v3 §11.2 item 8)')
    k_selection = _load_json(freeze_path, K_SELECTION_FILENAME)
    _validate_k_selection_record(k_selection)
    freeze_hash = _file_sha256(freeze_path)
    if freeze_hash != k_selection_sha256(k_selection):
        raise ValueError('k_selection.json bytes are not the canonical serialisation')
    binding_path, result_path = stage_dir / 'binding.json', stage_dir / 'result.json'
    checkpoint_path, study_path = stage_dir / 'best.pt', stage_dir / STUDY_BINDING_FILENAME
    for path in (binding_path, result_path, checkpoint_path):
        if not path.is_file():
            raise ValueError(f'checkpoint directory lacks {path.name}: {stage_dir}')
    if not study_path.is_file():
        raise ValueError(f'checkpoint binding lacks k_selection_sha256 (no {STUDY_BINDING_FILENAME} '
                         f'in {stage_dir}); only freeze-bound checkpoints are scored')
    binding = _load_json(binding_path, 'binding.json')
    result = _load_json(result_path, 'result.json')
    study_binding = _load_json(study_path, STUDY_BINDING_FILENAME)
    if study_binding.get('k_selection_sha256') != freeze_hash:
        raise ValueError('checkpoint k_selection_sha256 differs from the K-freeze record')
    k_selected = int(k_selection['k_selected'])
    stage = _stage_of_binding(stage_dir, binding, study_binding)
    if stage.k != k_selected or study_binding.get('k_selected') != k_selected:
        raise ValueError(f'checkpoint K {stage.k} differs from the frozen K* {k_selected}; '
                         'losing-K checkpoints are never scored on the screen')
    if stage.arm not in ARMS:
        raise ValueError(f'unknown arm {stage.arm!r}')
    check_stage_result(stage_dir, binding, result)
    if not isinstance(record, dict) or record.get('fold') != 'screen':
        raise ValueError("screen record must carry fold='screen'")
    validate_v3_binding(binding, stage, screen_record=record, k_selection=k_selection,
                        study_binding=study_binding)
    if encoder.fold != 'screen':
        raise ValueError(f"screen scoring encodes fold='screen' only, got {encoder.fold!r}; "
                         'validation and test are never scored here')
    ids = list(encoder.ids)
    if len(set(ids)) != len(ids):
        raise ValueError('screen ids contain duplicates')
    if record.get('row_count') != len(ids):
        raise ValueError(f'screen record row_count {record.get("row_count")!r} differs from the '
                         f'{len(ids)} encoder ids')
    if record.get('preprocessing_sha256') != binding['preprocessing_sha256']:
        raise ValueError('screen record preprocessing_sha256 differs from the checkpoint binding '
                         '(v3 §12.7)')
    if stage.arm == 'C':
        control_hashes = k_selection.get('control_binding_sha256', [])
        if _file_sha256(binding_path) not in control_hashes:
            raise ValueError('C checkpoint binding is not one of the frozen control bindings')
    out = Path(output_dir) if output_dir is not None else stage_dir / SCREEN_DIRNAME
    if out.exists():
        raise FileExistsError(f'Refusing occupied screen output {out}')
    record_hash = screen_record_sha256(record)

    factory = model_factory or _default_model_factory
    model = factory(binding, checkpoint_path)
    model.eval()
    logits_chunks, y_chunks, subjects, sample_ids, shares = [], [], [], [], []
    with torch.no_grad():   # one inference per checkpoint: logits and parts from the same pass
        for batch in DataLoader(list(encoder.rows()), batch_size=int(batch_size), shuffle=False):
            features = model.continuous_inputs(batch)
            parts = model.forward_continuous(features, batch.edge_index, batch, return_parts=True)
            logits_chunks.append(parts['logits'].detach().cpu().to(torch.float32))
            y_chunks.append(batch.y.view(-1).cpu())
            batch_subjects = batch.subject if isinstance(batch.subject, (list, tuple)) else [batch.subject]
            batch_ids = batch.sample_id if isinstance(batch.sample_id, (list, tuple)) else [batch.sample_id]
            subjects.extend(str(s) for s in batch_subjects)
            sample_ids.extend(str(s) for s in batch_ids)
            if stage.arm == 'C':
                shares.append(absence_share(parts, batch_index=batch.batch,
                                            edge_index=batch.edge_index,
                                            graph_count=int(batch.num_graphs)))
    if sample_ids != ids:
        raise ValueError('encoded screen rows differ from the bound screen ids (order or content)')
    logits = torch.cat(logits_chunks).numpy().astype(np.float32, copy=False)
    if not np.isfinite(logits).all():
        raise ValueError('non-finite screen logits')
    y = torch.cat(y_chunks).numpy()
    proba = torch.softmax(torch.from_numpy(logits), dim=1).numpy().astype(np.float32, copy=False)
    macro_f1 = weighted_macro_f1(y, logits.argmax(1), num_classes=binding['num_classes'])
    subjects_array = np.asarray(subjects).astype(str)
    ids_array = np.asarray(sample_ids).astype(str)
    share_array = np.concatenate(shares).astype(np.float64, copy=False) if shares else None
    row_mean, patient_mean = (absence_share_means(share_array, subjects_array)
                              if share_array is not None else (None, None))
    if stage.arm == 'C' and share_array is None:
        raise ValueError('absence-active screen result requires per-visit shares')
    out.mkdir(parents=True, exist_ok=False)
    logits_path, proba_path = out / 'logits.npz', out / 'proba.npz'
    np.savez_compressed(logits_path, logits=logits, y=y, subjects=subjects_array, sample_ids=ids_array)
    np.savez_compressed(proba_path, proba=proba, y=y, subjects=subjects_array, sample_ids=ids_array)
    shares_path, shares_hash = None, None
    if share_array is not None:
        shares_path = out / 'shares.npz'
        with shares_path.open('xb') as stream:
            np.savez_compressed(stream, absence_share=share_array,
                                subjects=subjects_array, sample_ids=ids_array)
        shares_hash = _file_sha256(shares_path)
    result_record = ScreenResult(
        arm=stage.arm, seed=stage.seed, k=stage.k,
        checkpoint_sha256=_file_sha256(checkpoint_path), binding_sha256=_file_sha256(binding_path),
        k_selection_sha256=freeze_hash, screen_record_sha256=record_hash, row_count=len(ids),
        macro_f1=float(macro_f1), logits_path=str(logits_path),
        logits_sha256=hashlib.sha256(np.ascontiguousarray(logits).tobytes()).hexdigest(),
        proba_path=str(proba_path),
        proba_sha256=hashlib.sha256(np.ascontiguousarray(proba).tobytes()).hexdigest(),
        absence_share_mean=row_mean, absence_share_row_mean=row_mean,
        absence_share_patient_mean=patient_mean,
        absence_share_aggregation_version=ABSENCE_AGGREGATION_VERSION,
        shares_path=None if shares_path is None else str(shares_path),
        shares_sha256=shares_hash)
    assert_absence_share_replay(result_record)
    _write_new_json(out / 'screen_result.json', asdict(result_record))
    return result_record
