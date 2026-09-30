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
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .. import cei_v3_study as study
from .arm_guards import CONTROL_ARM, EXTENSION_ARMS
from .offsets import OFFSET_SCREEN_DIRNAME

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


def _is_hex_digest(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _argv_options(argv) -> list:
    argv = list(argv)
    return [(argv[i], argv[i + 1]) for i in range(len(argv) - 1) if argv[i].startswith('--')]


def _assert_mandatory_flags(name, argv) -> None:
    """EXT §6 (E4): every CEI stage runs `--selection-fold dev --dev-limit 5000 --final-eval none`."""
    options = _argv_options(argv)
    for flag, value in (('--selection-fold', 'dev'), ('--dev-limit', '5000'), ('--final-eval', 'none')):
        found = [v for f, v in options if f == flag]
        if found != [value]:
            key = flag.lstrip('-').replace('-', '_')
            raise ValueError(f'{name}: stage argv must carry {flag} {value} exactly once ({key} = '
                             f'{found!r}); train.py would otherwise score validation (EXT §6 E4)')


def _validate_extension_config(config) -> Tuple[Tuple[str, ...], Dict[int, str]]:
    """Refuse protocol drift; return (family arms run, offset hashes by seed or {})."""
    if isinstance(config.m, bool) or config.m != FAMILY_M:
        raise ValueError(f'multiplicity m must be {FAMILY_M} (EXT §7, D10.4): five pre-registered '
                         f'contrasts vs C; got m = {config.m!r}; m stays 5 even if an arm is not run')
    arms = tuple(config.arms)
    foreign = [arm for arm in arms if arm not in FAMILY_ARMS]
    if foreign:
        raise ValueError(f'arm drift: {foreign} are not extension arms of the EXT §6 arm table '
                         f'{list(FAMILY_ARMS)} (C is the control, never a family arm)')
    duplicates = sorted({arm for arm in arms if arms.count(arm) > 1})
    if duplicates:
        raise ValueError(f'arm(s) {duplicates} listed twice: every arm is screened once per '
                         'frozen checkpoint, never twice (v3 §5, §11.2 item 9)')
    if not arms:
        raise ValueError('arm drift: the extension family has no arm to run')
    if tuple(config.seeds) != study.SEEDS:
        raise ValueError(f'seeds {tuple(config.seeds)} differ from the pre-registered seeds '
                         f'{study.SEEDS}')
    if config.selection_fold != 'dev':
        raise ValueError(f"selection_fold must be 'dev', got {config.selection_fold!r}")
    if isinstance(config.dev_limit, bool) or config.dev_limit != study.FULL_BUDGET[1]:
        raise ValueError(f'dev_limit must be {study.FULL_BUDGET[1]}, got {config.dev_limit!r}')
    if config.final_eval != 'none':
        raise ValueError(f"final_eval must be 'none' (EXT §6 E4), got {config.final_eval!r}; "
                         'train.py would otherwise score validation once per stage')
    approval, offsets_by_seed = config.approval_record_sha256, config.offset_record_sha256
    if approval is None and offsets_by_seed is None:
        return arms, {}
    if approval is None:
        raise ValueError('offset record hashes given without the item-5 approval record hash; '
                         'every O row binds both (EXT §4.2, §4.3)')
    if not _is_hex_digest(approval):
        raise ValueError('approval_record_sha256 is not a hex SHA-256 digest')
    if not isinstance(offsets_by_seed, dict):
        raise ValueError('offset_record_sha256 must map seed -> frozen offset record SHA-256; '
                         'the approval record hash alone does not bind delta')
    hashes = {}
    for seed in study.SEEDS:
        value = offsets_by_seed.get(seed)
        if not _is_hex_digest(value):
            raise ValueError(f'offset record hash for seed {seed} is missing or not a hex digest; '
                             'O needs one frozen delta per C checkpoint (EXT §4.2)')
        hashes[int(seed)] = value
    extra = sorted(set(offsets_by_seed) - set(study.SEEDS))
    if extra:
        raise ValueError(f'offset record hashes carry unknown seeds {extra}')
    return arms, hashes


def _validate_v3_plan(v3_plan, config) -> Tuple[int, Dict[int, study.Stage]]:
    """The v3 plan must be frozen at K* with the same record; return (K*, control stages)."""
    if not isinstance(v3_plan, study.Plan):
        raise ValueError('v3_plan must be the U5 Plan')
    if v3_plan.k_selection_sha256 is None:
        raise ValueError('the v3 plan is not frozen (no k_selection_sha256); extension arms are '
                         'planned only after the K freeze (v3 §11.2 item 6, EXT §8 step 1)')
    record = config.k_selection
    study._validate_k_selection_record(record)
    if study.k_selection_sha256(record) != v3_plan.k_selection_sha256:
        raise ValueError('config k_selection record hashes differently from the v3 plan '
                         'k_selection_sha256; extension arms bind exactly the frozen record')
    controls = record.get('control_binding_sha256')
    if (not isinstance(controls, list) or len(controls) != len(study.SEEDS)
            or not all(_is_hex_digest(h) for h in controls)):
        raise ValueError("k_selection lacks C's three control_binding_sha256 values (v3 §12.21)")
    if v3_plan.order[-1] != 'screen':
        raise ValueError('v3 plan order must end with the screen step')
    k_selected = int(record['k_selected'])
    by_name = {stage.name: stage for stage in v3_plan.stages}
    if len(by_name) != len(v3_plan.stages):
        raise ValueError('v3 plan carries a duplicated stage name')
    for stage in v3_plan.stages:
        if not stage.argv:
            raise ValueError(f'{stage.name}: v3 stage is not materialised (no argv)')
        _assert_mandatory_flags(stage.name, stage.argv)
    control_stages: Dict[int, study.Stage] = {}
    for seed in study.SEEDS:
        name = f'C_K{k_selected}_seed{seed}'
        stage = by_name.get(name)
        if stage is None:
            raise ValueError(f'v3 plan lacks the control stage {name} at the frozen K')
        if stage.arm != CONTROL_ARM or stage.k != k_selected or stage.seed != seed:
            raise ValueError(f'arm drift: control stage {name} is arm {stage.arm!r} K {stage.k!r} '
                             f'seed {stage.seed!r}, not C at K* = {k_selected}')
        options = _argv_options(stage.argv)
        if ('--method-option', 'arm=C') not in options or ('--edge-direction', 'forward') not in options:
            raise ValueError(f'arm drift: control stage {name} argv is not arm C on the forward view')
        control_stages[seed] = stage
    return k_selected, control_stages


def _extension_argv(arm, control_argv, output) -> Tuple[str, ...]:
    """C's command line with the output replaced and exactly one difference (EXT §6 table)."""
    argv = list(control_argv)
    argv[argv.index('--output') + 1] = str(output)
    if arm == 'E2w':
        argv += ['--hidden', '256']
    elif arm == 'E2d':
        argv += ['--layers', '2']
    elif arm == 'E6a':
        argv[argv.index('--edge-direction') + 1] = 'bidirectional'
    elif arm == 'E6b':
        argv += ['--method-option', 'comorbid_block=1']
    else:
        raise ValueError(f'{arm!r} is not a trained extension arm')
    return tuple(argv)


def extension_plan(v3_plan, config) -> study.Plan:
    """Locked combined run plan (EXT §8) on top of a frozen v3 plan (U5 `plan`).

    Order: the v3 plan unchanged (state fit, nine C stages, K freeze, A and B), the twelve
    extension stages E2w/E2d/E6a/E6b x seeds (each C's exact command line plus one
    difference of the §6 table, with the mandatory flags), the freeze steps (checkpoint
    hashes, item-5 approval record, three validation scorings of C, three delta fits),
    the 24 screen rows (21 inferences + O from C's stored logits + delta, §4.3) and one
    decision analysis. m = 5 is fixed even when an arm is not run (§7). Refuses arm drift,
    m != 5, an arm screened twice or an occupied screen output, an incomplete delta /
    approval binding, and any `final_eval != 'none'`. Nothing is executed or opened.
    """
    arms, offset_hashes = _validate_extension_config(config)
    k_selected, controls = _validate_v3_plan(v3_plan, config)
    root = Path(v3_plan.output_root)
    if config.output_root is not None and Path(config.output_root).resolve() != root.resolve():
        raise ValueError('config output_root differs from the v3 plan output_root')
    trained = tuple(arm for arm in TRAINED_EXTENSION_ARMS if arm in arms)
    stages = list(v3_plan.stages)
    for arm in trained:
        for seed in study.SEEDS:
            name = f'{arm}_seed{seed}'
            output = root / name
            if output.exists():
                raise FileExistsError(f'Refusing occupied stage output {output}')
            control = controls[seed]
            stages.append(study.Stage(
                name=name, phase='extension', arm=arm, k=k_selected, seed=seed,
                output=str(output), argv=_extension_argv(arm, control.argv, output),
                v3_state=control.v3_state))
    for stage in stages:
        _assert_mandatory_flags(stage.name, stage.argv)
    screen_arms = ('A', 'B', 'C') + trained + ((OFFSET_ARM,) if OFFSET_ARM in arms else ())
    rows = []
    approval = config.approval_record_sha256 if offset_hashes else None
    for arm in screen_arms:
        for seed in study.SEEDS:
            if arm == 'C' or arm == OFFSET_ARM:
                stage_name = controls[seed].name
            else:
                stage_name = f'{arm}_seed{seed}'
            screen_dir = root / stage_name / study.SCREEN_DIRNAME
            if arm == OFFSET_ARM:
                rows.append(ScreenRow(
                    name=f'O_seed{seed}', arm=OFFSET_ARM, seed=seed, stage=stage_name,
                    source='stored_logits+delta', output=str(screen_dir / OFFSET_SCREEN_DIRNAME),
                    offset_record_sha256=offset_hashes.get(seed), approval_record_sha256=approval))
            else:
                rows.append(ScreenRow(name=f'{arm}_seed{seed}', arm=arm, seed=seed,
                                      stage=stage_name, source='inference', output=str(screen_dir)))
    for row in rows:
        if Path(row.output).exists():
            raise FileExistsError(f'Refusing to plan screen row {row.name}: output {row.output} '
                                  'exists; a frozen checkpoint is screened once, never twice')
    freeze = ['freeze:checkpoint_hashes']
    if OFFSET_ARM in arms:
        freeze.append('freeze:item5_approval_record')
        freeze += [f'validation:C_seed{seed}' for seed in study.SEEDS]
        freeze += [f'offset_fit:C_seed{seed}' for seed in study.SEEDS]
    order = (tuple(v3_plan.order[:-1])
             + tuple(stage.name for stage in stages[len(v3_plan.stages):])
             + tuple(freeze)
             + tuple(f'screen:{row.name}' for row in rows)
             + ('decision',))
    family = tuple(arm for arm in FAMILY_ARMS if arm in arms)
    return ExtensionPlan(
        output_root=v3_plan.output_root, v3_state_paths=dict(v3_plan.v3_state_paths),
        stages=tuple(stages), k_selection_path=v3_plan.k_selection_path,
        k_selection_sha256=v3_plan.k_selection_sha256, order=order,
        arms=family, arms_not_run=tuple(arm for arm in FAMILY_ARMS if arm not in arms),
        m=FAMILY_M, contrasts=tuple((arm, CONTROL_ARM) for arm in family),
        control_stages=tuple(controls[seed].name for seed in study.SEEDS),
        control_binding_sha256=tuple(config.k_selection['control_binding_sha256']),
        screen_rows=tuple(rows), approval_record_sha256=approval,
        offset_record_sha256=dict(offset_hashes) if offset_hashes else None)


# ------------------------------------------------------------- family bounds

NOMINAL_P = 0.025
BOUND_RULE = 'order statistic: sorted[floor(resamples * p) - 1]'
LINEAR_QUANTILE_USE = 'cross-reading only, never the decision'


def family_m(arms_run) -> int:
    """Multiplicity of the pre-registered family (EXT §7): five contrasts vs C, fixed before
    any result; if an arm is not run, m stays 5 (conservative). `arms_run` must be a subset
    of the family."""
    foreign = [arm for arm in tuple(arms_run) if arm not in FAMILY_ARMS]
    if foreign:
        raise ValueError(f'arm drift: {foreign} are not extension arms of the family {list(FAMILY_ARMS)}')
    return FAMILY_M


def family_bounds(deltas, m: int = FAMILY_M) -> dict:
    """Exact bound rule of the extension family (EXT §7, E5) on the 1,000 averaged deltas.

    Both bounds are order statistics of the sorted deltas with 0-based index
    `floor(resamples * p) - 1`: nominal p = 0.025 -> sorted[24]; corrected p = 0.025 / m ->
    sorted[4] at m = 5. The decision uses the corrected bound; the nominal bound is reported
    next to it; `np.quantile(deltas, 0.025, method='linear')` (the v3 C-vs-A rule, v3 §12.20)
    is returned for cross-reading only. `m` must be 5 (D10.4).
    """
    if isinstance(m, bool) or not isinstance(m, int) or m != FAMILY_M:
        raise ValueError(f'multiplicity m must be {FAMILY_M} (EXT §7, D10.4); got m = {m!r}; the '
                         'family is fixed before any result and m stays 5 if an arm is not run')
    values = np.asarray(deltas, dtype=np.float64)
    resamples = study.BOOTSTRAP_RESAMPLES
    if values.ndim != 1 or values.shape[0] != resamples:
        raise ValueError(f'the family bounds require exactly {resamples} averaged deltas '
                         f'(v3 §6.2 condition 3), got shape {values.shape}')
    if not np.isfinite(values).all():
        raise ValueError('bootstrap deltas must be finite')
    ordered = np.sort(values)             # copy; the input is left unchanged
    nominal_index = math.floor(resamples * NOMINAL_P) - 1
    corrected_p = NOMINAL_P / m
    corrected_index = math.floor(resamples * corrected_p) - 1
    return {
        'nominal': float(ordered[nominal_index]),
        'corrected': float(ordered[corrected_index]),
        'linear_quantile': float(np.quantile(values, NOMINAL_P, method=study.QUANTILE_METHOD)),
        'm': int(m), 'resamples': int(resamples),
        'nominal_p': NOMINAL_P, 'corrected_p': corrected_p,
        'nominal_index': int(nominal_index), 'corrected_index': int(corrected_index),
        'corrected_tail_mass': (corrected_index + 1) / resamples,
        'rule': BOUND_RULE, 'decision_bound': 'corrected',
        'linear_quantile_method': study.QUANTILE_METHOD,
        'linear_quantile_use': LINEAR_QUANTILE_USE,
    }
