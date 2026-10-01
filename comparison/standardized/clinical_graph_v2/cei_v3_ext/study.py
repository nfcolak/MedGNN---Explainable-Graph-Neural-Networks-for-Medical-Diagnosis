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
import hashlib
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .. import cei_v3_study as study
from .arm_guards import CONTROL_ARM, EXTENSION_ARMS
from . import offsets as offsets_module
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


def extension_plan(v3_plan, config, *, allow_existing=False) -> study.Plan:
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
            if output.exists() and not allow_existing:
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
        if Path(row.output).exists() and not allow_existing:
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


# --------------------------------------------------------- extension decision

CLAIMS = {   # EXT §2.5, §4.3, §5.4 wording of a positive (family-corrected) decision
    'E2w': 'wider CEI v3 beats CEI v3 C on this screen',
    'E2d': 'deeper CEI v3 beats CEI v3 C on this screen',
    OFFSET_ARM: "validation-tuned per-class offsets improve CEI v3 C's screen macro-F1",
    'E6a': 'the bidirectional edge view improves CEI v3 C on this screen',
    'E6b': 'the additive comorbid pair term improves CEI v3 C on this screen',
}


def _result_dict(result) -> dict:
    if isinstance(result, dict):
        return dict(result)
    if hasattr(result, '__dataclass_fields__'):
        return dict(vars(result))
    raise ValueError('screen result must be a U5 ScreenResult, an O row dict or its JSON dict')


def _load_npz(path, keys) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'stored screen arrays missing at {path}')
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(keys):
            raise ValueError(f'{path.name} does not carry {sorted(keys)}')
        return {key: np.ascontiguousarray(saved[key]) for key in keys}


def _checked_c_screen_result(seed, result) -> dict:
    """Static + on-disk checks of one C screen result before any O output is written."""
    result = _result_dict(result)
    missing = [key for key in offsets_module._SCREEN_RESULT_KEYS if key not in result]
    if missing:
        raise ValueError(f'C screen result for seed {seed} lacks bound fields: {missing}')
    if result['arm'] != CONTROL_ARM:
        raise ValueError(f"O is built on arm C screen logits only; seed {seed} result is arm "
                         f"{result['arm']!r}")
    if result['seed'] != seed:
        raise ValueError(f'C screen result keyed by seed {seed} carries seed {result["seed"]!r}')
    if not _is_hex_digest(result['logits_sha256']):
        raise ValueError(f'C screen result for seed {seed} carries no logits hash; O is never '
                         'built from proba or unhashed logits (EXT E7)')
    arrays = _load_npz(result['logits_path'], ('logits', 'y', 'subjects', 'sample_ids'))
    logits = arrays['logits']
    if logits.dtype != np.float32:
        raise ValueError(f'stored screen logits for seed {seed} are not raw float32')
    if hashlib.sha256(logits.tobytes()).hexdigest() != result['logits_sha256']:
        raise ValueError(f'stored screen logits for seed {seed} differ from the hash in the C '
                         'screen result')
    return result


def offset_screen_rows(c_results, offset_records, *, approval_record_sha256,
                       output_dirs=None) -> dict:
    """Arm O for the three seeds: C's frozen delta applied to C's stored raw screen logits
    (EXT §4.3), through the X3 scorer; no inference, no fold is read.

    `c_results[seed]` = the U5 `ScreenResult` (or its JSON dict) of the C checkpoint;
    `offset_records[seed]` = the frozen X3 offset record of the same checkpoint. Every seed's
    inputs are checked (field set, arm C, seed, logits hash on disk, delta binding to the
    checkpoint / binding / K freeze / approval record) before the first row is written, so
    a refusal leaves no O output behind. Returns `{seed: o_result dict}`.
    """
    if not isinstance(c_results, dict) or not isinstance(offset_records, dict):
        raise ValueError('c_results and offset_records must map seed -> record')
    for label, mapping in (('c_results', c_results), ('offset_records', offset_records)):
        if sorted(mapping) != sorted(study.SEEDS):
            raise ValueError(f'{label} must carry exactly the seeds {list(study.SEEDS)}, got '
                             f'{sorted(mapping)}')
    if not _is_hex_digest(approval_record_sha256):
        raise ValueError('approval_record_sha256 must be the hex SHA-256 of the item-5 approval record')
    checked = {}
    for seed in study.SEEDS:
        result = _checked_c_screen_result(seed, c_results[seed])
        record = offset_records[seed]
        delta_int = offsets_module._validate_offset_record_fields(record)
        if record['seed'] != seed:
            raise ValueError(f'offset record keyed by seed {seed} carries seed {record["seed"]!r}')
        for key in ('checkpoint_sha256', 'binding_sha256', 'k_selection_sha256'):
            if record[key] != result[key]:
                raise ValueError(f'seed {seed}: offset record {key} differs from the screened C '
                                 'checkpoint; delta is bound to one checkpoint only')
        if record['k'] != result['k']:
            raise ValueError(f'seed {seed}: offset record K differs from the screened checkpoint')
        if record['approval_record_sha256'] != approval_record_sha256:
            raise ValueError(f'seed {seed}: offset record approval-record SHA-256 differs from the '
                             'bound approval')
        out = (Path(output_dirs[seed]) if output_dirs is not None and seed in output_dirs
               else Path(result['logits_path']).parent / OFFSET_SCREEN_DIRNAME)
        if out.exists():
            raise FileExistsError(f'Refusing occupied O output {out}')
        checked[seed] = (result, record, delta_int, out)
    rows = {}
    for seed in study.SEEDS:
        result, record, _delta, out = checked[seed]
        rows[seed] = offsets_module.score_offset_screen(
            result, record, approval_record_sha256=approval_record_sha256, output_dir=out)
    return rows


def load_screen_row(result) -> tuple:
    """`(y, pred, subjects)` of one screen row from its stored arrays, for `paired_bootstrap`.

    An O row (`o_result.json` dict: `pred_path`, `pred_sha256`) is read from `o_pred.npz`;
    every other row (U5 `ScreenResult` / `screen_result.json`) is `argmax` of its hashed raw
    `logits.npz`. Hashes are verified against the stored bytes; nothing is inferred.
    """
    result = _result_dict(result)
    if result.get('arm') == OFFSET_ARM:
        for key in ('pred_path', 'pred_sha256', 'screen_logits_sha256', 'offset_record_sha256'):
            if key not in result:
                raise ValueError(f'O screen row lacks {key}')
        arrays = _load_npz(result['pred_path'], ('pred', 'y', 'subjects', 'sample_ids'))
        pred = arrays['pred']
        if hashlib.sha256(pred.tobytes()).hexdigest() != result['pred_sha256']:
            raise ValueError('stored O predictions differ from the hash in the O result')
    else:
        for key in ('logits_path', 'logits_sha256'):
            if not result.get(key):
                raise ValueError(f'screen result lacks {key}; only hashed raw logits are read')
        arrays = _load_npz(result['logits_path'], ('logits', 'y', 'subjects', 'sample_ids'))
        logits = arrays['logits']
        if logits.dtype != np.float32:
            raise ValueError('stored screen logits are not raw float32')
        if hashlib.sha256(logits.tobytes()).hexdigest() != result['logits_sha256']:
            raise ValueError('stored screen logits differ from the hash in the screen result')
        pred = logits.argmax(1).astype(np.int64)
    if 'row_count' in result and int(result['row_count']) != pred.shape[0]:
        raise ValueError('stored screen rows differ from the result row_count')
    return (np.asarray(arrays['y']).astype(np.int64), np.asarray(pred).astype(np.int64),
            np.asarray(arrays['subjects']).astype(str))


def _validate_family_inputs(deltas, scores, m, seeds) -> Tuple[str, ...]:
    if isinstance(m, bool) or not isinstance(m, int) or m != FAMILY_M:
        raise ValueError(f'multiplicity m must be {FAMILY_M} (EXT §7, D10.4); got m = {m!r}')
    if not isinstance(deltas, dict) or not deltas:
        raise ValueError('deltas must map (arm, C) -> the 1,000 averaged bootstrap deltas of at '
                         'least one family contrast')
    if tuple(seeds) != study.SEEDS:
        raise ValueError(f'seeds {tuple(seeds)} differ from the pre-registered seeds {study.SEEDS}')
    arms = []
    for key in deltas:
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(f'contrast key {key!r} is not a (treatment, control) pair')
        arm, control = key
        if control != CONTROL_ARM:
            raise ValueError(f'contrast {key!r} is not vs C: every family contrast is arm vs C '
                             '(EXT §7)')
        if arm not in FAMILY_ARMS:
            raise ValueError(f'arm drift: {arm!r} is not one of the pre-registered family arms '
                             f'{list(FAMILY_ARMS)}; no contrast may be added after registration')
        arms.append(arm)
    if len(set(arms)) != len(arms):
        raise ValueError('a family contrast appears twice')
    if not isinstance(scores, dict) or CONTROL_ARM not in scores:
        raise ValueError("scores must map arm -> {seed: screen macro-F1} and include the control 'C'")
    for arm in arms + [CONTROL_ARM]:
        if arm not in scores:
            raise ValueError(f'scores lack arm {arm!r}')
        for seed in study.SEEDS:
            if seed not in scores[arm]:
                raise ValueError(f'scores for arm {arm!r} lack seed {seed}')
    return tuple(arm for arm in FAMILY_ARMS if arm in arms)


def decide_extension(deltas, scores, *, m: int = FAMILY_M, seeds=study.SEEDS) -> dict:
    """The pre-registered extension decision (EXT §7; v3 §6.2 conditions 1–3 per contrast).

    `deltas[(arm, 'C')]` = the 1,000 averaged bootstrap deltas of U5 `paired_bootstrap`;
    `scores[arm][seed]` = screen macro-F1. Conditions 1 (per-seed strict win) and 2
    (seed-mean win) are uncorrected; condition 3 requires the corrected order-statistic
    bound `sorted[4]` (m = 5) strictly above zero. An arm passing all three "beats C on this
    screen (family-corrected, m = 5)"; one passing 1–2 with only the nominal bound
    `sorted[24]` above zero is a "nominal win, not family-corrected" and not a positive
    decision. m stays 5 when an arm is not run; the linear quantile is reported for
    cross-reading only.
    """
    arms = _validate_family_inputs(deltas, scores, m, seeds)
    contrasts = {}
    for arm in arms:
        values = np.asarray(deltas[(arm, CONTROL_ARM)], dtype=np.float64)
        bounds = family_bounds(values, m=m)
        per_seed = {str(seed): {arm: float(scores[arm][seed]), CONTROL_ARM: float(scores[CONTROL_ARM][seed])}
                    for seed in study.SEEDS}
        means = {name: float(np.mean([per_seed[str(seed)][name] for seed in study.SEEDS]))
                 for name in (arm, CONTROL_ARM)}
        low = arm.lower()
        checks = {
            f'{low}_beats_c_each_seed': all(per_seed[str(seed)][arm] > per_seed[str(seed)][CONTROL_ARM]
                                            for seed in study.SEEDS),
            f'{low}_mean_above_c': means[arm] > means[CONTROL_ARM],
            f'{low}_minus_c_corrected_lower_bound_above_zero': bounds['corrected'] > 0,
        }
        nominal_above = bounds['nominal'] > 0
        uncorrected = checks[f'{low}_beats_c_each_seed'] and checks[f'{low}_mean_above_c']
        wins = all(checks.values())
        nominal_win = bool(uncorrected and nominal_above and not wins)
        if wins:
            statement = f'{arm} beats C on this screen (family-corrected, m = {m})'
        elif nominal_win:
            statement = f'{arm}: nominal win, not family-corrected'
        else:
            statement = f'{arm}: benefit not demonstrated on this screen'
        contrasts[arm] = {
            'contrast': [arm, CONTROL_ARM], 'metric': 'weighted_macro_f1',
            'per_seed': per_seed, 'seed_means': means,
            'point_delta': means[arm] - means[CONTROL_ARM],
            'bounds': bounds,
            'corrected_lower_bound': bounds['corrected'],
            'nominal_lower_bound': bounds['nominal'],
            'linear_quantile': bounds['linear_quantile'],
            'checks': checks,
            'nominal_lower_bound_above_zero': bool(nominal_above),
            'beats_control': bool(wins), 'nominal_win': nominal_win,
            'statement': statement,
            'claim': CLAIMS[arm] if wins else None,
        }
    return {
        'control': CONTROL_ARM, 'metric': 'weighted_macro_f1', 'm': int(m),
        'arms': arms, 'arms_not_run': tuple(arm for arm in FAMILY_ARMS if arm not in arms),
        'family': list(FAMILY_ARMS), 'contrasts': contrasts,
        'winners': tuple(arm for arm in arms if contrasts[arm]['beats_control']),
        'nominal_only': tuple(arm for arm in arms if contrasts[arm]['nominal_win']),
        'resamples': int(study.BOOTSTRAP_RESAMPLES), 'bootstrap_seed': int(study.BOOTSTRAP_SEED),
        'bound_rule': BOUND_RULE, 'decision_bound': 'corrected',
        'linear_quantile_use': LINEAR_QUANTILE_USE,
        'multiplicity_rule': ('Bonferroni on condition 3 only, m = 5 fixed before any result; '
                              'm stays 5 if an arm is not run (EXT §7, D10.4)'),
        'validation_evaluated': False, 'test_evaluated': False,
    }


def _per_graph_abs(values, owner, graph_count, classes):
    import torch

    total = torch.zeros((graph_count, classes), dtype=torch.float64)
    if values.numel():
        total.index_add_(0, owner.long(), values.detach().abs().to(torch.float64))
    return total


def _comorbid_owner(parts, batch_index):
    pairs = parts['comorbid_pairs']
    return batch_index[pairs[0]] if pairs.numel() else pairs.new_zeros((0,))


def comorbid_share(parts, *, batch_index, edge_index, graph_count) -> np.ndarray:
    """Secondary E6b metric (EXT §5.4), defined as v3 §6.2 defines the absence share: per
    graph, mean over classes of `sum|comorbid| / sum|node + edge + pair + absence + comorbid|`,
    0 where the denominator is 0. Parts without the block (arm C) give 0."""
    import torch

    graph_count = int(graph_count)
    batch_index = torch.as_tensor(batch_index, dtype=torch.long).view(-1)
    classes = int(parts['node_contributions'].size(1))
    node = _per_graph_abs(parts['node_contributions'], batch_index, graph_count, classes)
    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    edge = _per_graph_abs(parts['edge_contributions'], batch_index[edge_index[0]]
                          if edge_index.numel() else edge_index.new_zeros((0,)), graph_count, classes)
    pairs = parts['pairs']
    pair = _per_graph_abs(parts['pair_contributions'], batch_index[pairs[0]]
                          if pairs.numel() else pairs.new_zeros((0,)), graph_count, classes)
    absence = _per_graph_abs(parts['absence_contributions'], parts['absence_items'][0],
                             graph_count, classes)
    if 'comorbid_contributions' in parts:
        comorbid = _per_graph_abs(parts['comorbid_contributions'],
                                  _comorbid_owner(parts, batch_index), graph_count, classes)
    else:
        comorbid = torch.zeros((graph_count, classes), dtype=torch.float64)
    denominator = node + edge + pair + absence + comorbid
    share = torch.where(denominator > 0, comorbid / denominator.clamp(min=1e-300),
                        torch.zeros_like(denominator))
    return share.mean(dim=1).numpy()


def comorbid_nonempty(parts, *, batch_index, graph_count) -> np.ndarray:
    """Per-graph flag: the comorbid pair set is non-empty (EXT §5.4 secondary fraction)."""
    import torch

    graph_count = int(graph_count)
    flags = np.zeros(graph_count, dtype=bool)
    if 'comorbid_pairs' not in parts:
        return flags
    batch_index = torch.as_tensor(batch_index, dtype=torch.long).view(-1)
    owner = _comorbid_owner(parts, batch_index)
    if owner.numel():
        flags[np.unique(owner.cpu().numpy())] = True
    return flags
