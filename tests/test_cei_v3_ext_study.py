"""Unit X6 — extension study: plan lock, family bounds, extension decision (EXT spec §6–§9).

Synthetic fixtures only: no data file, no run directory, no fold is opened. The v3 plan is
built with U5 `plan` on empty temporary inputs; bindings, records, deltas and logits are
synthetic arrays and dicts.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from comparison.standardized.clinical_graph_v2 import cei_v3_study as study
from comparison.standardized.clinical_graph_v2.cei_v3_ext import arm_guards, offsets
from comparison.standardized.clinical_graph_v2.cei_v3_ext import study as ext

TRAIN_PY = 'comparison.standardized.clinical_graph_v2.train'
SEEDS = (1234, 2025, 7)
MANDATORY = (('--selection-fold', 'dev'), ('--dev-limit', '5000'), ('--final-eval', 'none'))


# ------------------------------------------------------------------ fixtures


def _inputs(tmp_path):
    artifact = tmp_path / 'artifact'
    artifact.mkdir(exist_ok=True)
    targets = tmp_path / 'targets.csv'
    if not targets.exists():
        targets.write_text('sample_id,target,split,subject_id\n')
    canonical = tmp_path / 'canonical.json'
    if not canonical.exists():
        canonical.write_text('{}')
    return artifact, targets, canonical, tmp_path / 'study_root'


def _k_selection(k_selected=8):
    return {'version': study.K_SELECTION_VERSION, 'k_grid': list(study.K_GRID),
            'k_selected': k_selected, 'rule': study.K_SELECTION_RULE,
            'control_binding_sha256': ['1' * 64, '2' * 64, '3' * 64]}


def _v3_plan(tmp_path, k_selected=8, frozen=True):
    artifact, targets, canonical, root = _inputs(tmp_path)
    config = study.StudyConfig(artifact=str(artifact), targets=str(targets),
                               canonical=str(canonical), output_root=str(root),
                               k_selection=_k_selection(k_selected) if frozen else None)
    return study.plan(config), config


def _hashes():
    return {seed: hashlib.sha256(f'offset_record seed {seed}'.encode()).hexdigest()
            for seed in SEEDS}


def _ext_config(k_selected=8, **overrides):
    fields = dict(k_selection=_k_selection(k_selected))
    fields.update(overrides)
    return ext.ExtensionConfig(**fields)


def _stage(plan_, name):
    matches = [stage for stage in plan_.stages if stage.name == name]
    assert len(matches) == 1, f'plan must contain exactly one stage {name!r}'
    return matches[0]


def _pairs(argv):
    """(flag, value) options of a train.py argv; every option here takes one value."""
    argv = list(argv)
    return [(argv[i], argv[i + 1]) for i in range(len(argv) - 1) if argv[i].startswith('--')]


# ------------------------------------------------------ step 1: extension plan lock


def test_extension_plan_lists_the_eight_section_stages_in_order(tmp_path):
    v3_plan, _ = _v3_plan(tmp_path)
    plan_ = ext.extension_plan(v3_plan, _ext_config())
    assert isinstance(plan_, study.Plan)
    ext_names = [f'{arm}_seed{seed}' for arm in ('E2w', 'E2d', 'E6a', 'E6b') for seed in SEEDS]
    # Step 1: the v3 plan (state fit, nine C stages, K freeze, A and B), unchanged.
    v3_part = tuple(v3_plan.order[:-1])
    assert v3_plan.order[-1] == 'screen'
    assert plan_.order[:len(v3_part)] == v3_part
    # Step 2: the twelve extension stages.
    n = len(v3_part)
    assert list(plan_.order[n:n + 12]) == ext_names
    # Step 3: freeze — checkpoint hashes, the item-5 approval record, 3 validation
    # scorings of C, 3 offset fits.
    freeze = plan_.order[n + 12:n + 20]
    assert freeze[0] == 'freeze:checkpoint_hashes'
    assert freeze[1] == 'freeze:item5_approval_record'
    assert list(freeze[2:5]) == [f'validation:C_seed{seed}' for seed in SEEDS]
    assert list(freeze[5:8]) == [f'offset_fit:C_seed{seed}' for seed in SEEDS]
    # Step 4: 24 screen rows (7 arms x 3 inferences + O x 3 from stored logits), then the
    # decision analysis once.
    screen = plan_.order[n + 20:n + 44]
    assert list(screen) == [f'screen:{arm}_seed{seed}'
                            for arm in ('A', 'B', 'C', 'E2w', 'E2d', 'E6a', 'E6b', 'O')
                            for seed in SEEDS]
    assert plan_.order[n + 44:] == ('decision',)
    assert len(plan_.order) == n + 45
    # Stage list: the 15 v3 stages then the 12 extension stages; 27 CEI stages (EXT §8).
    assert [s.name for s in plan_.stages] == [s.name for s in v3_plan.stages] + ext_names
    assert len(plan_.stages) == 27
    assert [s.phase for s in plan_.stages[15:]] == ['extension'] * 12
    assert plan_.output_root == v3_plan.output_root
    assert plan_.v3_state_paths == v3_plan.v3_state_paths
    assert plan_.k_selection_path == v3_plan.k_selection_path
    assert plan_.k_selection_sha256 == v3_plan.k_selection_sha256


def test_extension_stage_argv_is_c_plus_exactly_one_difference_with_the_mandatory_flags(tmp_path):
    v3_plan, config = _v3_plan(tmp_path, k_selected=16)
    plan_ = ext.extension_plan(v3_plan, _ext_config(k_selected=16))
    root = Path(config.output_root).resolve()
    control = _stage(v3_plan, 'C_K16_seed2025')
    for arm, extra in (('E2w', ('--hidden', '256')), ('E2d', ('--layers', '2')),
                       ('E6a', ('--edge-direction', 'bidirectional')),
                       ('E6b', ('--method-option', 'comorbid_block=1'))):
        stage = _stage(plan_, f'{arm}_seed2025')
        assert (stage.arm, stage.k, stage.seed, stage.phase) == (arm, 16, 2025, 'extension')
        assert stage.output == str(root / f'{arm}_seed2025')
        assert stage.v3_state == control.v3_state == str(root / 'v3_state' / 'K16.json')
        argv = stage.argv
        assert argv[:3] == (sys.executable, '-m', TRAIN_PY)
        assert argv[argv.index('--output') + 1] == stage.output
        assert argv[argv.index('--seed') + 1] == '2025'
        assert argv[argv.index('--method') + 1] == 'cei_gnn_v3'
        for pair in MANDATORY:
            assert pair in _pairs(argv)
        assert ('--final-eval', 'validation') not in _pairs(argv)
        assert ('--method-option', 'arm=C') in _pairs(argv)
        assert ('--method-option', 'k=16') in _pairs(argv)
        assert ('--method-option', f'v3_state={root / "v3_state" / "K16.json"}') in _pairs(argv)
        assert extra in _pairs(argv)
        # Exactly one difference from C's argv (besides the output path): the arm flag.
        control_pairs = [p for p in _pairs(control.argv) if p[0] != '--output'
                         and p[1] != str(control.output)]
        stage_pairs = [p for p in _pairs(argv) if p[0] != '--output' and p[1] != stage.output]
        added = [p for p in stage_pairs if p not in control_pairs]
        removed = [p for p in control_pairs if p not in stage_pairs]
        if arm == 'E6a':
            assert added == [('--edge-direction', 'bidirectional')]
            assert removed == [('--edge-direction', 'forward')]
        else:
            assert added == [extra] and removed == []
        assert argv.count('--hidden') == (1 if arm == 'E2w' else 0)
        assert argv.count('--layers') == (1 if arm == 'E2d' else 0)
    # E6b is run at hidden 128 only; E2w at depth 1 (EXT §2.3, §5.3).
    assert '--layers' not in _stage(plan_, 'E2w_seed7').argv
    assert '--hidden' not in _stage(plan_, 'E6b_seed7').argv
    # Every one of the 27 CEI stages carries the mandatory flags.
    for stage in plan_.stages:
        for pair in MANDATORY:
            assert pair in _pairs(stage.argv), stage.name


def test_extension_plan_binds_family_contrasts_control_stages_and_screen_rows(tmp_path):
    v3_plan, config = _v3_plan(tmp_path)
    plan_ = ext.extension_plan(v3_plan, _ext_config())
    root = Path(config.output_root).resolve()
    assert plan_.m == 5
    assert plan_.arms == ('E2w', 'E2d', 'O', 'E6a', 'E6b')
    assert plan_.contrasts == (('E2w', 'C'), ('E2d', 'C'), ('O', 'C'), ('E6a', 'C'), ('E6b', 'C'))
    assert plan_.control_stages == tuple(f'C_K8_seed{seed}' for seed in SEEDS)
    assert plan_.control_binding_sha256 == ('1' * 64, '2' * 64, '3' * 64)
    rows = {row.name: row for row in plan_.screen_rows}
    assert list(rows) == [f'{arm}_seed{seed}' for arm in ('A', 'B', 'C', 'E2w', 'E2d', 'E6a',
                                                          'E6b', 'O') for seed in SEEDS]
    assert len(plan_.screen_rows) == 24
    assert sum(row.source == 'inference' for row in plan_.screen_rows) == 21
    assert sum(row.source == 'stored_logits+delta' for row in plan_.screen_rows) == 3
    c_row = rows['C_seed2025']
    assert (c_row.arm, c_row.seed, c_row.stage, c_row.source) == ('C', 2025, 'C_K8_seed2025',
                                                                    'inference')
    assert c_row.output == str(root / 'C_K8_seed2025' / study.SCREEN_DIRNAME)
    e_row = rows['E6b_seed7']
    assert (e_row.arm, e_row.stage, e_row.output) == ('E6b', 'E6b_seed7',
                                                      str(root / 'E6b_seed7' / study.SCREEN_DIRNAME))
    a_row = rows['A_seed1234']
    assert (a_row.stage, a_row.output) == ('A_seed1234', str(root / 'A_seed1234' / study.SCREEN_DIRNAME))
    o_row = rows['O_seed7']
    # O = C's stored screen logits + frozen delta: same checkpoint, no inference (§4.3).
    assert (o_row.arm, o_row.seed, o_row.stage, o_row.source) == ('O', 7, 'C_K8_seed7',
                                                                    'stored_logits+delta')
    assert o_row.output == str(root / 'C_K8_seed7' / study.SCREEN_DIRNAME / offsets.OFFSET_SCREEN_DIRNAME)
    # Before the freeze step the delta / approval hashes are not yet known: unbound rows.
    assert o_row.offset_record_sha256 is None and o_row.approval_record_sha256 is None
    assert plan_.approval_record_sha256 is None and plan_.offset_record_sha256 is None
    assert plan_.validation_evaluated is False and plan_.test_evaluated is False


def test_extension_plan_binds_delta_and_approval_hashes_once_frozen(tmp_path):
    v3_plan, _ = _v3_plan(tmp_path)
    approval = hashlib.sha256(b'item5 approval').hexdigest()
    plan_ = ext.extension_plan(v3_plan, _ext_config(approval_record_sha256=approval,
                                                    offset_record_sha256=_hashes()))
    assert plan_.approval_record_sha256 == approval
    assert plan_.offset_record_sha256 == _hashes()
    for seed in SEEDS:
        row = [r for r in plan_.screen_rows if r.name == f'O_seed{seed}'][0]
        assert row.offset_record_sha256 == _hashes()[seed]
        assert row.approval_record_sha256 == approval
    for row in plan_.screen_rows:
        if row.arm != 'O':
            assert row.offset_record_sha256 is None and row.approval_record_sha256 is None


@pytest.mark.parametrize('overrides, message', [
    ({'m': 4}, 'm'),                                              # m != 5
    ({'m': 6}, 'm'),
    ({'arms': ('E2w', 'E2d', 'O', 'E6a', 'E6b', 'E6c')}, 'arm'),  # arm drift: foreign arm
    ({'arms': ('E2w', 'E2d', 'O', 'E6a', 'C')}, 'arm'),           # control as a family arm
    ({'arms': ('E2w', 'E2d', 'O', 'E6a', 'E6b', 'E2w')}, 'twice'),  # E2w screened twice
    ({'final_eval': 'validation'}, 'final_eval'),
    ({'final_eval': 'test'}, 'final_eval'),
    ({'selection_fold': 'screen'}, 'selection_fold'),
    ({'dev_limit': 4000}, 'dev_limit'),
    ({'seeds': (1234, 2025)}, 'seeds'),
    ({'approval_record_sha256': 'a' * 64}, 'offset'),             # approval without delta
    ({'offset_record_sha256': _hashes()}, 'approval'),            # delta without approval
    ({'approval_record_sha256': 'a' * 64,
      'offset_record_sha256': {1234: 'b' * 64, 2025: 'c' * 64}}, 'offset'),   # seed 7 missing
    ({'approval_record_sha256': 'a' * 64,
      'offset_record_sha256': {**_hashes(), 7: 'short'}}, 'offset'),          # not a digest
    ({'approval_record_sha256': 'not-a-digest', 'offset_record_sha256': _hashes()}, 'approval'),
])
def test_extension_plan_refuses_protocol_and_arm_drift(tmp_path, overrides, message):
    v3_plan, _ = _v3_plan(tmp_path)
    with pytest.raises(ValueError, match=message):
        ext.extension_plan(v3_plan, _ext_config(**overrides))


def test_extension_plan_keeps_m_at_five_when_an_arm_is_not_run(tmp_path):
    v3_plan, _ = _v3_plan(tmp_path)
    plan_ = ext.extension_plan(v3_plan, _ext_config(arms=('E2w', 'E2d', 'O', 'E6b')))
    assert plan_.m == 5
    assert plan_.arms == ('E2w', 'E2d', 'O', 'E6b')
    assert plan_.arms_not_run == ('E6a',)
    assert plan_.contrasts == (('E2w', 'C'), ('E2d', 'C'), ('O', 'C'), ('E6b', 'C'))
    assert not any(s.arm == 'E6a' for s in plan_.stages)
    assert not any(r.arm == 'E6a' for r in plan_.screen_rows)
    assert len(plan_.stages) == 24 and len(plan_.screen_rows) == 21
    assert 'E6a_seed1234' not in plan_.order and 'screen:E6a_seed7' not in plan_.order


def test_extension_plan_refuses_an_unfrozen_or_mismatching_v3_plan(tmp_path):
    unfrozen, _ = _v3_plan(tmp_path, frozen=False)
    with pytest.raises(ValueError, match='froz'):
        ext.extension_plan(unfrozen, _ext_config())
    frozen, _ = _v3_plan(tmp_path, k_selected=8)
    with pytest.raises(ValueError, match='k_selection'):     # config record != the plan's freeze
        ext.extension_plan(frozen, _ext_config(k_selected=16))
    # A v3 stage whose argv lost --final-eval none is refused (E4), as is one scoring validation.
    stages = list(frozen.stages)
    argv = list(stages[0].argv)
    argv[argv.index('--final-eval') + 1] = 'validation'
    stages[0] = study.Stage(**{**stages[0].__dict__, 'argv': tuple(argv)})
    tampered = study.Plan(**{**frozen.__dict__, 'stages': tuple(stages)})
    with pytest.raises(ValueError, match='final.eval'):
        ext.extension_plan(tampered, _ext_config())
    stages = list(frozen.stages)
    argv = [a for a in stages[3].argv if a not in ('--final-eval', 'none')]
    stages[3] = study.Stage(**{**stages[3].__dict__, 'argv': tuple(argv)})
    with pytest.raises(ValueError, match='final.eval'):
        ext.extension_plan(study.Plan(**{**frozen.__dict__, 'stages': tuple(stages)}), _ext_config())
    # Arm drift inside the v3 plan: the control stage at K* is not arm C.
    stages = list(frozen.stages)
    index = [s.name for s in stages].index('C_K8_seed7')
    stages[index] = study.Stage(**{**stages[index].__dict__, 'arm': 'B'})
    with pytest.raises(ValueError, match='arm'):
        ext.extension_plan(study.Plan(**{**frozen.__dict__, 'stages': tuple(stages)}), _ext_config())


def test_extension_plan_refuses_occupied_extension_outputs_and_a_screen_scored_twice(tmp_path):
    v3_plan, config = _v3_plan(tmp_path)
    root = Path(config.output_root)
    (root / 'E2d_seed2025').mkdir(parents=True)
    with pytest.raises(FileExistsError, match='occupied'):
        ext.extension_plan(v3_plan, _ext_config())
    (root / 'E2d_seed2025').rmdir()
    ext.extension_plan(v3_plan, _ext_config())
    (root / 'C_K8_seed1234' / study.SCREEN_DIRNAME).mkdir(parents=True)
    with pytest.raises(FileExistsError, match='twice'):
        ext.extension_plan(v3_plan, _ext_config())
    (root / 'C_K8_seed1234' / study.SCREEN_DIRNAME).rmdir()
    (root / 'C_K8_seed1234' / study.SCREEN_DIRNAME / offsets.OFFSET_SCREEN_DIRNAME).mkdir(parents=True)
    with pytest.raises(FileExistsError, match='twice'):
        ext.extension_plan(v3_plan, _ext_config())


def test_extension_plan_never_opens_inputs_or_executes(tmp_path, monkeypatch):
    import subprocess
    from comparison.standardized.clinical_graph_v2 import cei_v3_screen, contracts, tensorize
    v3_plan, _ = _v3_plan(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError('extension_plan must not execute, encode or load anything')
    for module, name in ((subprocess, 'run'), (subprocess, 'Popen'), (subprocess, 'check_call'),
                         (cei_v3_screen, 'encode_rows'), (contracts, 'iter_graphs_with_membership'),
                         (tensorize, 'encode_graph'), (study, '_default_model_factory'),
                         (study, 'score_screen'), (offsets, 'score_offset_screen')):
        monkeypatch.setattr(module, name, forbidden)
    plan_ = ext.extension_plan(v3_plan, _ext_config())
    assert len(plan_.stages) == 27
    assert not any(Path(s.output).exists() for s in plan_.stages)


# ------------------------------------------------------------ step 2: family bounds


def test_family_bounds_are_the_exact_order_statistics_on_one_to_thousand():
    deltas = np.arange(1, 1001, dtype=np.float64)
    bounds = ext.family_bounds(deltas)
    assert bounds['nominal'] == 25.0          # sorted[floor(1000 * 0.025) - 1] = sorted[24]
    assert bounds['corrected'] == 5.0         # sorted[floor(1000 * 0.025 / 5) - 1] = sorted[4]
    assert bounds['linear_quantile'] == 25.975   # np.quantile(deltas, 0.025, method='linear')
    assert bounds['linear_quantile'] == float(np.quantile(deltas, 0.025, method='linear'))
    assert bounds['m'] == 5
    assert bounds['nominal_index'] == 24 and bounds['corrected_index'] == 4
    assert bounds['nominal_p'] == 0.025 and bounds['corrected_p'] == 0.005
    assert bounds['corrected_tail_mass'] == 5 / 1000
    assert bounds['resamples'] == 1000
    assert bounds['rule'] == 'order statistic: sorted[floor(resamples * p) - 1]'
    assert bounds['decision_bound'] == 'corrected'
    assert bounds['linear_quantile_use'] == 'cross-reading only, never the decision'
    assert isinstance(bounds['nominal'], float) and isinstance(bounds['corrected'], float)


def test_family_bounds_sort_the_input_and_leave_it_unchanged():
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(np.arange(1, 1001, dtype=np.float64))
    copy = shuffled.copy()
    bounds = ext.family_bounds(shuffled)
    assert (bounds['nominal'], bounds['corrected'], bounds['linear_quantile']) == (25.0, 5.0, 25.975)
    assert np.array_equal(shuffled, copy)
    # Ties: the order statistic is still the (index)-th smallest value.
    tied = np.concatenate([np.full(30, -1.0), np.full(970, 1.0)])
    bounds = ext.family_bounds(rng.permutation(tied))
    assert bounds['nominal'] == -1.0 and bounds['corrected'] == -1.0
    tied = np.concatenate([np.full(5, -1.0), np.full(995, 1.0)])
    bounds = ext.family_bounds(rng.permutation(tied))
    assert bounds['corrected'] == -1.0 and bounds['nominal'] == 1.0


def test_family_bounds_keep_m_at_five_when_an_arm_is_missing():
    deltas = np.arange(1, 1001, dtype=np.float64)
    # The family is pre-registered with five contrasts; a missing arm does not shrink m (§7).
    for run_arms in (('E2w', 'E2d', 'O', 'E6b'), ('E2w',), ('E2w', 'E2d', 'O', 'E6a', 'E6b')):
        bounds = ext.family_bounds(deltas, m=ext.family_m(run_arms))
        assert bounds['m'] == 5 and bounds['corrected'] == 5.0
    assert ext.family_m(('E2w', 'E2d', 'O', 'E6b')) == 5
    assert ext.family_m(()) == 5
    assert ext.family_bounds(deltas)['m'] == 5
    # Any other m is refused: the correction is fixed before any result (D10.4).
    for m in (4, 6, 1, 0, -5, 5.0, True):
        with pytest.raises(ValueError, match='m'):
            ext.family_bounds(deltas, m=m)


def test_family_bounds_refuse_the_wrong_resample_count_or_non_finite_deltas():
    with pytest.raises(ValueError, match='1000'):
        ext.family_bounds(np.arange(1, 1000, dtype=np.float64))
    with pytest.raises(ValueError, match='1000'):
        ext.family_bounds(np.arange(1, 1002, dtype=np.float64))
    with pytest.raises(ValueError, match='1000'):
        ext.family_bounds(np.arange(1, 1001, dtype=np.float64).reshape(10, 100))
    bad = np.arange(1, 1001, dtype=np.float64)
    bad[3] = np.nan
    with pytest.raises(ValueError, match='finite'):
        ext.family_bounds(bad)
    bad[3] = np.inf
    with pytest.raises(ValueError, match='finite'):
        ext.family_bounds(bad)


def test_family_bounds_on_u5_bootstrap_output_are_distinct_from_the_v3_linear_rule():
    """The extension bound is applied to the U5 averaged deltas; the v3 C-vs-A rule keeps
    np.quantile(..., 'linear') (v3 §12.20) and the two are printed side by side, not mixed."""
    rng = np.random.default_rng(7)
    n, subjects = 90, 30
    y = rng.integers(0, 10, n)
    subj = np.repeat(np.arange(subjects), n // subjects).astype(str)
    arms = {}
    for arm, flip in (('E2w', 0.35), ('C', 0.55)):
        arms[arm] = {}
        for seed in SEEDS:
            r = np.random.default_rng(seed + int(flip * 100))
            pred = np.where(r.random(n) < flip, r.integers(0, 10, n), y)
            arms[arm][seed] = (y, pred, subj)
    deltas = study.paired_bootstrap(arms, [('E2w', 'C')])[('E2w', 'C')]
    assert deltas.shape == (1000,)
    bounds = ext.family_bounds(deltas)
    ordered = np.sort(deltas)
    assert bounds['nominal'] == float(ordered[24])
    assert bounds['corrected'] == float(ordered[4])
    assert bounds['corrected'] <= bounds['nominal']
    assert bounds['linear_quantile'] == float(np.quantile(deltas, 0.025, method='linear'))
    assert ordered[24] <= bounds['linear_quantile'] <= ordered[25]


# -------------------------------------------------------- step 3: extension decision


def _deltas(*, corrected=0.004, nominal=0.010, seed=1):
    """1,000 averaged deltas whose sorted[4] and sorted[24] are exactly the given values."""
    rng = np.random.default_rng(seed)
    values = np.sort(rng.normal(0.02, 0.005, 1000))
    values = values - values[4] + corrected            # sorted[4] == corrected
    if nominal is not None:
        values[5:25] = np.linspace(corrected, nominal, 21)[1:]   # sorted[24] == nominal
        values[25:] = np.maximum(values[25:], nominal)
    return rng.permutation(values)


def _scores(treatment=(0.42, 0.43, 0.44), control=(0.40, 0.41, 0.42)):
    return {'C': dict(zip(SEEDS, control)), 'E2w': dict(zip(SEEDS, treatment)),
            'E2d': dict(zip(SEEDS, treatment)), 'O': dict(zip(SEEDS, treatment)),
            'E6a': dict(zip(SEEDS, treatment)), 'E6b': dict(zip(SEEDS, treatment))}


def _family_deltas(**per_arm):
    return {(arm, 'C'): per_arm.get(arm, _deltas()) for arm in ('E2w', 'E2d', 'O', 'E6a', 'E6b')}


def test_decide_extension_applies_the_three_condition_rule_with_the_corrected_bound():
    deltas = _family_deltas()
    decision = ext.decide_extension(deltas, _scores())
    assert decision['m'] == 5
    assert decision['arms'] == ('E2w', 'E2d', 'O', 'E6a', 'E6b')
    assert decision['arms_not_run'] == ()
    assert decision['control'] == 'C'
    assert decision['metric'] == 'weighted_macro_f1'
    assert decision['resamples'] == 1000 and decision['bootstrap_seed'] == 2026
    assert decision['bound_rule'] == ext.BOUND_RULE
    assert decision['validation_evaluated'] is False and decision['test_evaluated'] is False
    assert list(decision['contrasts']) == ['E2w', 'E2d', 'O', 'E6a', 'E6b']
    result = decision['contrasts']['E2w']
    values = deltas[('E2w', 'C')]
    assert result['contrast'] == ['E2w', 'C']
    assert result['per_seed'] == {'1234': {'E2w': 0.42, 'C': 0.40}, '2025': {'E2w': 0.43, 'C': 0.41},
                                  '7': {'E2w': 0.44, 'C': 0.42}}
    assert result['seed_means'] == {'E2w': pytest.approx(0.43), 'C': pytest.approx(0.41)}
    assert result['point_delta'] == pytest.approx(0.02)
    assert result['bounds'] == ext.family_bounds(values)
    assert result['bounds']['corrected'] == pytest.approx(0.004)
    assert result['bounds']['nominal'] == pytest.approx(0.010)
    assert result['corrected_lower_bound'] == float(np.sort(values)[4])
    assert result['nominal_lower_bound'] == float(np.sort(values)[24])
    assert result['linear_quantile'] == float(np.quantile(values, 0.025, method='linear'))
    assert result['checks'] == {'e2w_beats_c_each_seed': True, 'e2w_mean_above_c': True,
                                'e2w_minus_c_corrected_lower_bound_above_zero': True}
    assert result['nominal_lower_bound_above_zero'] is True
    assert result['beats_control'] is True
    assert result['nominal_win'] is False
    assert result['statement'] == 'E2w beats C on this screen (family-corrected, m = 5)'
    assert result['claim'] == 'wider CEI v3 beats CEI v3 C on this screen'
    assert decision['contrasts']['E2d']['claim'] == 'deeper CEI v3 beats CEI v3 C on this screen'
    assert decision['contrasts']['O']['claim'] == ("validation-tuned per-class offsets improve "
                                                   "CEI v3 C's screen macro-F1")
    assert decision['contrasts']['E6a']['claim'] == ('the bidirectional edge view improves CEI v3 C '
                                                     'on this screen')
    assert decision['contrasts']['E6b']['claim'] == ('the additive comorbid pair term improves CEI '
                                                     'v3 C on this screen')
    assert decision['winners'] == ('E2w', 'E2d', 'O', 'E6a', 'E6b')
    assert decision['nominal_only'] == ()


def test_decide_extension_labels_a_nominal_win_and_never_counts_it_as_positive():
    # sorted[24] > 0 but sorted[4] <= 0: the nominal bound passes, the corrected one fails.
    deltas = _family_deltas(E6b=_deltas(corrected=-0.001, nominal=0.003))
    decision = ext.decide_extension(deltas, _scores())
    assert 'E6b' in decision['contrasts']
    result = decision['contrasts']['E6b']
    assert result['checks']['e6b_beats_c_each_seed'] is True
    assert result['checks']['e6b_mean_above_c'] is True
    assert result['checks']['e6b_minus_c_corrected_lower_bound_above_zero'] is False
    assert result['nominal_lower_bound_above_zero'] is True
    assert result['beats_control'] is False
    assert result['nominal_win'] is True
    assert result['statement'] == 'E6b: nominal win, not family-corrected'
    assert result['claim'] is None
    assert decision['winners'] == ('E2w', 'E2d', 'O', 'E6a')
    assert decision['nominal_only'] == ('E6b',)
    # A corrected bound exactly at zero is a failure (strictly above zero).
    deltas = _family_deltas(E2d=_deltas(corrected=0.0, nominal=0.003))
    result = ext.decide_extension(deltas, _scores())['contrasts']['E2d']
    assert result['beats_control'] is False and result['nominal_win'] is True
    # No nominal win either when the nominal bound is at or below zero.
    deltas = _family_deltas(O=_deltas(corrected=-0.002, nominal=0.0))
    result = ext.decide_extension(deltas, _scores())['contrasts']['O']
    assert result['beats_control'] is False and result['nominal_win'] is False
    assert result['statement'] == 'O: benefit not demonstrated on this screen'


@pytest.mark.parametrize('scores, failing', [
    (_scores(treatment=(0.42, 0.41, 0.44)), 'e2w_beats_c_each_seed'),          # seed tie
    (_scores(treatment=(0.42, 0.43, 0.44), control=(0.40, 0.41, 0.50)), 'e2w_beats_c_each_seed'),
    (_scores(treatment=(0.42, 0.43, 0.44), control=(0.44, 0.43, 0.42)), 'e2w_mean_above_c'),
])
def test_decide_extension_fails_closed_on_the_uncorrected_conditions(scores, failing):
    decision = ext.decide_extension(_family_deltas(), scores)
    assert 'E2w' in decision['contrasts']
    result = decision['contrasts']['E2w']
    assert result['checks'][failing] is False
    assert result['beats_control'] is False and result['nominal_win'] is False
    assert result['statement'] == 'E2w: benefit not demonstrated on this screen'
    assert 'E2w' not in decision['winners'] and 'E2w' not in decision['nominal_only']


def test_decide_extension_keeps_m_at_five_when_an_arm_is_missing():
    deltas = _family_deltas()
    del deltas[('E6a', 'C')]
    scores = _scores()
    del scores['E6a']
    decision = ext.decide_extension(deltas, scores)
    assert decision['m'] == 5
    assert decision['arms'] == ('E2w', 'E2d', 'O', 'E6b')
    assert decision['arms_not_run'] == ('E6a',)
    assert list(decision['contrasts']) == ['E2w', 'E2d', 'O', 'E6b']
    for result in decision['contrasts'].values():
        assert result['bounds']['m'] == 5 and result['bounds']['corrected_index'] == 4
        assert result['statement'].endswith('(family-corrected, m = 5)')


@pytest.mark.parametrize('kind', ['m_four', 'm_six', 'not_vs_c', 'foreign_arm', 'short_deltas',
                                  'missing_seed', 'missing_scores', 'no_contrast', 'nan_delta',
                                  'control_as_arm'])
def test_decide_extension_refuses_family_drift_and_malformed_inputs(kind):
    deltas, scores, kwargs = _family_deltas(), _scores(), {}
    if kind == 'm_four':
        kwargs['m'] = 4
    elif kind == 'm_six':
        kwargs['m'] = 6
    elif kind == 'not_vs_c':
        deltas[('E2w', 'A')] = deltas.pop(('E2w', 'C'))
        scores['A'] = scores['C']
    elif kind == 'foreign_arm':
        deltas[('E6c', 'C')] = _deltas()
        scores['E6c'] = scores['E6b']
    elif kind == 'short_deltas':
        deltas[('O', 'C')] = deltas[('O', 'C')][:999]
    elif kind == 'missing_seed':
        del scores['E6b'][7]
    elif kind == 'missing_scores':
        del scores['E2d']
    elif kind == 'no_contrast':
        deltas = {}
    elif kind == 'nan_delta':
        deltas[('E6a', 'C')] = deltas[('E6a', 'C')].copy()
        deltas[('E6a', 'C')][10] = np.nan
    elif kind == 'control_as_arm':
        deltas[('C', 'C')] = _deltas()
    with pytest.raises(ValueError):
        ext.decide_extension(deltas, scores, **kwargs)


# ---- O rows from C's stored screen logits + frozen delta (no inference) ----


def _stored_c_screen(tmp_path, seed, *, n=40, rng_seed=3):
    """A synthetic C screen row (as U5 writes it): logits.npz + screen_result.json dict."""
    rng = np.random.default_rng(rng_seed + seed)
    y = rng.integers(0, 10, n)
    logits = rng.normal(0.0, 1.0, (n, 10)).astype(np.float32)
    logits[np.arange(n), y] += rng.random(n).astype(np.float32) * 2.0
    subjects = np.asarray([f'S{i // 2}' for i in range(n)]).astype(str)
    sample_ids = np.asarray([f'row{seed}_{i}' for i in range(n)]).astype(str)
    out = tmp_path / f'C_K8_seed{seed}' / study.SCREEN_DIRNAME
    out.mkdir(parents=True)
    logits_path = out / 'logits.npz'
    np.savez_compressed(logits_path, logits=logits, y=y, subjects=subjects, sample_ids=sample_ids)
    result = {
        'arm': 'C', 'seed': seed, 'k': 8,
        'checkpoint_sha256': hashlib.sha256(f'best.pt {seed}'.encode()).hexdigest(),
        'binding_sha256': hashlib.sha256(f'binding {seed}'.encode()).hexdigest(),
        'k_selection_sha256': 'k' * 64, 'screen_record_sha256': 's' * 64, 'row_count': n,
        'macro_f1': float(study.weighted_macro_f1(y, logits.argmax(1))),
        'logits_path': str(logits_path),
        'logits_sha256': hashlib.sha256(np.ascontiguousarray(logits).tobytes()).hexdigest(),
        'proba_path': str(out / 'proba.npz'), 'proba_sha256': 'p' * 64,
        'absence_share_mean': 0.1, 'validation_evaluated': False, 'test_evaluated': False,
    }
    return result, logits, y, subjects


def _offset_record_for(c_result, delta_int, approval_hash):
    """A frozen delta record bound to `c_result`'s checkpoint (synthetic, X3 field set)."""
    return {
        'version': offsets.OFFSET_RECORD_VERSION, 'arm': 'O', 'control_arm': 'C',
        'stage': f"C_K8_seed{c_result['seed']}", 'seed': c_result['seed'], 'k': 8,
        'delta_int': list(delta_int), 'delta': [v / 10 for v in delta_int],
        'grid': list(range(-20, 21)), 'sweeps': 5, 'delta_scale': 10,
        'metric': offsets.METRIC_NAME, 'optimizer_rule': offsets.OPTIMIZER_RULE,
        'checkpoint_sha256': c_result['checkpoint_sha256'],
        'binding_sha256': c_result['binding_sha256'],
        'k_selection_sha256': c_result['k_selection_sha256'],
        'validation_sample_ids_sha256': 'v' * 64, 'validation_row_count': 4254,
        'validation_logits_sha256': 'l' * 64, 'validation_logits_path': '/dev/null',
        'approval_record_sha256': approval_hash, 'validation_result_sha256': 'r' * 64,
        'tuning_macro_f1_before': 0.4, 'tuning_macro_f1_after': 0.41,
        'tuning_score_caveat': offsets.TUNING_SCORE_CAVEAT,
        'validation_evaluated': True, 'test_evaluated': False, 'screen_read_before_freeze': False,
    }


def _o_fixture(tmp_path):
    approval = hashlib.sha256(b'approval').hexdigest()
    deltas = {1234: [3, -2, 0, 0, 5, 0, -7, 0, 0, 1], 2025: [0, 0, 4, 0, 0, -3, 0, 0, 2, 0],
              7: [-1, 0, 0, 6, 0, 0, 0, -4, 0, 0]}
    c_results, records, arrays = {}, {}, {}
    for seed in SEEDS:
        result, logits, y, subjects = _stored_c_screen(tmp_path, seed)
        c_results[seed] = result
        records[seed] = _offset_record_for(result, deltas[seed], approval)
        arrays[seed] = (logits, y, subjects)
    return approval, c_results, records, arrays


def test_offset_screen_rows_build_o_from_stored_logits_and_delta_without_inference(tmp_path, monkeypatch):
    from comparison.standardized.clinical_graph_v2 import cei_v3_screen, contracts, tensorize
    approval, c_results, records, arrays = _o_fixture(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError('O rows must come from stored logits: no loader, encoder or model')
    for module, name in ((cei_v3_screen, 'encode_rows'), (contracts, 'iter_graphs_with_membership'),
                         (tensorize, 'encode_graph'), (study, '_default_model_factory'),
                         (study, 'score_screen')):
        monkeypatch.setattr(module, name, forbidden)
    rows = ext.offset_screen_rows(c_results, records, approval_record_sha256=approval)
    assert set(rows) == set(SEEDS)
    for seed in SEEDS:
        logits, y, subjects = arrays[seed]
        row = rows[seed]
        assert (row['arm'], row['control_arm'], row['seed'], row['k']) == ('O', 'C', seed, 8)
        assert row['reinference'] is False
        assert row['checkpoint_sha256'] == c_results[seed]['checkpoint_sha256']
        assert row['screen_logits_sha256'] == c_results[seed]['logits_sha256']
        assert row['offset_record_sha256'] == offsets.offset_record_sha256(records[seed])
        assert row['approval_record_sha256'] == approval
        expected = offsets.apply_offsets(logits, np.asarray(records[seed]['delta_int']))
        with np.load(row['pred_path'], allow_pickle=False) as saved:
            assert np.array_equal(saved['pred'], expected)
            assert np.array_equal(saved['y'], y)
        assert row['macro_f1'] == study.weighted_macro_f1(y, expected)
        assert Path(row['pred_path']).parent == (Path(c_results[seed]['logits_path']).parent
                                                 / offsets.OFFSET_SCREEN_DIRNAME)
        # The screen row loader reads O from o_pred.npz and C from logits.npz (argmax).
        o_y, o_pred, o_subjects = ext.load_screen_row(row)
        assert np.array_equal(o_pred, expected) and np.array_equal(o_y, y)
        assert list(o_subjects) == list(subjects)
        c_y, c_pred, c_subjects = ext.load_screen_row(c_results[seed])
        assert np.array_equal(c_pred, logits.argmax(1)) and np.array_equal(c_y, y)
        assert list(c_subjects) == list(subjects)
    # The loaded rows feed U5 paired_bootstrap directly, paired on identical rows per seed.
    arms = {'O': {seed: ext.load_screen_row(rows[seed]) for seed in SEEDS}}
    for seed in SEEDS:                       # rows differ across seeds in this synthetic
        arms['C'] = {s: ext.load_screen_row(c_results[s]) for s in SEEDS}
    with pytest.raises(ValueError, match='paired'):
        study.paired_bootstrap(arms, [('O', 'C')])


@pytest.mark.parametrize('kind', ['missing_seed', 'swapped_records', 'approval_mismatch',
                                  'no_logits_hash', 'tampered_logits', 'extra_seed'])
def test_offset_screen_rows_refuse_unbound_or_mismatched_inputs(tmp_path, kind):
    approval, c_results, records, _ = _o_fixture(tmp_path)
    if kind == 'missing_seed':
        del records[7]
    elif kind == 'swapped_records':
        records[1234], records[2025] = records[2025], records[1234]
    elif kind == 'approval_mismatch':
        approval = '0' * 64
    elif kind == 'no_logits_hash':
        del c_results[2025]['logits_sha256']
    elif kind == 'tampered_logits':
        path = Path(c_results[7]['logits_path'])
        with np.load(path, allow_pickle=False) as saved:
            arrays = {k: saved[k] for k in saved.files}
        arrays['logits'] = arrays['logits'] * np.float32(2.0)
        path.unlink()
        np.savez_compressed(path, **arrays)
    elif kind == 'extra_seed':
        c_results[99] = c_results[7]
        records[99] = records[7]
    with pytest.raises(ValueError):
        ext.offset_screen_rows(c_results, records, approval_record_sha256=approval)
    for seed in SEEDS:
        assert not (Path(c_results[seed]['logits_path']).parent / offsets.OFFSET_SCREEN_DIRNAME).exists()


def test_load_screen_row_refuses_stored_logits_that_differ_from_the_bound_hash(tmp_path):
    result, _, _, _ = _stored_c_screen(tmp_path, 1234)
    path = Path(result['logits_path'])
    with np.load(path, allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in saved.files}
    arrays['logits'] = arrays['logits'].copy()
    arrays['logits'][0, 0] += np.float32(1.0)
    path.unlink()
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match='logits'):
        ext.load_screen_row(result)


def test_load_screen_row_refuses_o_predictions_that_differ_from_the_bound_hash(tmp_path):
    approval, c_results, records, _ = _o_fixture(tmp_path)
    rows = ext.offset_screen_rows(c_results, records, approval_record_sha256=approval)
    row = rows[7]
    path = Path(row['pred_path'])
    with np.load(path, allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in saved.files}
    arrays['pred'] = arrays['pred'].copy()
    arrays['pred'][0] = (arrays['pred'][0] + 1) % 10
    path.unlink()
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match='O predictions'):
        ext.load_screen_row(row)


def test_offset_screen_rows_approval_mismatch_on_the_last_seed_leaves_no_o_output(tmp_path):
    approval, c_results, records, _ = _o_fixture(tmp_path)
    records[7]['approval_record_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        ext.offset_screen_rows(c_results, records, approval_record_sha256=approval)
    for seed in SEEDS:
        assert not (Path(c_results[seed]['logits_path']).parent / offsets.OFFSET_SCREEN_DIRNAME).exists()


def test_offset_screen_rows_occupied_o_output_on_the_last_seed_leaves_no_new_o_output(tmp_path):
    approval, c_results, records, _ = _o_fixture(tmp_path)
    last_out = Path(c_results[7]['logits_path']).parent / offsets.OFFSET_SCREEN_DIRNAME
    last_out.mkdir()
    with pytest.raises(FileExistsError):
        ext.offset_screen_rows(c_results, records, approval_record_sha256=approval)
    for seed in SEEDS[:2]:
        assert not (Path(c_results[seed]['logits_path']).parent / offsets.OFFSET_SCREEN_DIRNAME).exists()


# ---- secondary E6b metric: comorbid-block share and non-empty pair-set fraction ----


def test_comorbid_share_and_nonempty_fraction_follow_the_absence_share_definition():
    import torch
    parts = {
        'node_contributions': torch.tensor([[1.0, -1.0], [0.0, 0.0], [2.0, 0.0], [0.0, 0.0]]),
        'edge_contributions': torch.tensor([[0.5, 0.5]]),
        'pair_contributions': torch.tensor([[0.0, 1.0]]),
        'pairs': torch.tensor([[0], [1]]),
        'absence_contributions': torch.tensor([[1.0, 0.5], [3.0, 0.0]]),
        'absence_items': torch.tensor([[0, 1], [0, 1]]),
        'comorbid_contributions': torch.tensor([[0.5, 1.0], [1.0, 0.0]]),
        'comorbid_pairs': torch.tensor([[0, 2], [1, 3]]),      # one pair in graph 0, one in graph 1
        'comorbid_gates': torch.tensor([0.5, 0.5]),
        'comorbid_denominator': torch.tensor([[1.5, 1.5], [1.5, 1.5], [1.0, 1.0]]),
    }
    batch_index = torch.tensor([0, 0, 1, 1])
    edge_index = torch.tensor([[0], [1]])
    share = ext.comorbid_share(parts, batch_index=batch_index, edge_index=edge_index, graph_count=3)
    # graph 0: |comorbid| = (0.5, 1.0); totals = node (1, 1) + edge (0.5, 0.5) + pair (0, 1)
    #          + absence (1.0, 0.5) + comorbid (0.5, 1.0) = (3.0, 4.0) -> mean(0.5/3, 1.0/4)
    # graph 1: |comorbid| = (1.0, 0); totals = node (2, 0) + absence (3, 0) + comorbid (1, 0)
    #          = (6, 0) -> classes (1/6, 0) -> mean 1/12
    # graph 2: nothing -> 0
    assert share.shape == (3,)
    assert share.tolist() == pytest.approx([(0.5 / 3.0 + 1.0 / 4.0) / 2, 1.0 / 12.0, 0.0])
    nonempty = ext.comorbid_nonempty(parts, batch_index=batch_index, graph_count=3)
    assert nonempty.dtype == bool and nonempty.tolist() == [True, True, False]
    assert float(nonempty.mean()) == pytest.approx(2 / 3)
    # Without the block (arm C parts) the share is zero and no graph has a pair.
    without = {k: v for k, v in parts.items() if not k.startswith('comorbid_')}
    assert ext.comorbid_share(without, batch_index=batch_index, edge_index=edge_index,
                              graph_count=3).tolist() == [0.0, 0.0, 0.0]
    assert ext.comorbid_nonempty(without, batch_index=batch_index, graph_count=3).tolist() == [False] * 3
