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
    return list(zip(argv, argv[1:]))


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
