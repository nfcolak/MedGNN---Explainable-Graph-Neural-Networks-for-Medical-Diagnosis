"""Synthetic tests for the CEI-GNN v3 study core (unit U5).

Everything here is synthetic: no data file, run directory, checkpoint or fold is opened.
Spec: v3 design §6, §11.2 as amended by §12 (F4–F7, F17, F18, F20, item 21), extensions
spec §4 (E4, E7) and §9 U5.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from comparison.standardized.clinical_graph_v2 import cei_v3_study as study

TRAIN_PY = 'comparison.standardized.clinical_graph_v2.train'


# ------------------------------------------------------------------ fixtures


def _inputs(tmp_path):
    artifact = tmp_path / 'artifact'
    artifact.mkdir()
    targets = tmp_path / 'targets.csv'
    targets.write_text('sample_id,target,split,subject_id\n')
    canonical = tmp_path / 'canonical.json'
    canonical.write_text('{}')
    root = tmp_path / 'study_root'
    return artifact, targets, canonical, root


def _config(tmp_path, **overrides):
    artifact, targets, canonical, root = _inputs(tmp_path)
    fields = dict(artifact=str(artifact), targets=str(targets), canonical=str(canonical),
                  output_root=str(root))
    fields.update(overrides)
    return study.StudyConfig(**fields)


def _k_selection(k_selected=8):
    """Minimal frozen K record as `plan` consumes it (full record: step 2)."""
    return {'version': study.K_SELECTION_VERSION, 'k_grid': list(study.K_GRID),
            'k_selected': k_selected, 'rule': study.K_SELECTION_RULE}


def _binding(stage, *, root, **overrides):
    """A synthetic `binding.json` in the train.py layout for one planned stage."""
    arm = stage.arm
    ple_active, absence_active = {'A': (False, False), 'B': (True, False),
                                  'C': (True, True)}[arm]
    method_config = {
        'method': 'cei_gnn_v3', 'arm': arm, 'k': stage.k, 'v3_state_path': stage.v3_state,
        'v3_state_sha256': 'c' * 64, 'knot_table_sha256': 'd' * 64,
        'effective_settings': {'arm': arm, 'k': stage.k, 'v3_state': stage.v3_state,
                               'encoder_depth': 1, 'comorbid_block': 0},
        'encoder_depth': 1, 'comorbid_block': 0, 'edge_direction': 'forward', 'hidden': 128,
        'ple_active': ple_active, 'absence_active': absence_active,
        'common_init_identical_to_c': True, 'pair_mode': 'additive',
        'preprocessing_sha256': 'b' * 64,
        'architecture': {'parameter_count': 1000, 'active_parameter_count': 900,
                         'inactive_parameter_count': 100, 'k': stage.k},
    }
    binding = {
        'method': 'cei_gnn_v3', 'method_config': method_config,
        'method_overrides': ['option:arm', 'option:k', 'option:v3_state'],
        'artifact': str(Path(root).parent / 'artifact'), 'artifact_graphs_sha256': 'a' * 64,
        'artifact_visit_membership_sha256': 'a' * 64, 'targets_sha256': 't' * 64,
        'target_binding_sha256': 't' * 64, 'label_order': [f'L{i}' for i in range(10)],
        'source_code': {}, 'preprocessing_sha256': 'b' * 64, 'seed': stage.seed,
        'sample_seed': 1234, 'selection_fold': 'dev', 'final_eval': 'none',
        'train_limit': 10000, 'dev_limit': 5000, 'epochs': 40, 'patience': 40,
        'test_evaluated': False, 'parameter_count': 1000, 'active_parameter_count': 900,
        'weights': 'sqrt_inverse', 'edge_direction': 'forward', 'edges': 'all',
        'top_k_labels': 10, 'kept_label_indices': list(range(10)), 'message_passing': True,
        'edge_payload': True, 'num_classes': 10, 'input_contract_version': 'clinical_inputs_v3',
        'hidden': 128, 'layers': 1, 'dropout': 0.3, 'lr': 1.79e-3, 'weight_decay': 4.3e-5,
        'batch_size': 128, 'min_prior_visits': 0, 'token_min_count': 20,
        'split_sample_ids_sha256': {'train': 'e' * 64, 'dev': 'f' * 64, 'validation': '9' * 64},
        'counts': {'train': 10000, 'dev': 5000, 'validation': 4254},
        'selected_dev': {'epoch': 12, 'epoch_index': 11, 'metric': 'macro_f1',
                         'metric_value': 0.41, 'prediction_sha256': '1' * 64,
                         'sample_ids_sha256': 'f' * 64},
    }
    binding.update(overrides)
    return binding


def _result(binding, **overrides):
    result = {'status': 'completed', 'binding': binding, 'metrics': None, 'per_class': None,
              'dev_metrics': {'macro_f1': binding['selected_dev']['metric_value']},
              'selection_fold': 'dev', 'validation_evaluations': 0, 'test_evaluated': False,
              'history': [], 'total_seconds': 120.0}
    result.update(overrides)
    return result


def _stage(plan_, name):
    matches = [stage for stage in plan_.stages if stage.name == name]
    assert len(matches) == 1, f'plan must contain exactly one stage {name!r}'
    return matches[0]


# ---------------------------------------------------------- step 1: plan lock


def test_plan_orders_state_fit_c_grid_freeze_a_b_then_screen(tmp_path):
    plan_ = study.plan(_config(tmp_path))
    c_names = [f'C_K{k}_seed{seed}' for k in (4, 8, 16) for seed in (1234, 2025, 7)]
    assert plan_.order[:3] == ('fit_v3_state:K4', 'fit_v3_state:K8', 'fit_v3_state:K16')
    assert list(plan_.order[3:12]) == c_names
    assert plan_.order[12] == 'k_selection'
    assert list(plan_.order[13:19]) == [f'{arm}_seed{seed}' for arm in 'AB'
                                        for seed in (1234, 2025, 7)]
    assert plan_.order[19:] == ('screen',)
    assert [stage.name for stage in plan_.stages] == c_names + list(plan_.order[13:19])
    assert [stage.phase for stage in plan_.stages] == ['c_grid'] * 9 + ['post_freeze'] * 6
    assert plan_.k_selection_sha256 is None
    assert plan_.k_selection_path == str(Path(plan_.output_root) / 'k_selection.json')
    assert plan_.v3_state_paths == {k: str(Path(plan_.output_root) / 'v3_state' / f'K{k}.json')
                                    for k in (4, 8, 16)}
    # A/B stages cannot be materialised before the K freeze (F17).
    for stage in plan_.stages[9:]:
        assert stage.k is None and stage.argv == () and stage.v3_state == ''


def test_c_stage_argv_is_exact_and_carries_the_mandatory_flags(tmp_path):
    config = _config(tmp_path)
    plan_ = study.plan(config)
    stage = _stage(plan_, 'C_K4_seed2025')
    root = Path(config.output_root).resolve()
    expected = (
        sys.executable, '-m', TRAIN_PY,
        '--artifact', str(Path(config.artifact).resolve()),
        '--targets', str(Path(config.targets).resolve()),
        '--canonical', str(Path(config.canonical).resolve()),
        '--output', str(root / 'C_K4_seed2025'), '--method', 'cei_gnn_v3',
        '--train-limit', '10000', '--dev-limit', '5000', '--sample-seed', '1234',
        '--seed', '2025', '--top-k-labels', '10', '--edges', 'all',
        '--edge-direction', 'forward', '--weights', 'sqrt_inverse',
        '--selection-fold', 'dev', '--final-eval', 'none', '--epochs', '40',
        '--patience', '40', '--method-option', 'arm=C',
        '--method-option', f'v3_state={root / "v3_state" / "K4.json"}',
        '--method-option', 'k=4')
    assert stage.argv == expected
    assert stage.output == str(root / 'C_K4_seed2025')
    assert (stage.arm, stage.k, stage.seed, stage.phase) == ('C', 4, 2025, 'c_grid')
    for stage in plan_.stages[:9]:
        argv = ' '.join(stage.argv)
        assert '--selection-fold dev --final-eval none' in argv
        assert '--dev-limit 5000' in argv
        assert '--final-eval validation' not in argv


def test_a_and_b_stages_are_materialised_at_the_frozen_k_after_the_freeze(tmp_path):
    config = _config(tmp_path, k_selection=_k_selection(k_selected=16))
    plan_ = study.plan(config)
    root = Path(config.output_root).resolve()
    stage = _stage(plan_, 'A_seed7')
    assert (stage.arm, stage.k, stage.seed, stage.phase) == ('A', 16, 7, 'post_freeze')
    assert stage.v3_state == str(root / 'v3_state' / 'K16.json')
    assert stage.argv[-6:] == ('--method-option', 'arm=A',
                               '--method-option', f'v3_state={root / "v3_state" / "K16.json"}',
                               '--method-option', 'k=16')
    assert stage.argv[stage.argv.index('--seed') + 1] == '7'
    assert '--selection-fold' in stage.argv and '--final-eval' in stage.argv
    assert stage.argv[stage.argv.index('--final-eval') + 1] == 'none'
    assert stage.argv[stage.argv.index('--dev-limit') + 1] == '5000'
    assert _stage(plan_, 'B_seed1234').argv[-5] == 'arm=B'
    assert plan_.k_selection_sha256 == study.k_selection_sha256(config.k_selection)
    assert len(plan_.k_selection_sha256) == 64


def test_plan_refuses_occupied_stage_output(tmp_path):
    config = _config(tmp_path)
    occupied = Path(config.output_root) / 'C_K8_seed7'
    occupied.mkdir(parents=True)
    with pytest.raises(FileExistsError, match='occupied'):
        study.plan(config)
    # After the freeze the nine C outputs exist by construction; A/B outputs must not.
    frozen = _config(tmp_path, k_selection=_k_selection())
    study.plan(frozen)
    (Path(frozen.output_root) / 'B_seed2025').mkdir(parents=True)
    with pytest.raises(FileExistsError, match='occupied'):
        study.plan(frozen)


@pytest.mark.parametrize('overrides, message', [
    ({'final_eval': 'validation'}, 'final_eval'),
    ({'selection_fold': 'validation'}, 'selection_fold'),
    ({'selection_fold': 'test'}, 'selection_fold'),
    ({'budget': (10000, 4000, 40)}, 'dev'),
    ({'budget': (10000, 5000, 20)}, 'budget'),
    ({'k_grid': (4, 8)}, 'grid'),
    ({'seeds': (1234, 2025)}, 'seeds'),
    ({'arms': ('A', 'C')}, 'arms'),
])
def test_plan_lock_refuses_protocol_drift(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        study.plan(_config(tmp_path, **overrides))


def test_plan_refuses_a_frozen_k_outside_the_grid_or_a_foreign_rule(tmp_path):
    with pytest.raises(ValueError, match='K'):
        study.plan(_config(tmp_path, k_selection=_k_selection(k_selected=32)))
    record = _k_selection()
    record['rule'] = 'argmax of screen macro-F1'
    with pytest.raises(ValueError, match='rule'):
        study.plan(_config(tmp_path, k_selection=record))


def test_plan_refuses_relative_or_missing_inputs(tmp_path):
    config = _config(tmp_path, output_root='relative/root')
    with pytest.raises(ValueError, match='absolute'):
        study.plan(config)
    config = _config(tmp_path, targets=str(tmp_path / 'absent.csv'))
    with pytest.raises(FileNotFoundError):
        study.plan(config)


def test_check_stage_result_accepts_a_dev_only_completed_stage(tmp_path):
    plan_ = study.plan(_config(tmp_path))
    stage = _stage(plan_, 'C_K4_seed1234')
    stage_dir = Path(stage.output)
    stage_dir.mkdir(parents=True)
    binding = _binding(stage, root=plan_.output_root)
    assert study.check_stage_result(stage_dir, binding, _result(binding)) is None


@pytest.mark.parametrize('kind', ['metrics', 'validation_npz', 'validation_evaluations',
                                  'selected_validation', 'test_evaluated', 'test_npz',
                                  'status', 'binding_mismatch', 'final_eval'])
def test_check_stage_result_refuses_validation_or_test_traces(tmp_path, kind):
    plan_ = study.plan(_config(tmp_path))
    stage = _stage(plan_, 'C_K4_seed1234')
    stage_dir = Path(stage.output)
    stage_dir.mkdir(parents=True)
    binding = _binding(stage, root=plan_.output_root)
    result = _result(binding)
    if kind == 'metrics':
        result['metrics'] = {'macro_f1': 0.5}
    elif kind == 'validation_npz':
        (stage_dir / 'validation.npz').write_bytes(b'x')
    elif kind == 'validation_evaluations':
        result['validation_evaluations'] = 1
    elif kind == 'selected_validation':
        binding['selected_validation'] = {'metric_value': 0.5}
        result['binding'] = binding
    elif kind == 'test_evaluated':
        result['test_evaluated'] = True
    elif kind == 'test_npz':
        (stage_dir / 'test.npz').write_bytes(b'x')
    elif kind == 'status':
        result['status'] = 'running'
    elif kind == 'binding_mismatch':
        result['binding'] = dict(binding, seed=99)
    elif kind == 'final_eval':
        binding['final_eval'] = 'validation'
        result['binding'] = binding
    with pytest.raises(ValueError):
        study.check_stage_result(stage_dir, binding, result)


def test_validate_v3_binding_passes_on_a_conforming_c_stage(tmp_path):
    plan_ = study.plan(_config(tmp_path))
    stage = _stage(plan_, 'C_K16_seed7')
    assert study.validate_v3_binding(_binding(stage, root=plan_.output_root), stage) is None


@pytest.mark.parametrize('mutate, message', [
    (lambda b: b['method_config'].__setitem__('arm', 'B'), 'arm'),
    (lambda b: b['method_config']['effective_settings'].__setitem__('arm', 'A'), 'arm'),
    (lambda b: b.__setitem__('seed', 2025), 'seed'),
    (lambda b: b['method_config'].__setitem__('k', 8), 'k'),
    (lambda b: b['method_config']['effective_settings'].__setitem__(
        'v3_state', '/elsewhere/K16.json'), 'v3_state'),
    (lambda b: b['method_config'].__setitem__('ple_active', False), 'ple_active'),
    (lambda b: b['method_config'].__setitem__('absence_active', False), 'absence_active'),
    (lambda b: b['method_config'].__setitem__('encoder_depth', 2), 'encoder_depth'),
    (lambda b: b['method_config'].__setitem__('comorbid_block', 1), 'comorbid_block'),
    (lambda b: b['method_config'].__setitem__('hidden', 256), 'hidden'),
    (lambda b: b.__setitem__('method', 'cei_gnn_v2'), 'method'),
    (lambda b: b.__setitem__('epochs', 20), 'epochs'),
    (lambda b: b.__setitem__('patience', 10), 'patience'),
    (lambda b: b.__setitem__('sample_seed', 7), 'sample_seed'),
    (lambda b: b.__setitem__('edge_direction', 'bidirectional'), 'edge_direction'),
    (lambda b: b.__setitem__('train_limit', 34305), 'train_limit'),
    (lambda b: b.__setitem__('dev_limit', 4000), 'dev_limit'),
    (lambda b: b.__setitem__('final_eval', 'validation'), 'final_eval'),
    (lambda b: b.__setitem__('selection_fold', 'validation'), 'selection_fold'),
    (lambda b: b.__setitem__('test_evaluated', True), 'test'),
    (lambda b: b['split_sample_ids_sha256'].__setitem__('test', '0' * 64), 'test'),
    (lambda b: b['counts'].__setitem__('test', 1), 'test'),
    (lambda b: b.__setitem__('preprocessing_sha256', '0' * 64), 'preprocessing_sha256'),
    (lambda b: b.pop('selected_dev'), 'selected_dev'),
])
def test_validate_v3_binding_refuses_arm_drift_and_test_access(tmp_path, mutate, message):
    plan_ = study.plan(_config(tmp_path))
    stage = _stage(plan_, 'C_K16_seed7')
    binding = _binding(stage, root=plan_.output_root)
    mutate(binding)
    with pytest.raises(ValueError, match=message):
        study.validate_v3_binding(binding, stage)


def test_validate_v3_binding_refuses_screen_reuse_for_selection(tmp_path):
    plan_ = study.plan(_config(tmp_path))
    stage = _stage(plan_, 'C_K4_seed1234')
    binding = _binding(stage, root=plan_.output_root)
    record = {'fold': 'screen', 'screen_sample_ids_sha256': 's' * 64}
    assert study.validate_v3_binding(binding, stage, screen_record=record) is None
    reused = _binding(stage, root=plan_.output_root)
    reused['split_sample_ids_sha256']['dev'] = 's' * 64
    reused['selected_dev']['sample_ids_sha256'] = 's' * 64
    with pytest.raises(ValueError, match='screen'):
        study.validate_v3_binding(reused, stage, screen_record=record)
    extra = _binding(stage, root=plan_.output_root)
    extra['split_sample_ids_sha256']['screen'] = 'z' * 64
    with pytest.raises(ValueError, match='screen'):
        study.validate_v3_binding(extra, stage)


def test_post_freeze_binding_needs_a_study_binding_with_the_freeze_hash(tmp_path):
    record = _k_selection(k_selected=8)
    config = _config(tmp_path, k_selection=record)
    plan_ = study.plan(config)
    stage = _stage(plan_, 'A_seed1234')
    stage_dir = Path(stage.output)
    stage_dir.mkdir(parents=True)
    binding = _binding(stage, root=plan_.output_root)
    (stage_dir / 'binding.json').write_text(json.dumps(binding, indent=2, sort_keys=True) + '\n')
    with pytest.raises(ValueError, match='k_selection_sha256'):
        study.validate_v3_binding(binding, stage, k_selection=record)
    study_binding = study.write_study_binding(stage_dir, stage, record)
    expected_binding_sha = hashlib.sha256((stage_dir / 'binding.json').read_bytes()).hexdigest()
    assert study_binding == {
        'stage': 'A_seed1234', 'arm': 'A', 'seed': 1234, 'k': 8, 'k_grid': [4, 8, 16],
        'k_selected': 8, 'k_selection_rule': study.K_SELECTION_RULE,
        'k_selection_sha256': study.k_selection_sha256(record),
        'binding_sha256': expected_binding_sha, 'v3_state': stage.v3_state,
        'selection_fold': 'dev', 'final_eval': 'none'}
    assert json.loads((stage_dir / 'study_binding.json').read_text()) == study_binding
    assert study.validate_v3_binding(binding, stage, k_selection=record,
                                     study_binding=study_binding) is None
    wrong_hash = dict(study_binding, k_selection_sha256='0' * 64)
    with pytest.raises(ValueError, match='k_selection_sha256'):
        study.validate_v3_binding(binding, stage, k_selection=record, study_binding=wrong_hash)
    wrong_k = dict(study_binding, k_selected=4)
    with pytest.raises(ValueError, match='K'):
        study.validate_v3_binding(binding, stage, k_selection=record, study_binding=wrong_k)
    tampered = dict(study_binding, binding_sha256='0' * 64)
    with pytest.raises(ValueError, match='binding_sha256'):
        study.validate_v3_binding(binding, stage, k_selection=record, study_binding=tampered)
    with pytest.raises(FileExistsError):
        study.write_study_binding(stage_dir, stage, record)


def test_post_freeze_stage_k_must_match_the_frozen_k(tmp_path):
    record = _k_selection(k_selected=8)
    plan_ = study.plan(_config(tmp_path, k_selection=record))
    stage = _stage(plan_, 'B_seed7')
    binding = _binding(stage, root=plan_.output_root)
    binding['method_config']['k'] = 16
    binding['method_config']['effective_settings']['k'] = 16
    binding['method_config']['architecture']['k'] = 16
    with pytest.raises(ValueError, match='k'):
        study.validate_v3_binding(binding, stage, k_selection=record)
