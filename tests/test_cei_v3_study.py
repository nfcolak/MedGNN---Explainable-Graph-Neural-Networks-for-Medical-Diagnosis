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
    artifact.mkdir(exist_ok=True)
    targets = tmp_path / 'targets.csv'
    if not targets.exists():
        targets.write_text('sample_id,target,split,subject_id\n')
    canonical = tmp_path / 'canonical.json'
    if not canonical.exists():
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


# -------------------------------------------------------- step 2: k selection

from comparison.standardized.clinical_graph_v2.cei_v3_absence import fit_universe  # noqa: E402
from comparison.standardized.clinical_graph_v2.cei_v3_ple import fit_knots  # noqa: E402
from comparison.standardized.clinical_graph_v2.methods import plugin_cei_gnn_v3 as plugin  # noqa: E402

TOKENS = ['complaint:a', 'measurement:lab1', 'vital:hr', 'analyte:x', 'measurement:lab2']
VOCAB = {token: i + 1 for i, token in enumerate(TOKENS)}
LAYOUT = ['kind:complaint', 'scaled_value', 'has_value', 'time_signed_log']
PREP_SHA = hashlib.sha256(b'synthetic preprocessing state').hexdigest()


def _fit_inputs():
    values = {'measurement:lab1': np.linspace(-2.0, 2.0, 100, dtype=np.float32),
              'vital:hr': np.linspace(0.0, 1.0, 40, dtype=np.float32),
              'measurement:lab2': np.full((25,), 0.5, dtype=np.float32),
              'analyte:x': np.zeros((5,), dtype=np.float32)}
    return study.V3FitInputs(
        values_by_item=values, transform_by_item={'vital:hr': 'signed_log'},
        identity_graph_counts={'measurement:lab1': 100, 'vital:hr': 30, 'measurement:lab2': 25,
                               'analyte:x': 5},
        vocabulary=dict(VOCAB), vocabulary_tokens=tuple(TOKENS), token_min_count=20,
        preprocessing_sha256=PREP_SHA, node_feature_layout=tuple(LAYOUT), train_count=10000)


def test_fit_v3_state_writes_the_exact_v3_state_document_once(tmp_path):
    calls = []

    def reader():
        calls.append(1)
        return _fit_inputs()

    root = tmp_path / 'root'
    path = study.fit_v3_state(root, 8, reader=reader)
    assert path == root / 'v3_state' / 'K8.json'
    assert calls == [1]
    inputs = _fit_inputs()
    table = fit_knots(inputs.values_by_item, 8, min_values=20,
                      transform_by_item=inputs.transform_by_item, token_min_count=20)
    universe = fit_universe(inputs.identity_graph_counts, inputs.vocabulary, min_graphs=20,
                            token_min_count=20)
    expected = plugin.build_v3_state(K=8, knot_table=table, universe=universe,
                                     vocabulary_tokens=TOKENS, vocabulary_min_count=20,
                                     preprocessing_sha256=PREP_SHA, node_feature_layout=LAYOUT)
    assert json.loads(path.read_text()) == expected
    assert path.read_bytes() == (json.dumps(expected, indent=2, sort_keys=True) + '\n').encode()
    state = plugin.load_v3_state(path)
    assert state.K == 8 and state.knot_table.items['vital:hr']['transform'] == 'signed_log'
    assert state.knot_table.items['measurement:lab2']['active'] is False
    assert 'analyte:x' not in state.knot_table.items
    with pytest.raises(FileExistsError):
        study.fit_v3_state(root, 8, reader=reader)
    with pytest.raises(ValueError, match='grid'):
        study.fit_v3_state(root, 32, reader=reader)
    assert calls == [1]


def _grid_bindings(plan_, metric_values, **per_stage_overrides):
    """Nine C bindings; `metric_values[(k, seed)]` is selected_dev.metric_value."""
    bindings = {}
    for stage in plan_.stages[:9]:
        binding = _binding(stage, root=plan_.output_root)
        binding['selected_dev']['metric_value'] = metric_values[(stage.k, stage.seed)]
        binding['method_config']['knot_table_sha256'] = hashlib.sha256(
            f'knots-K{stage.k}'.encode()).hexdigest()
        binding['method_config']['v3_state_sha256'] = hashlib.sha256(
            f'state-K{stage.k}'.encode()).hexdigest()
        binding.update(per_stage_overrides.get(stage.name, {}))
        bindings[stage.name] = binding
    return bindings


METRICS_CLEAR = {(4, 1234): 0.401, (4, 2025): 0.398, (4, 7): 0.405,
                 (8, 1234): 0.411, (8, 2025): 0.409, (8, 7): 0.415,
                 (16, 1234): 0.410, (16, 2025): 0.412, (16, 7): 0.408}
METRICS_TIE = {(4, 1234): 0.401, (4, 2025): 0.398, (4, 7): 0.405,
               (8, 1234): 0.41, (8, 2025): 0.42, (8, 7): 0.43,
               (16, 1234): 0.40, (16, 2025): 0.42, (16, 7): 0.44}


def test_select_k_statistic_is_the_unweighted_seed_mean_and_winner_the_argmax(tmp_path):
    plan_ = study.plan(_config(tmp_path))
    selection = study.select_k(_grid_bindings(plan_, METRICS_CLEAR))
    assert selection.k_grid == (4, 8, 16) and selection.seeds == (1234, 2025, 7)
    assert selection.arm == 'C' and selection.rule == study.K_SELECTION_RULE
    assert selection.statistic == METRICS_CLEAR
    assert selection.seed_means == {4: pytest.approx(0.4013333333), 8: pytest.approx(0.4116666667),
                                    16: pytest.approx(0.41)}
    assert selection.k_selected == 8
    assert selection.tie_rule_applied is False
    assert selection.knot_table_sha256_by_k == {
        k: hashlib.sha256(f'knots-K{k}'.encode()).hexdigest() for k in (4, 8, 16)}
    assert selection.v3_state_sha256_by_k == {
        k: hashlib.sha256(f'state-K{k}'.encode()).hexdigest() for k in (4, 8, 16)}
    assert selection.dev_sample_ids_sha256 == 'f' * 64


def test_select_k_tie_at_six_decimals_goes_to_the_smaller_k(tmp_path):
    plan_ = study.plan(_config(tmp_path))
    selection = study.select_k(_grid_bindings(plan_, METRICS_TIE))
    assert set(selection.seed_means) == {4, 8, 16}
    assert round(selection.seed_means[8], 6) == round(selection.seed_means[16], 6) == 0.42
    assert selection.k_selected == 8
    assert selection.tie_rule_applied is True
    # A 7th-decimal difference is still a tie for the runner's 6-decimal metric.
    metrics = dict(METRICS_TIE)
    metrics[(16, 7)] = 0.4400004
    selection = study.select_k(_grid_bindings(plan_, metrics))
    assert (selection.k_selected, selection.tie_rule_applied) == (8, True)
    metrics[(16, 7)] = 0.440003
    selection = study.select_k(_grid_bindings(plan_, metrics))
    assert (selection.k_selected, selection.tie_rule_applied) == (16, False)


def test_select_k_refuses_an_incomplete_grid_arm_drift_and_inconsistent_state_hashes(tmp_path):
    plan_ = study.plan(_config(tmp_path))
    bindings = _grid_bindings(plan_, METRICS_CLEAR)
    del bindings['C_K8_seed7']
    with pytest.raises(ValueError, match='C_K8_seed7'):
        study.select_k(bindings)
    bindings = _grid_bindings(plan_, METRICS_CLEAR)
    bindings['C_K4_seed1234']['method_config']['arm'] = 'A'
    bindings['C_K4_seed1234']['method_config']['effective_settings']['arm'] = 'A'
    with pytest.raises(ValueError, match='arm'):
        study.select_k(bindings)
    bindings = _grid_bindings(plan_, METRICS_CLEAR)
    bindings['C_K4_seed1234']['method_config']['knot_table_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='knot_table_sha256'):
        study.select_k(bindings)
    bindings = _grid_bindings(plan_, METRICS_CLEAR)
    bindings['C_K16_seed7']['split_sample_ids_sha256']['dev'] = '0' * 64
    bindings['C_K16_seed7']['selected_dev']['sample_ids_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='dev'):
        study.select_k(bindings)
    bindings = _grid_bindings(plan_, METRICS_CLEAR)
    bindings['C_K16_seed7']['final_eval'] = 'validation'
    with pytest.raises(ValueError, match='final_eval'):
        study.select_k(bindings)
    bindings = _grid_bindings(plan_, METRICS_CLEAR)
    bindings['extra_seed'] = dict(bindings['C_K8_seed7'])
    with pytest.raises(ValueError, match='extra_seed'):
        study.select_k(bindings)


def _completed_grid(tmp_path, metrics=METRICS_CLEAR):
    plan_ = study.plan(_config(tmp_path))
    bindings = _grid_bindings(plan_, metrics)
    stage_dirs = {}
    for stage in plan_.stages[:9]:
        stage_dir = Path(stage.output)
        stage_dir.mkdir(parents=True)
        (stage_dir / 'binding.json').write_text(
            json.dumps(bindings[stage.name], indent=2, sort_keys=True) + '\n')
        (stage_dir / 'best.pt').write_bytes(f'checkpoint {stage.name}'.encode())
        stage_dirs[stage.name] = stage_dir
    return plan_, bindings, stage_dirs


def test_write_k_selection_binds_checkpoint_and_binding_hashes_and_the_rule(tmp_path):
    plan_, bindings, stage_dirs = _completed_grid(tmp_path)
    selection = study.select_k(bindings)
    path = Path(plan_.k_selection_path)
    record = study.write_k_selection(selection, stage_dirs, path)
    assert path.is_file()
    assert json.loads(path.read_text()) == record
    assert study.k_selection_sha256(record) == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record['version'] == study.K_SELECTION_VERSION
    assert record['k_grid'] == [4, 8, 16] and record['seeds'] == [1234, 2025, 7]
    assert record['arm'] == 'C' and record['rule'] == study.K_SELECTION_RULE
    assert record['statistic'] == 'unweighted mean over seeds of selected_dev.metric_value'
    assert record['tie_rule'] == 'equal 6-decimal seed-mean -> smaller K'
    assert record['k_selected'] == 8 and record['tie_rule_applied'] is False
    assert record['seed_means'] == {'4': pytest.approx(0.4013333333),
                                    '8': pytest.approx(0.4116666667), '16': pytest.approx(0.41)}
    assert record['seed_means_rounded'] == {'4': 0.401333, '8': 0.411667, '16': 0.41}
    assert record['dev_sample_ids_sha256'] == 'f' * 64
    assert record['knot_table_sha256'] == {
        str(k): hashlib.sha256(f'knots-K{k}'.encode()).hexdigest() for k in (4, 8, 16)}
    assert record['v3_state_sha256'] == {
        str(k): hashlib.sha256(f'state-K{k}'.encode()).hexdigest() for k in (4, 8, 16)}
    assert record['selected_knot_table_sha256'] == record['knot_table_sha256']['8']
    assert record['selected_v3_state_sha256'] == record['v3_state_sha256']['8']
    stages = record['stages']
    assert [s['stage'] for s in stages] == [stage.name for stage in plan_.stages[:9]]
    for entry, stage in zip(stages, plan_.stages[:9]):
        stage_dir = stage_dirs[stage.name]
        assert entry == {
            'stage': stage.name, 'k': stage.k, 'seed': stage.seed, 'output': str(stage_dir),
            'checkpoint_sha256': hashlib.sha256((stage_dir / 'best.pt').read_bytes()).hexdigest(),
            'binding_sha256': hashlib.sha256((stage_dir / 'binding.json').read_bytes()).hexdigest(),
            'metric': 'macro_f1', 'metric_value': METRICS_CLEAR[(stage.k, stage.seed)]}
    assert record['control_binding_sha256'] == [
        s['binding_sha256'] for s in stages if s['k'] == 8]
    assert record['k_selection_completed_before_screen'] is True
    assert record['screen_record_sha256'] is None
    assert 'k_selection_sha256' not in record
    with pytest.raises(FileExistsError):
        study.write_k_selection(selection, stage_dirs, path)


def test_write_k_selection_refuses_stage_dirs_that_do_not_match_the_selection(tmp_path):
    plan_, bindings, stage_dirs = _completed_grid(tmp_path)
    selection = study.select_k(bindings)
    missing = dict(stage_dirs)
    del missing['C_K4_seed7']
    with pytest.raises(ValueError, match='C_K4_seed7'):
        study.write_k_selection(selection, missing, Path(plan_.k_selection_path))
    drifted = dict(stage_dirs)
    (drifted['C_K8_seed1234'] / 'binding.json').write_text(json.dumps(
        dict(bindings['C_K8_seed1234'], seed=1234, patience=41), indent=2, sort_keys=True) + '\n')
    with pytest.raises(ValueError, match='binding'):
        study.write_k_selection(selection, drifted, Path(plan_.k_selection_path))
    assert not Path(plan_.k_selection_path).exists()


def test_k_selection_replay_passes_and_detects_every_drift(tmp_path):
    plan_, bindings, stage_dirs = _completed_grid(tmp_path, METRICS_TIE)
    selection = study.select_k(bindings)
    path = Path(plan_.k_selection_path)
    record = study.write_k_selection(selection, stage_dirs, path)
    assert {'k_selected', 'tie_rule_applied', 'stages', 'rule', 'k_grid'} <= set(record)
    assert record['k_selected'] == 8 and record['tie_rule_applied'] is True
    assert study.assert_k_selection_replay(record, stage_dirs) is None
    assert json.loads(path.read_text()) == record  # replay is pure

    tampered = json.loads(json.dumps(record))
    tampered['k_selected'] = 16
    with pytest.raises(ValueError, match='winner'):
        study.assert_k_selection_replay(tampered, stage_dirs)
    tampered = json.loads(json.dumps(record))
    tampered['stages'][4]['metric_value'] = 0.99
    with pytest.raises(ValueError, match='metric_value'):
        study.assert_k_selection_replay(tampered, stage_dirs)
    tampered = json.loads(json.dumps(record))
    tampered['tie_rule_applied'] = False
    with pytest.raises(ValueError, match='tie'):
        study.assert_k_selection_replay(tampered, stage_dirs)
    tampered = json.loads(json.dumps(record))
    tampered['rule'] = 'argmax of screen macro-F1'
    with pytest.raises(ValueError, match='rule'):
        study.assert_k_selection_replay(tampered, stage_dirs)
    tampered = json.loads(json.dumps(record))
    tampered['k_grid'] = [4, 8, 16, 32]
    with pytest.raises(ValueError, match='grid'):
        study.assert_k_selection_replay(tampered, stage_dirs)

    (stage_dirs['C_K16_seed7'] / 'best.pt').write_bytes(b'another checkpoint')
    with pytest.raises(ValueError, match='checkpoint_sha256'):
        study.assert_k_selection_replay(record, stage_dirs)
    (stage_dirs['C_K16_seed7'] / 'best.pt').write_bytes(b'checkpoint C_K16_seed7')
    assert study.assert_k_selection_replay(record, stage_dirs) is None
    text = (stage_dirs['C_K8_seed7'] / 'binding.json').read_text()
    (stage_dirs['C_K8_seed7'] / 'binding.json').write_text(text.replace('0.43', '0.44'))
    with pytest.raises(ValueError, match='binding_sha256'):
        study.assert_k_selection_replay(record, stage_dirs)


# ------------------------------------------- step 3: bootstrap and screen result

import torch  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

from comparison.standardized.clinical_graph_v2 import cei_v2_study as v2  # noqa: E402


def _synthetic_rows(n=60, subjects=20, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 10, n)
    subj = np.asarray([f'p{i % subjects}' for i in range(n)])
    return y, subj, rng


def test_weighted_macro_f1_equals_the_v2_implementation_on_a_synthetic_case():
    y, _, rng = _synthetic_rows()
    pred = rng.integers(0, 10, len(y))
    pred[:5] = y[:5]
    weights = rng.integers(0, 4, len(y)).astype(float)
    assert study.weighted_macro_f1(y, pred) == v2.weighted_macro_f1(y, pred, 10)
    assert study.weighted_macro_f1(y, pred, weights) == v2.weighted_macro_f1(y, pred, 10, weights)
    assert study.weighted_macro_f1(y, y) == 1.0
    assert study.weighted_macro_f1(y, pred) != 0.0
    # zero_division=0 with the fixed ten-label universe (v3 §6.2, F20)
    small_y, small_pred = np.array([0, 0, 1]), np.array([0, 0, 0])
    assert study.weighted_macro_f1(small_y, small_pred) == pytest.approx(0.8 / 10)
    from sklearn.metrics import f1_score
    assert study.weighted_macro_f1(y, pred) == pytest.approx(f1_score(
        y, pred, labels=np.arange(10), average='macro', zero_division=0))


def _three_arm_fixture():
    y, subj, rng = _synthetic_rows(n=80, subjects=25, seed=3)
    probas = {(mode, seed): rng.random((len(y), 10)) for mode in v2.MODES for seed in v2.SEEDS}
    v2_arms = {f'{mode}_seed{seed}': proba for (mode, seed), proba in probas.items()}
    v3_arms = {mode: {seed: (y, probas[(mode, seed)].argmax(1), subj) for seed in v2.SEEDS}
               for mode in v2.MODES}
    return y, subj, v2_arms, v3_arms


def test_paired_bootstrap_reproduces_the_v2_bootstrap_numerically():
    y, subj, v2_arms, v3_arms = _three_arm_fixture()
    point, comparisons = v2.paired_bootstrap(v2_arms, y, subj, num_classes=10)
    contrasts = [('product', 'additive'), ('product', 'off'), ('additive', 'off')]
    deltas = study.paired_bootstrap(v3_arms, contrasts)
    assert set(deltas) == set(contrasts)
    for (first, second), key in zip(contrasts, v2.COMPARISONS):
        values = deltas[(first, second)]
        assert isinstance(values, np.ndarray) and values.shape == (1000,)
        assert values.dtype == np.float64
        assert [float(np.quantile(values, 0.025, method='linear')),
                float(np.quantile(values, 0.975, method='linear'))] == comparisons[key]['interval_95']
        assert np.std(values) > 0
    scores = {mode: {seed: study.weighted_macro_f1(y, pred) for seed, (_, pred, _) in arm.items()}
              for mode, arm in v3_arms.items()}
    for mode, seed in point.keys() and [(m, s) for m in v2.MODES for s in v2.SEEDS]:
        assert scores[mode][seed] == point[f'{mode}_seed{seed}']
    again = study.paired_bootstrap(v3_arms, contrasts)
    assert all(np.array_equal(again[c], deltas[c]) for c in contrasts)
    other = study.paired_bootstrap(v3_arms, contrasts, seed=2027)
    assert not np.array_equal(other[contrasts[0]], deltas[contrasts[0]])
    short = study.paired_bootstrap(v3_arms, contrasts[:1], resamples=10)
    assert short[contrasts[0]].shape == (10,)
    assert np.array_equal(short[contrasts[0]], deltas[contrasts[0]][:10])


def test_paired_bootstrap_refuses_unpaired_rows_and_unknown_contrasts():
    _, _, _, v3_arms = _three_arm_fixture()
    with pytest.raises(ValueError, match='contrast'):
        study.paired_bootstrap(v3_arms, [('product', 'missing')])
    y, pred, subj = v3_arms['product'][1234]
    broken = json.loads(json.dumps({k: {} for k in v3_arms}))
    broken = {mode: dict(arm) for mode, arm in v3_arms.items()}
    broken['off'][7] = (np.roll(y, 1), pred, subj)
    with pytest.raises(ValueError, match='paired'):
        study.paired_bootstrap(broken, [('product', 'off')])
    broken = {mode: dict(arm) for mode, arm in v3_arms.items()}
    broken['off'][7] = (y, pred, np.roll(subj, 1))
    with pytest.raises(ValueError, match='paired'):
        study.paired_bootstrap(broken, [('product', 'off')])
    broken = {mode: dict(arm) for mode, arm in v3_arms.items()}
    del broken['off'][7]
    with pytest.raises(ValueError, match='seed'):
        study.paired_bootstrap(broken, [('product', 'off')])


def _decision_inputs(c=(0.42, 0.43, 0.44), a=(0.40, 0.41, 0.42), lower=0.005):
    rng = np.random.default_rng(1)
    deltas = rng.normal(0.02, 0.005, 1000)
    deltas = deltas - np.quantile(deltas, 0.025, method='linear') + lower
    scores = {'C': dict(zip((1234, 2025, 7), c)), 'A': dict(zip((1234, 2025, 7), a))}
    return {('C', 'A'): deltas}, scores


def test_decide_v3_applies_the_three_condition_rule_with_the_linear_quantile():
    deltas, scores = _decision_inputs()
    decision = study.decide_v3(deltas, scores)
    values = deltas[('C', 'A')]
    assert {'contrast', 'checks', 'per_seed', 'interval_95'} <= set(decision)
    assert decision['contrast'] == ['C', 'A']
    assert decision['per_seed'] == {'1234': {'C': 0.42, 'A': 0.40}, '2025': {'C': 0.43, 'A': 0.41},
                                    '7': {'C': 0.44, 'A': 0.42}}
    assert decision['seed_means'] == {'C': pytest.approx(0.43), 'A': pytest.approx(0.41)}
    assert decision['point_delta'] == pytest.approx(0.02)
    assert decision['interval_95'] == [float(np.quantile(values, 0.025, method='linear')),
                                       float(np.quantile(values, 0.975, method='linear'))]
    assert decision['interval_95'][0] == pytest.approx(0.005)
    assert decision['quantile_method'] == 'linear'
    assert decision['metric'] == 'weighted_macro_f1'
    assert decision['resamples'] == 1000 and decision['bootstrap_seed'] == 2026
    assert decision['checks'] == {'c_beats_a_each_seed': True, 'c_mean_above_a': True,
                                  'c_minus_a_lower_bound_above_zero': True}
    assert decision['v3_beats_control'] is True
    assert decision['statement'] == 'v3 beats the v2 additive control on this screen'
    assert decision['test_evaluated'] is False and decision['validation_evaluated'] is False


@pytest.mark.parametrize('kwargs, failing', [
    (dict(c=(0.42, 0.41, 0.44)), 'c_beats_a_each_seed'),          # tie on seed 2025 fails
    (dict(c=(0.42, 0.43, 0.44), a=(0.40, 0.41, 0.50)), 'c_beats_a_each_seed'),
    (dict(lower=0.0), 'c_minus_a_lower_bound_above_zero'),        # bound exactly zero fails
    (dict(lower=-0.001), 'c_minus_a_lower_bound_above_zero'),
])
def test_decide_v3_fails_closed_on_ties_and_zero_bounds(kwargs, failing):
    deltas, scores = _decision_inputs(**kwargs)
    decision = study.decide_v3(deltas, scores)
    assert 'checks' in decision
    assert decision['checks'][failing] is False
    assert decision['v3_beats_control'] is False
    assert decision['statement'] == 'benefit not demonstrated on this screen'


def test_decide_v3_mean_condition_can_fail_independently():
    deltas, scores = _decision_inputs(c=(0.42, 0.43, 0.44), a=(0.40, 0.41, 0.42))
    scores['A'][7] = 0.42
    decision = study.decide_v3(deltas, scores)
    assert 'checks' in decision
    assert decision['checks']['c_beats_a_each_seed'] is True
    # Force the mean check off via a control whose mean equals the treatment mean.
    deltas, scores = _decision_inputs(c=(0.42, 0.43, 0.44), a=(0.44, 0.43, 0.42))
    decision = study.decide_v3(deltas, scores)
    assert decision['checks']['c_mean_above_a'] is False
    assert decision['v3_beats_control'] is False


def test_decide_v3_refuses_wrong_resample_count_or_missing_seed():
    deltas, scores = _decision_inputs()
    with pytest.raises(ValueError, match='1000'):
        study.decide_v3({('C', 'A'): deltas[('C', 'A')][:999]}, scores)
    del scores['A'][7]
    with pytest.raises(ValueError, match='seed'):
        study.decide_v3(deltas, scores)


def test_absence_share_is_the_patient_mean_class_share_with_zero_denominator_rule():
    parts = {
        'node_contributions': torch.tensor([[1.0, -1.0], [0.0, 0.0], [2.0, 0.0]]),
        'edge_contributions': torch.tensor([[0.5, 0.5]]),
        'pair_contributions': torch.tensor([[0.0, 1.0]]),
        'pairs': torch.tensor([[0], [1]]),
        'absence_contributions': torch.tensor([[1.0, 0.5], [3.0, 0.0]]),
        'absence_items': torch.tensor([[0, 1], [0, 1]]),   # (graph, slot) per absent item
    }
    batch_index = torch.tensor([0, 0, 1])
    edge_index = torch.tensor([[0], [1]])
    share = study.absence_share(parts, batch_index=batch_index, edge_index=edge_index,
                                graph_count=3)
    # graph 0: |abs| per class = (1.0, 0.5); totals (1+0.5+0+1.0, 1+0.5+1+0.5) = (2.5, 3.0)
    # graph 1: node 2 only -> absence (3.0, 0.0), totals (5.0, 0.0) -> classes (0.6, 0)
    # graph 2: nothing -> 0
    assert share.shape == (3,)
    assert share.tolist() == pytest.approx([(1.0 / 2.5 + 0.5 / 3.0) / 2, 0.3, 0.0])


# ---- score_screen with a synthetic adapter stub (no checkpoint, no data file) ----


class _StubAdapter:
    """Adapter protocol used by score_screen: continuous_inputs + forward_continuous."""

    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.calls = 0
        self.training = True

    def eval(self):
        self.training = False
        return self

    def continuous_inputs(self, batch):
        return batch.x

    def forward_continuous(self, features, edge_index, metadata, *, return_parts=False):
        self.calls += 1
        graphs = int(metadata.num_graphs)
        batch_index = metadata.batch
        logits = torch.zeros((graphs, self.num_classes), dtype=torch.float32)
        logits.index_add_(0, batch_index, features[:, :self.num_classes])
        logits = logits * 40.0   # large magnitudes: float32 softmax underflows (E7)
        node = features[:, :self.num_classes]
        absence = torch.ones((graphs, self.num_classes)) * 0.25
        parts = {'logits': logits, 'node_contributions': node,
                 'edge_contributions': torch.zeros((edge_index.size(1), self.num_classes)),
                 'pair_contributions': torch.zeros((0, self.num_classes)),
                 'pairs': torch.zeros((2, 0), dtype=torch.long),
                 'absence_contributions': absence,
                 'absence_items': torch.stack([torch.arange(graphs), torch.zeros(graphs, dtype=torch.long)])}
        return parts if return_parts else logits


def _screen_rows(n=12, seed=5):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        x = torch.tensor(rng.normal(size=(3, 10)), dtype=torch.float32)
        data = Data(x=x, edge_index=torch.tensor([[0, 1], [1, 2]]))
        data.y = torch.tensor([int(rng.integers(0, 10))])
        data.subject = f'ps-{i // 2}'
        data.sample_id = f'sc-{i:05d}'
        rows.append(data)
    return rows


def _screen_record(row_count=12, screen_hash='s' * 64):
    return {'fold': 'screen', 'selector_version': 'cei_v3_screen_selector_v1',
            'screen_seed': 20260929, 'screen_limit': row_count, 'row_count': row_count,
            'screen_sample_ids_sha256': screen_hash, 'targets_sha256': 't' * 64,
            'artifact_sha256': 'a' * 64, 'preprocessing_sha256': 'b' * 64,
            'parent_train_sample_ids_sha256': 'e' * 64, 'parent_dev_sample_ids_sha256': 'f' * 64,
            'serialization_version': 'json_compact_utf8_v1'}


def _frozen_study(tmp_path, metrics=METRICS_CLEAR):
    """Nine C stages on disk + k_selection.json + A_seed1234 stage with study binding."""
    plan_, bindings, stage_dirs = _completed_grid(tmp_path, metrics)
    selection = study.select_k(bindings)
    record = study.write_k_selection(selection, stage_dirs, Path(plan_.k_selection_path))
    for stage in plan_.stages[:9]:
        if stage.k == record['k_selected']:
            (stage_dirs[stage.name] / 'result.json').write_text(json.dumps(
                _result(bindings[stage.name]), indent=2, sort_keys=True) + '\n')
            study.write_study_binding(stage_dirs[stage.name], stage, record)
    frozen = study.plan(_config(tmp_path, k_selection=record))
    a_stage = _stage(frozen, 'A_seed1234')
    a_dir = Path(a_stage.output)
    a_dir.mkdir(parents=True)
    a_binding = _binding(a_stage, root=frozen.output_root)
    (a_dir / 'binding.json').write_text(json.dumps(a_binding, indent=2, sort_keys=True) + '\n')
    (a_dir / 'result.json').write_text(json.dumps(_result(a_binding), indent=2, sort_keys=True) + '\n')
    (a_dir / 'best.pt').write_bytes(b'checkpoint A_seed1234')
    study.write_study_binding(a_dir, a_stage, record)
    return frozen, record, stage_dirs, a_dir


def test_score_screen_writes_hashed_raw_logits_and_proba_and_a_bound_result(tmp_path):
    frozen, record, stage_dirs, a_dir = _frozen_study(tmp_path)
    c_dir = stage_dirs['C_K8_seed7']
    rows = _screen_rows()
    stub = _StubAdapter()
    encoder = study.ScreenEncoder(ids=tuple(r.sample_id for r in rows), fold='screen',
                                  rows=lambda: iter(rows))
    checkpoint = study.Checkpoint(stage_dir=str(c_dir), k_selection_path=frozen.k_selection_path)
    result = study.score_screen(checkpoint, _screen_record(), encoder,
                                model_factory=lambda binding, path: stub, batch_size=5)
    assert isinstance(result, study.ScreenResult)
    assert stub.calls == 3 and stub.training is False
    assert (result.arm, result.seed, result.k) == ('C', 7, 8)
    assert result.row_count == 12
    out = c_dir / 'screen'
    assert Path(result.logits_path) == out / 'logits.npz'
    assert Path(result.proba_path) == out / 'proba.npz'
    with np.load(out / 'logits.npz', allow_pickle=False) as saved:
        assert set(saved.files) == {'logits', 'y', 'subjects', 'sample_ids'}
        logits = saved['logits']
        assert logits.dtype == np.float32 and logits.shape == (12, 10)
        assert np.array_equal(saved['y'], np.asarray([int(r.y) for r in rows]))
        assert list(saved['subjects'].astype(str)) == [r.subject for r in rows]
        assert list(saved['sample_ids'].astype(str)) == [r.sample_id for r in rows]
    with np.load(out / 'proba.npz', allow_pickle=False) as saved:
        assert set(saved.files) == {'proba', 'y', 'subjects', 'sample_ids'}
        proba = saved['proba']
        assert proba.dtype == np.float32 and proba.shape == (12, 10)
    expected_logits = np.concatenate([
        stub.forward_continuous(r.x, r.edge_index, Data(batch=torch.zeros(3, dtype=torch.long),
                                                         num_graphs=1)).numpy() for r in rows])
    assert np.array_equal(logits, expected_logits.astype(np.float32))
    assert np.array_equal(proba, torch.softmax(torch.from_numpy(logits), dim=1).numpy())
    assert (proba == 0.0).any()   # raw logits are needed because proba underflows (E7)
    assert result.logits_sha256 == hashlib.sha256(np.ascontiguousarray(logits).tobytes()).hexdigest()
    assert result.proba_sha256 == hashlib.sha256(np.ascontiguousarray(proba).tobytes()).hexdigest()
    assert result.logits_sha256 != result.proba_sha256
    assert result.macro_f1 == study.weighted_macro_f1([int(r.y) for r in rows], logits.argmax(1))
    assert result.checkpoint_sha256 == hashlib.sha256((c_dir / 'best.pt').read_bytes()).hexdigest()
    assert result.binding_sha256 == hashlib.sha256((c_dir / 'binding.json').read_bytes()).hexdigest()
    assert result.k_selection_sha256 == study.k_selection_sha256(record)
    assert result.screen_record_sha256 == study.screen_record_sha256(_screen_record())
    # Patient-mean over ALL screen graphs of the per-graph absence share (v3 §6.2).
    single = Data(batch=torch.zeros(3, dtype=torch.long), num_graphs=1)
    per_graph = [study.absence_share(
        stub.forward_continuous(r.x, r.edge_index, single, return_parts=True),
        batch_index=single.batch, edge_index=r.edge_index, graph_count=1)[0] for r in rows]
    assert result.absence_share_mean == pytest.approx(float(np.mean(per_graph)))
    assert len(set(np.round(per_graph, 6))) > 1   # the mean is not a single-graph value
    assert result.validation_evaluated is False and result.test_evaluated is False
    written = json.loads((out / 'screen_result.json').read_text())
    assert written == json.loads(json.dumps(result.__dict__))
    assert not (c_dir / 'validation.npz').exists() and not (out / 'validation.npz').exists()
    with pytest.raises(FileExistsError):
        study.score_screen(checkpoint, _screen_record(), encoder,
                           model_factory=lambda binding, path: stub)


def test_score_screen_scores_a_and_refuses_freeze_and_k_violations(tmp_path):
    frozen, record, stage_dirs, a_dir = _frozen_study(tmp_path)
    rows = _screen_rows()
    encoder = study.ScreenEncoder(ids=tuple(r.sample_id for r in rows), fold='screen',
                                  rows=lambda: iter(rows))
    factory = lambda binding, path: _StubAdapter()  # noqa: E731
    a_result = study.score_screen(study.Checkpoint(str(a_dir), frozen.k_selection_path),
                                  _screen_record(), encoder, model_factory=factory)
    assert (a_result.arm, a_result.k, a_result.k_selection_sha256) == (
        'A', 8, study.k_selection_sha256(record))

    # Losing-K C checkpoints (K != K*) are never scored on the screen (v3 §11.2 item 5).
    losing = stage_dirs['C_K4_seed1234']
    (losing / 'result.json').write_text(json.dumps(
        _result(json.loads((losing / 'binding.json').read_text())), indent=2, sort_keys=True) + '\n')
    with pytest.raises(ValueError, match='K'):
        study.score_screen(study.Checkpoint(str(losing), frozen.k_selection_path),
                           _screen_record(), encoder, model_factory=factory)
    # A checkpoint whose binding lacks the K-freeze hash is refused.
    winner = stage_dirs['C_K8_seed1234']
    (winner / 'study_binding.json').unlink()
    with pytest.raises(ValueError, match='k_selection_sha256'):
        study.score_screen(study.Checkpoint(str(winner), frozen.k_selection_path),
                           _screen_record(), encoder, model_factory=factory)
    # No freeze record -> no screen tensor is read (v3 §11.2 item 8).
    reads = []
    guarded = study.ScreenEncoder(ids=encoder.ids, fold='screen',
                                  rows=lambda: reads.append(1) or iter(rows))
    with pytest.raises(ValueError, match='freeze'):
        study.score_screen(study.Checkpoint(str(a_dir), str(tmp_path / 'absent.json')),
                           _screen_record(), guarded, model_factory=factory)
    assert reads == []
    # A freeze record whose hash differs from the checkpoint's binding is refused.
    other = json.loads(json.dumps(record))
    other['screen_record_sha256'] = 'x' * 64
    other_path = tmp_path / 'other_k_selection.json'
    other_path.write_text(json.dumps(other, indent=2, sort_keys=True) + '\n')
    with pytest.raises(ValueError, match='k_selection_sha256'):
        study.score_screen(study.Checkpoint(str(a_dir), str(other_path)), _screen_record(),
                           guarded, model_factory=factory)
    assert reads == []


@pytest.mark.parametrize('kind', ['fold', 'row_count', 'ids', 'preprocessing', 'screen_reuse',
                                  'validation_npz', 'record_fold'])
def test_score_screen_refuses_fold_and_binding_violations(tmp_path, kind):
    frozen, record, stage_dirs, a_dir = _frozen_study(tmp_path)
    rows = _screen_rows()
    reads = []
    ids = tuple(r.sample_id for r in rows)
    fold, screen_record = 'screen', _screen_record()
    if kind == 'fold':
        fold = 'dev'
    elif kind == 'row_count':
        screen_record['row_count'] = 11
    elif kind == 'ids':
        ids = ids[:-1] + ('sc-99999',)
    elif kind == 'preprocessing':
        screen_record['preprocessing_sha256'] = '0' * 64
    elif kind == 'screen_reuse':
        screen_record['screen_sample_ids_sha256'] = 'f' * 64   # equals the bindings' dev hash
    elif kind == 'validation_npz':
        (a_dir / 'validation.npz').write_bytes(b'x')
    elif kind == 'record_fold':
        screen_record['fold'] = 'validation'
    encoder = study.ScreenEncoder(ids=ids, fold=fold, rows=lambda: reads.append(1) or iter(rows))
    with pytest.raises(ValueError):
        study.score_screen(study.Checkpoint(str(a_dir), frozen.k_selection_path), screen_record,
                           encoder, model_factory=lambda binding, path: _StubAdapter())
    if kind != 'ids':
        assert reads == []
    assert not (a_dir / 'screen').exists()


def test_screen_record_sha256_is_canonical_and_tamper_sensitive():
    record = _screen_record()
    digest = study.screen_record_sha256(record)
    assert len(digest) == 64 and digest == study.screen_record_sha256(dict(reversed(list(record.items()))))
    tampered = dict(record, row_count=4999)
    assert study.screen_record_sha256(tampered) != digest
