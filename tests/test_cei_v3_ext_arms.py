"""Synthetic tests for the CEI-GNN v3 extension arms E2w, E2d and E6a (unit X14).

Extensions spec §2 (size arms), §5.2 / §5.5 item 3 (bidirectional arm), §6 (arm table) and
§9 X14 (binding guards). Fixtures are the U3/U3x synthetic ones plus small in-memory graph
documents; nothing here opens a data file, run directory, checkpoint or test-fold row.
"""
from __future__ import annotations

from argparse import Namespace

import pytest
import torch

from comparison.standardized.clinical_graph_v2.cei_v3_ext import arm_guards
from comparison.standardized.clinical_graph_v2.methods import METHOD_REGISTRY
from comparison.standardized.clinical_graph_v2.methods.base import parameter_count
from comparison.standardized.clinical_graph_v2.tensorize import (ALL_RELATIONS, PAYLOAD_WIDTH,
                                                                 REVERSE_RELATIONS)
from tests.test_cei_gnn_v3_core import (CLASSES, EDGE_DIM, HIDDEN, K, NODE_DIM, NUM_TOKENS,
                                        TOKEN_DIM, TRIPLES, _write_state)
from tests.test_cei_gnn_v3_hooks import (BLOCK_DELTA_AT_128, CONTROL_HIDDEN, E6A_WIDENED,
                                         _count, _network)

RELATIONS_FORWARD = len(ALL_RELATIONS)                       # 15
RELATIONS_BIDIRECTIONAL = RELATIONS_FORWARD + len(REVERSE_RELATIONS)   # 24
UNIVERSE_SIZE = 3   # the U3 fixture universe: lab1, lab2, hr
FIXTURE_DIMENSIONS = dict(num_tokens=NUM_TOKENS, node_dim=NODE_DIM, edge_dim=EDGE_DIM,
                          num_classes=CLASSES, hidden=HIDDEN, token_dim=TOKEN_DIM,
                          num_triples=TRIPLES, num_relations=RELATIONS_FORWARD, pair_rank=16,
                          num_node_types=8)
# Recorded v2 full-run binding (spec §1) and the analytic arm table of §2.3, §5.2, §5.3.
SPEC_V2_TOTAL = 92_300
SPEC_DELTAS = {'E2w': 177_792, 'E2d': 16_768, 'E6a': 4_224, 'E6b': 2_228}
SPEC_TOTALS = {'E2w': 270_092, 'E2d': 109_068, 'E6a': 96_524, 'E6b': 94_528}


def _adapter(tmp_path, *, hidden=CONTROL_HIDDEN, layers=1, edge_direction=None, seed=1234,
             num_triples=TRIPLES, edge_dim=EDGE_DIM, **options):
    """`cei_gnn_v3` adapter on the U3 state file with overridable runner dimensions."""
    state_path = _write_state(tmp_path)
    args = Namespace(method_options={'arm': 'C', 'v3_state': str(state_path), 'k': K, **options},
                     seed=seed)
    if edge_direction is not None:
        args.edge_direction = edge_direction
    torch.manual_seed(seed)
    return METHOD_REGISTRY['cei_gnn_v3'](
        num_tokens=NUM_TOKENS, node_dim=NODE_DIM, edge_dim=edge_dim, num_classes=CLASSES,
        hidden=hidden, layers=layers, dropout=0.0, token_dim=TOKEN_DIM, num_triples=num_triples,
        args=args)


def _fixture_dimensions(**overrides):
    dims = dict(FIXTURE_DIMENSIONS)
    dims.update(overrides)
    return dims


# ------------------------------------------------------------------ step 1: size arms

def test_analytic_counts_reproduce_the_recorded_v2_total_and_the_spec_arm_table():
    dims = arm_guards.V2_REFERENCE_DIMENSIONS
    assert dims['hidden'] == 128 and dims['num_relations'] == 15 and dims['num_triples'] == 16
    assert arm_guards.V2_REFERENCE_PARAMETER_COUNT == SPEC_V2_TOTAL
    assert arm_guards.analytic_parameter_count(dims) == SPEC_V2_TOTAL
    assert arm_guards.ANALYTIC_V2_SCHEMA_DELTAS == SPEC_DELTAS
    for arm, delta in SPEC_DELTAS.items():
        assert arm_guards.analytic_arm_delta(arm, dims) == delta, arm
        assert SPEC_V2_TOTAL + delta == SPEC_TOTALS[arm], arm
    assert arm_guards.analytic_arm_delta('C', dims) == 0
    # The v3 blocks (PLE projection, absence tables) are added on top of the v2 schema and
    # do not change the deltas of E2d, E6a and E6b; E2w's delta grows with the PLE width.
    assert arm_guards.analytic_parameter_count(dims, K=8, universe_size=50) == (
        SPEC_V2_TOTAL + 9 * 128 + 2 * 50 * 10)
    for arm in ('E2d', 'E6a', 'E6b'):
        assert arm_guards.analytic_arm_delta(arm, dims, K=8, universe_size=50) == SPEC_DELTAS[arm]
    assert arm_guards.analytic_arm_delta('E2w', dims, K=8, universe_size=50) == SPEC_DELTAS['E2w'] + 9 * 128
    with pytest.raises(ValueError, match='arm'):
        arm_guards.analytic_arm_delta('O', dims)


def test_analytic_count_equals_parameter_count_of_the_v3_network_on_fixtures():
    for hidden in (HIDDEN, 16):
        for depth in (1, 2):
            network = _network(hidden=hidden, encoder_depth=depth)
            expected = arm_guards.analytic_parameter_count(
                _fixture_dimensions(hidden=hidden, pair_rank=4), K=K, universe_size=UNIVERSE_SIZE,
                encoder_depth=depth)
            assert _count(network) == expected, (hidden, depth)
    widened = _network(num_relations=RELATIONS_BIDIRECTIONAL, num_triples=2 * (TRIPLES - 1) + 1,
                       edge_dim=EDGE_DIM + len(REVERSE_RELATIONS))
    assert _count(widened) == arm_guards.analytic_parameter_count(
        _fixture_dimensions(pair_rank=4, num_relations=RELATIONS_BIDIRECTIONAL,
                            num_triples=2 * (TRIPLES - 1) + 1,
                            edge_dim=EDGE_DIM + len(REVERSE_RELATIONS)),
        K=K, universe_size=UNIVERSE_SIZE)
    for bad in ({'hidden': 0}, {'num_relations': -1}, {'pair_rank': 1.5}):
        with pytest.raises(ValueError):
            arm_guards.analytic_parameter_count(_fixture_dimensions(**bad), K=K, universe_size=1)
    with pytest.raises(ValueError, match='encoder_depth'):
        arm_guards.analytic_parameter_count(_fixture_dimensions(), K=K, universe_size=1, encoder_depth=0)


def test_e2w_and_e2d_analytic_deltas_equal_adapter_parameter_count_differences(tmp_path):
    control = _adapter(tmp_path)
    control_total = parameter_count(control)
    assert control.run_config()['parameter_count'] == control_total
    dims = _fixture_dimensions(hidden=CONTROL_HIDDEN)
    assert control_total == arm_guards.analytic_parameter_count(dims, K=K, universe_size=UNIVERSE_SIZE)
    wide = _adapter(tmp_path, hidden=256)
    assert parameter_count(wide) - control_total == arm_guards.analytic_arm_delta(
        'E2w', dims, K=K, universe_size=UNIVERSE_SIZE)
    deep = _adapter(tmp_path, layers=2)
    assert parameter_count(deep) - control_total == arm_guards.analytic_arm_delta(
        'E2d', dims, K=K, universe_size=UNIVERSE_SIZE)
    assert parameter_count(deep) - control_total == BLOCK_DELTA_AT_128
    # Inventory diff against C (§1, E17): E2w widens every tensor, E2d adds exactly one block.
    control_inventory = control.run_config()['parameter_inventory']
    wide_inventory, deep_inventory = wide.run_config()['parameter_inventory'], deep.run_config()['parameter_inventory']
    assert set(wide_inventory) == set(control_inventory)
    assert set(deep_inventory) - set(control_inventory) == {
        'encoder_blocks.0.linear.weight', 'encoder_blocks.0.linear.bias',
        'encoder_blocks.0.norm.weight', 'encoder_blocks.0.norm.bias'}
    assert all(deep_inventory[name] == control_inventory[name] for name in control_inventory)
    diff = arm_guards.inventory_diff(control_inventory, wide_inventory)
    assert diff['added'] == [] and diff['removed'] == []
    assert set(diff['widened']) == {name for name in control_inventory
                                    if wide_inventory[name]['shape'] != control_inventory[name]['shape']}
    assert 'token_embedding.weight' in diff['unchanged'] and 'bias' in diff['widened']
    diff = arm_guards.inventory_diff(control_inventory, deep_inventory)
    assert diff['widened'] == [] and set(diff['added']) == set(deep_inventory) - set(control_inventory)
    assert sorted(diff['unchanged']) == sorted(control_inventory)


def test_run_config_records_the_size_arm_definition_fields(tmp_path):
    control = _adapter(tmp_path).run_config()
    wide, deep = _adapter(tmp_path, hidden=256).run_config(), _adapter(tmp_path, layers=2).run_config()
    assert (control['hidden'], control['encoder_depth'], control['common_init_identical_to_c']) == (128, 1, True)
    assert (wide['hidden'], wide['encoder_depth'], wide['common_init_identical_to_c']) == (256, 1, False)
    assert (deep['hidden'], deep['encoder_depth'], deep['common_init_identical_to_c']) == (128, 2, True)
    for config in (control, wide, deep):
        assert config['arm'] == 'C' and config['edge_direction'] == 'forward'
        assert config['comorbid_block'] == 0 and config['pair_mode'] == 'additive'
        assert config['parameter_count'] == config['active_parameter_count'] + config['inactive_parameter_count']
        assert config['architecture']['encoder_depth'] == config['encoder_depth']
        assert config['architecture']['hidden'] == config['hidden']
    assert wide['architecture']['layers'] == 1 and deep['architecture']['layers'] == 2
    assert arm_guards.arm_of_run_config(control) == 'C'
    assert arm_guards.arm_of_run_config(wide) == 'E2w'
    assert arm_guards.arm_of_run_config(deep) == 'E2d'
    assert arm_guards.ARM_DEFINITIONS['E2w'] == {'hidden': 256, 'encoder_depth': 1,
                                                 'edge_direction': 'forward', 'comorbid_block': 0,
                                                 'common_init_identical_to_c': False}
    assert arm_guards.ARM_DEFINITIONS['E2d'] == {'hidden': 128, 'encoder_depth': 2,
                                                 'edge_direction': 'forward', 'comorbid_block': 0,
                                                 'common_init_identical_to_c': True}
    assert set(arm_guards.ARM_DEFINITIONS) == {'C', 'E2w', 'E2d', 'E6a', 'E6b'}
    # A configuration outside the arm table is never mapped to an arm.
    with pytest.raises(ValueError, match='arm'):
        arm_guards.arm_of_run_config(_adapter(tmp_path, hidden=256, layers=2).run_config())
    with pytest.raises(ValueError, match='arm'):
        arm_guards.arm_of_run_config(_adapter(tmp_path, hidden=64).run_config())
    with pytest.raises(ValueError, match='arm'):
        arm_guards.arm_of_run_config({**control, 'arm': 'A'})


def test_v3_adapter_accepts_layers_2_and_the_v2_adapter_still_refuses_it(tmp_path):
    assert _adapter(tmp_path, layers=2).network.encoder_depth == 2
    v2 = METHOD_REGISTRY['cei_gnn_v2']
    common = dict(num_tokens=NUM_TOKENS, node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_classes=CLASSES,
                  hidden=CONTROL_HIDDEN, dropout=0.0, token_dim=TOKEN_DIM, num_triples=TRIPLES,
                  args=Namespace(method_options={'pair_mode': 'additive'}, seed=1234))
    with pytest.raises(ValueError, match='layers=1 only'):
        v2(layers=2, **common)
    assert v2(layers=1, **common).layers_count == 1
    # The v2 adapter has no depth option either: the runner cannot reach E2d through v2.
    with pytest.raises(ValueError, match='unknown method option'):
        v2(layers=1, **{**common, 'args': Namespace(
            method_options={'pair_mode': 'additive', 'encoder_depth': 2}, seed=1234)})
    assert arm_guards.assert_size_arm_support() == {'cei_gnn_v3': True, 'cei_gnn_v2': False}
