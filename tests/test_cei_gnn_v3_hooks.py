"""Synthetic tests for the CEI-GNN v3 core extension hooks (unit U3x).

Extensions spec §2.2 (E6 RNG/init rule), §9 U3x (encoder_depth residual blocks, the
`extra_blocks` protocol, extension parity). Fixtures are imported from the U3 core tests;
nothing here opens a data file, run directory or checkpoint, and X5's comorbid module is
never imported (a synthetic stub block stands in for it).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from comparison.standardized.clinical_graph_v2 import NODE_KINDS
from comparison.standardized.clinical_graph_v2.methods import cei_gnn_v3 as v3
from tests.test_cei_gnn_v3_core import (CLASSES, EDGE_DIM, HIDDEN, K, LAYOUT, NODE_DIM,
                                        NUM_TOKENS, PAIR_RANK, RELATIONS, TOKEN_DIM, TRIPLES,
                                        VOCAB, _graph, _knot_table, _metadata, _randomise_gates,
                                        _run, _share_weights, _universe, _visit_graph)

CONTROL_HIDDEN = 128
BLOCK_DELTA_AT_128 = 128 * 128 + 128 + 2 * 128   # +16,768 per block (spec §2.3)


def _network(arm='C', *, seed=11, dropout=0.0, init_seed=1234, hidden=HIDDEN,
             num_relations=RELATIONS, **extra):
    """`EvidenceNetworkV3` on the U3 fixtures with overridable width / relation count."""
    table, universe = _knot_table(), _universe()
    knots, active, _ = table.tensor()
    knot_row_of_token = torch.full((NUM_TOKENS,), -1, dtype=torch.long)
    for item, row in table.rows().items():
        knot_row_of_token[VOCAB[item]] = row
    torch.manual_seed(seed)
    return v3.EvidenceNetworkV3(
        num_tokens=NUM_TOKENS, node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_classes=CLASSES,
        hidden=hidden, token_dim=TOKEN_DIM, num_triples=TRIPLES, num_relations=num_relations,
        dropout=dropout, pair_rank=PAIR_RANK, num_node_types=len(NODE_KINDS), arm=arm,
        knots=knots, knot_active=active, slot_of_token=universe.slot_of_token(NUM_TOKENS),
        universe_size=len(universe), feature_layout=list(LAYOUT), seed=init_seed,
        knot_row_of_token=knot_row_of_token, **extra)


def _count(network):
    return sum(parameter.numel() for parameter in network.parameters())


def _rebuild(parts, *extra_keys):
    total = (parts['bias'] + parts['node_contributions'].sum(0) + parts['edge_contributions'].sum(0)
             + parts['pair_contributions'].sum(0) + parts['absence_contributions'].sum(0))
    for key in extra_keys:
        total = total + parts[key].sum(0)
    return total


# ------------------------------------------------------------ step 1: encoder depth

BLOCK_NAMES = ('encoder_blocks.0.linear.weight', 'encoder_blocks.0.linear.bias',
               'encoder_blocks.0.norm.weight', 'encoder_blocks.0.norm.bias')


def test_depth_one_registers_no_block_and_keeps_the_u3_parameter_set():
    plain, explicit = _network(), _network(encoder_depth=1)
    names = {name for name, _ in explicit.named_parameters()}
    assert names == {name for name, _ in plain.named_parameters()}
    assert not any(name.startswith('encoder_blocks') for name in names)
    assert explicit.encoder_depth == 1
    assert not any(name.startswith('encoder_blocks') for name, _ in explicit.named_modules())
    assert torch.equal(_run(explicit, _graph())['logits'], _run(plain, _graph())['logits'])
    for depth in (0, 1.5, True):
        with pytest.raises(ValueError, match='encoder_depth'):
            _network(encoder_depth=depth)


def test_depth_two_adds_one_named_residual_block_with_the_analytic_parameter_delta():
    base, deep = _network(), _network(encoder_depth=2)
    shapes = {name: tuple(p.shape) for name, p in deep.named_parameters()}
    base_shapes = {name: tuple(p.shape) for name, p in base.named_parameters()}
    assert set(shapes) == set(base_shapes) | set(BLOCK_NAMES)
    assert shapes['encoder_blocks.0.linear.weight'] == (HIDDEN, HIDDEN)
    assert shapes['encoder_blocks.0.linear.bias'] == (HIDDEN,)
    assert shapes['encoder_blocks.0.norm.weight'] == (HIDDEN,)
    assert shapes['encoder_blocks.0.norm.bias'] == (HIDDEN,)
    assert _count(deep) - _count(base) == HIDDEN * HIDDEN + 3 * HIDDEN
    inventory = deep.parameter_inventory()
    assert all(inventory[name] == (shapes[name], True) for name in BLOCK_NAMES)
    assert deep.encoder_depth == 2
    # Spec arithmetic at the control width: +16,768 per block, +2 blocks for depth 3.
    wide_base = _network(hidden=CONTROL_HIDDEN)
    assert _count(_network(hidden=CONTROL_HIDDEN, encoder_depth=2)) - _count(wide_base) == BLOCK_DELTA_AT_128
    assert _count(_network(hidden=CONTROL_HIDDEN, encoder_depth=3)) - _count(wide_base) == 2 * BLOCK_DELTA_AT_128


def test_block_tensors_come_from_per_tensor_generators_not_the_global_stream():
    torch.manual_seed(1)
    first = _network(seed=1, init_seed=1234, encoder_depth=2)
    torch.manual_seed(999)
    second = _network(seed=999, init_seed=1234, encoder_depth=2)
    other = _network(seed=999, init_seed=2025, encoder_depth=2)
    weight = first.encoder_blocks[0].linear.weight
    assert torch.equal(weight, second.encoder_blocks[0].linear.weight)
    assert not torch.equal(weight, other.encoder_blocks[0].linear.weight)
    bound = 1.0 / HIDDEN ** 0.5   # nn.Linear's default uniform bound at fan_in = hidden
    expected = torch.empty(HIDDEN, HIDDEN).uniform_(
        -bound, bound, generator=v3.tensor_generator(1234, 'encoder_blocks.0.linear.weight'))
    assert torch.equal(weight, expected)
    expected_bias = torch.empty(HIDDEN).uniform_(
        -bound, bound, generator=v3.tensor_generator(1234, 'encoder_blocks.0.linear.bias'))
    assert torch.equal(first.encoder_blocks[0].linear.bias, expected_bias)
    assert torch.equal(first.encoder_blocks[0].norm.weight, torch.ones(HIDDEN))
    assert torch.equal(first.encoder_blocks[0].norm.bias, torch.zeros(HIDDEN))


def test_residual_block_is_h_plus_dropout_gelu_layernorm_linear():
    block = _network(encoder_depth=2).eval().encoder_blocks[0]
    h = torch.randn(5, HIDDEN, generator=torch.Generator().manual_seed(3))
    expected = h + torch.nn.functional.gelu(block.norm(block.linear(h)))
    assert torch.equal(block(h), expected)


def _identity_blocks(network):
    with torch.no_grad():
        for block in network.encoder_blocks:
            block.linear.weight.zero_()
            block.linear.bias.zero_()


def test_depth_two_changes_only_h_and_keeps_exact_reconstruction():
    graph = _graph()
    base = _network().eval()
    _randomise_gates(base)
    with torch.no_grad():
        base.absence_vote.normal_()
        base.absence_gate.normal_()
    deep = _network(encoder_depth=2).eval()
    _share_weights(base, deep)
    reference, parts = _run(base, graph), _run(deep, graph)
    assert set(parts) == set(reference)
    for key in ('node_contributions', 'edge_contributions', 'pair_contributions'):
        assert tuple(parts[key].shape) == tuple(reference[key].shape)
        assert not torch.allclose(parts[key], reference[key]), f'{key} unchanged at depth 2'
    for key in ('bias', 'pairs', 'pair_denominator', 'absence_contributions', 'absence_items',
                'absence_gates', 'absence_denominator'):
        assert torch.equal(parts[key], reference[key]), f'{key} changed at depth 2'
    torch.testing.assert_close(parts['logits'][0], _rebuild(parts), rtol=1e-5, atol=1e-5)
    # Identity blocks (zero linear) reproduce depth 1 bit-for-bit, in eval and training mode.
    _identity_blocks(deep)
    assert torch.equal(_run(deep, graph)['logits'], reference['logits'])
    base.train(), deep.train()
    torch.manual_seed(5)
    expected = _run(base, graph)['logits']
    torch.manual_seed(5)
    assert torch.equal(_run(deep, graph)['logits'], expected)


def _three_training_forwards(network, graph):
    metadata = _metadata(graph)
    features = network.continuous_inputs(metadata)
    torch.manual_seed(947)
    states, outputs = [], []
    for _ in range(3):
        outputs.append(network.forward_continuous(features, metadata.edge_index, metadata,
                                                  graph.visit_membership_index,
                                                  return_parts=True,
                                                  visit_graph=_visit_graph(graph))['logits'])
        states.append(torch.get_rng_state().clone())
    return states, outputs


def test_residual_dropout_draws_from_a_dedicated_generator_and_leaves_the_global_stream_to_c():
    graph = _graph()
    torch.manual_seed(123)
    control = _network(seed=123, dropout=0.3).train()
    control_state = torch.get_rng_state().clone()
    torch.manual_seed(123)
    deep = _network(seed=123, dropout=0.3, encoder_depth=2).train()
    assert torch.equal(torch.get_rng_state(), control_state), 'depth-2 construction touched the global RNG'
    control_states, _ = _three_training_forwards(control, graph)
    deep_states, outputs = _three_training_forwards(deep, graph)
    for index, (state, expected) in enumerate(zip(deep_states, control_states)):
        assert torch.equal(state, expected), f'global RNG state differs after forward {index + 1}'
    # The block dropout is real (masks change between forwards) and seeded by (seed, name):
    assert not torch.equal(outputs[0], outputs[1])
    torch.manual_seed(123)
    replay = _network(seed=123, dropout=0.3, encoder_depth=2).train()
    _, replayed = _three_training_forwards(replay, graph)
    assert all(torch.equal(a, b) for a, b in zip(outputs, replayed))
    torch.manual_seed(123)
    other = _network(seed=123, dropout=0.3, init_seed=2025, encoder_depth=2).train()
    _share_weights(deep, other)
    _, other_outputs = _three_training_forwards(other, graph)
    assert not torch.equal(outputs[0], other_outputs[0]), 'block dropout ignored the init seed'
    deep.eval(), other.eval()
    assert torch.equal(_run(deep, graph)['logits'], _run(other, graph)['logits'])
