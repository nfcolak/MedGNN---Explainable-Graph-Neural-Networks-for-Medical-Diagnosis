"""Synthetic tests for the CEI-GNN v3 E6b additive comorbid pair block (unit X5).

Extensions spec §5.3 (additive form, D10.6), §5.5 items 1–2 and §9 X5. Fixtures extend the
U3 synthetic graph with diagnosis nodes joined by ``comorbid_with`` edges; nothing here
opens a data file, run directory or checkpoint.
"""
from __future__ import annotations

import pytest
import torch

from comparison.standardized.clinical_graph_v2 import NODE_KINDS
from comparison.standardized.clinical_graph_v2.cei_v3_ext import comorbid_block as cb
from comparison.standardized.clinical_graph_v2.methods import cei_gnn_v3 as v3
from comparison.standardized.clinical_graph_v2.tensorize import ALL_RELATIONS
from tests.test_cei_gnn_v3_core import (CLASSES, EDGE_DIM, HIDDEN, KIND, VOCAB, _graph,
                                        _randomise_gates, _run, _share_weights)
from tests.test_cei_gnn_v3_hooks import _network, _rebuild

LAYOUT = {name: index for index, name in enumerate(ALL_RELATIONS)}
COMORBID = LAYOUT['comorbid_with']
PRIOR = LAYOUT['has_prior_diagnosis']
RECURRENCE = LAYOUT['recurrence_of']
COMORBID_KEYS = ('comorbid_contributions', 'comorbid_pairs', 'comorbid_gates',
                 'comorbid_denominator')


def _comorbid_graph(seed=7):
    """U3 graph plus three prior-diagnosis nodes 8, 9, 10.

    ``comorbid_with`` records: (8, 9) coded together in two past encounters (4 directed
    records), (9, 10) once (2 records); 8 → 10 carries a non-comorbid relation. Unique
    unordered pairs: {(8, 9), (9, 10)}.
    """
    graph = _graph(seed)
    generator = torch.Generator().manual_seed(seed + 100)
    extra_x = torch.randn(3, graph.x.size(1), generator=generator)
    extra_x[:, 1] = 0.0
    extra_x[:, 2] = 0.0
    graph.x = torch.cat((graph.x, extra_x), dim=0)
    graph.node_type = torch.cat((graph.node_type, torch.tensor([KIND['diagnosis']] * 3)))
    graph.token = torch.cat((graph.token, torch.tensor([VOCAB['diagnosis:d']] * 3)))
    src = [0, 0, 0, 8, 9, 8, 9, 9, 10, 8]
    dst = [8, 9, 10, 9, 8, 9, 8, 10, 9, 10]
    relation = [PRIOR, PRIOR, PRIOR, COMORBID, COMORBID, COMORBID, COMORBID, COMORBID,
                COMORBID, RECURRENCE]
    graph.edge_index = torch.cat((graph.edge_index, torch.tensor([src, dst])), dim=1)
    graph.edge_attr = torch.cat((graph.edge_attr,
                                 torch.randn(len(src), EDGE_DIM, generator=generator)), dim=0)
    graph.edge_relation = torch.cat((graph.edge_relation, torch.tensor(relation)))
    graph.edge_triple = torch.cat((graph.edge_triple, torch.zeros(len(src), dtype=torch.long)))
    return graph


def _batch(*graphs):
    from torch_geometric.data import Batch

    return Batch.from_data_list(list(graphs))


def _block(hidden=HIDDEN, num_classes=CLASSES, *, layout=None, seed=1234):
    return cb.build_block(hidden, num_classes, relation_layout=dict(layout or LAYOUT), seed=seed)


def _comorbid_network(arm='C', *, seed=11, dropout=0.0, init_seed=1234, hidden=HIDDEN,
                      **extra):
    block = _block(hidden, seed=init_seed)
    return _network(arm, seed=seed, dropout=dropout, init_seed=init_seed, hidden=hidden,
                    extra_blocks=(block,), **extra), block


# ---------------------------------------------------- step 1: comorbid pair extraction

def test_comorbid_pairs_dedupe_both_directions_and_repeats_and_ignore_other_relations():
    graph = _comorbid_graph()
    batch_index = torch.zeros(graph.num_nodes, dtype=torch.long)
    pairs = cb.comorbid_pairs(graph.edge_index, graph.edge_relation, COMORBID, batch_index)
    assert pairs.dtype == torch.long and pairs.ndim == 2 and pairs.size(0) == 2
    assert pairs.tolist() == [[8, 9], [9, 10]]
    assert bool((pairs[0] < pairs[1]).all())
    # Only the relation id passed in counts: asking for another relation finds its pairs.
    recurrence = cb.comorbid_pairs(graph.edge_index, graph.edge_relation, RECURRENCE, batch_index)
    assert recurrence.tolist() == [[8], [10]]
    # A single directed record and its reverse collapse to one pair regardless of order.
    reversed_index = graph.edge_index.flip(0)
    assert torch.equal(cb.comorbid_pairs(reversed_index, graph.edge_relation, COMORBID, batch_index),
                       pairs)
    none = cb.comorbid_pairs(graph.edge_index, graph.edge_relation, LAYOUT['co_complaint'], batch_index)
    assert tuple(none.shape) == (2, 0) and none.dtype == torch.long


def test_comorbid_pairs_follow_batch_offsets_and_never_cross_graphs():
    batch = _batch(_comorbid_graph(), _graph(seed=8), _comorbid_graph(seed=9))
    offset = _comorbid_graph().num_nodes + _graph().num_nodes
    pairs = cb.comorbid_pairs(batch.edge_index, batch.edge_relation, COMORBID, batch.batch)
    assert pairs.tolist() == [[8, 9, offset + 8, offset + 9], [9, 10, offset + 9, offset + 10]]
    assert torch.equal(batch.batch[pairs[0]], batch.batch[pairs[1]])
    crossing = batch.edge_index.clone()
    assert int(batch.edge_relation[-2]) == COMORBID   # the last comorbid record of graph 2
    crossing[1, -2] = 0                                # ... now ends in graph 0
    with pytest.raises(ValueError, match='graph'):
        cb.comorbid_pairs(crossing, batch.edge_relation, COMORBID, batch.batch)
    with pytest.raises(ValueError, match='edge_relation'):
        cb.comorbid_pairs(batch.edge_index, batch.edge_relation[:-1], COMORBID, batch.batch)


def test_build_block_resolves_the_comorbid_relation_by_name():
    block = _block()
    assert block.name == 'comorbid' and block.uses_rng is False
    assert isinstance(block, v3.ExtraBlock)
    assert block.relation_id == COMORBID
    permuted = {name: index for index, name in enumerate(reversed(ALL_RELATIONS))}
    assert permuted['comorbid_with'] != COMORBID
    assert _block(layout=permuted).relation_id == permuted['comorbid_with']
    with pytest.raises(ValueError, match='comorbid_with'):
        _block(layout={name: index for index, name in enumerate(ALL_RELATIONS)
                       if name != 'comorbid_with'})
    graph = _comorbid_graph()
    h = torch.randn(graph.num_nodes, HIDDEN, generator=torch.Generator().manual_seed(1))
    _, parts = block(h, graph.edge_index, graph.edge_relation,
                     torch.zeros(graph.num_nodes, dtype=torch.long), 1)
    assert parts['comorbid_pairs'].tolist() == [[8, 9], [9, 10]]


def test_graph_without_comorbid_edges_yields_an_exact_zero_block_and_denominator_one():
    graph = _graph()
    base = _network().eval()
    _randomise_gates(base)
    network, _ = _comorbid_network()
    network.eval()
    _share_weights(base, network)
    reference, parts = _run(base, graph), _run(network, graph)
    assert set(parts) - set(reference) == set(COMORBID_KEYS)
    assert torch.equal(parts['comorbid_denominator'], torch.ones(1, CLASSES))
    assert tuple(parts['comorbid_contributions'].shape) == (0, CLASSES)
    assert tuple(parts['comorbid_pairs'].shape) == (2, 0)
    assert parts['comorbid_pairs'].dtype == torch.long
    assert tuple(parts['comorbid_gates'].shape) == (CLASSES,)
    assert torch.equal(parts['logits'], reference['logits'])
    for key in reference:
        assert torch.equal(parts[key], reference[key]), f'{key} changed by an empty comorbid block'
    batch = _batch(_graph(), _comorbid_graph(seed=8), _graph(seed=9))
    parts = _run(network, batch)
    assert tuple(parts['comorbid_denominator'].shape) == (3, CLASSES)
    assert torch.equal(parts['comorbid_denominator'][0], torch.ones(CLASSES))
    assert torch.equal(parts['comorbid_denominator'][2], torch.ones(CLASSES))
    assert parts['comorbid_pairs'].size(1) == 2
    assert bool((batch.batch[parts['comorbid_pairs'][0]] == 1).all())


# -------------------------------------------------------- step 2: comorbid block maths

PARAMETER_NAMES = ('extra_blocks.comorbid.pair_projection.weight',
                   'extra_blocks.comorbid.vote.weight', 'extra_blocks.comorbid.vote.bias',
                   'extra_blocks.comorbid.gate')
CONTROL_HIDDEN, CONTROL_CLASSES = 128, 10
E6B_DELTA_AT_128 = 128 * 16 + 16 * 10 + 10 + 10   # +2,228 (spec §5.3)


def _count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def _randomise_block(block, seed=21):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator))


def _expected_block(block, h, pairs, batch_index, graph_count):
    """§5.3 formulas written out independently of the implementation."""
    P_a, V_a = block.pair_projection.weight, block.vote.weight
    b_a, gamma = block.vote.bias, block.gate
    left, right = pairs
    a = torch.tanh(h[left] @ P_a.t()) + torch.tanh(h[right] @ P_a.t())     # [pairs, 16]
    v = a @ V_a.t() + b_a                                                   # [pairs, C]
    g = gamma.sigmoid()                                                     # [C]
    graph = batch_index[left]
    denominator = torch.ones(graph_count, g.numel()).index_add(0, graph, g.expand(pairs.size(1), -1))
    total = torch.zeros(graph_count, g.numel()).index_add(0, graph, g * v) / denominator
    return total, g * v / denominator[graph], g, denominator


def test_block_parameters_have_the_spec_shapes_names_and_analytic_delta():
    block = _block(CONTROL_HIDDEN, CONTROL_CLASSES)
    shapes = {name: tuple(p.shape) for name, p in block.named_parameters()}
    assert shapes == {'pair_projection.weight': (16, CONTROL_HIDDEN),
                      'vote.weight': (CONTROL_CLASSES, 16), 'vote.bias': (CONTROL_CLASSES,),
                      'gate': (CONTROL_CLASSES,)}
    assert getattr(block.pair_projection, 'bias', None) is None
    assert _count(block) == E6B_DELTA_AT_128 == 2228
    assert cb.PAIR_RANK == 16
    assert tuple(block.parameter_names()) == PARAMETER_NAMES
    network, small = _comorbid_network()
    assert _count(network) - _count(_network()) == HIDDEN * 16 + 16 * CLASSES + 2 * CLASSES
    assert {name for name, _ in network.named_parameters()} >= set(PARAMETER_NAMES)
    inventory = network.parameter_inventory()
    assert inventory['extra_blocks.comorbid.pair_projection.weight'] == ((16, HIDDEN), True)
    assert inventory['extra_blocks.comorbid.gate'] == ((CLASSES,), True)
    assert network.inactive_parameter_count() == 0
    assert network.extra_blocks['comorbid'] is small


def test_block_tensors_are_initialised_from_per_tensor_generators_and_zero_gates():
    torch.manual_seed(1)
    first = _block(seed=1234)
    torch.manual_seed(999)
    second = _block(seed=1234)
    other = _block(seed=2025)
    for (name, parameter), (_, again) in zip(first.named_parameters(), second.named_parameters()):
        assert torch.equal(parameter, again), f'{name} depends on the global stream'
    bound = 1.0 / HIDDEN ** 0.5   # nn.Linear default bound at fan_in = hidden
    expected = torch.empty(16, HIDDEN).uniform_(
        -bound, bound, generator=v3.tensor_generator(1234, 'extra_blocks.comorbid.pair_projection.weight'))
    assert torch.equal(first.pair_projection.weight, expected)
    assert not torch.equal(other.pair_projection.weight, expected)
    bound = 1.0 / 16 ** 0.5      # fan_in = 16 for V_a and b_a
    assert torch.equal(first.vote.weight, torch.empty(CLASSES, 16).uniform_(
        -bound, bound, generator=v3.tensor_generator(1234, 'extra_blocks.comorbid.vote.weight')))
    assert torch.equal(first.vote.bias, torch.empty(CLASSES).uniform_(
        -bound, bound, generator=v3.tensor_generator(1234, 'extra_blocks.comorbid.vote.bias')))
    assert torch.equal(first.gate, torch.zeros(CLASSES))   # v2 pair_gate / v3 absence_gate pattern
    assert first.pair_projection.weight.abs().sum() > 0 and first.vote.weight.abs().sum() > 0


def test_block_call_implements_the_additive_formula_with_symmetric_pair_votes():
    block = _block()
    _randomise_block(block)
    batch = _batch(_comorbid_graph(), _graph(seed=8), _comorbid_graph(seed=9))
    h = torch.randn(batch.num_nodes, HIDDEN, generator=torch.Generator().manual_seed(2))
    total, parts = block(h, batch.edge_index, batch.edge_relation, batch.batch, 3)
    pairs = parts['comorbid_pairs']
    assert pairs.size(1) == 4
    expected_total, contributions, gates, denominator = _expected_block(block, h, pairs, batch.batch, 3)
    torch.testing.assert_close(total, expected_total)
    torch.testing.assert_close(parts['comorbid_contributions'], contributions)
    torch.testing.assert_close(parts['comorbid_gates'], gates)
    torch.testing.assert_close(parts['comorbid_denominator'], denominator)
    assert tuple(parts['comorbid_contributions'].shape) == (4, CLASSES)
    assert tuple(parts['comorbid_gates'].shape) == (CLASSES,)
    assert tuple(parts['comorbid_denominator'].shape) == (3, CLASSES)
    assert torch.equal(parts['comorbid_denominator'][1], torch.ones(CLASSES))
    assert torch.count_nonzero(total[1]) == 0
    assert total.abs().sum() > 0
    # Symmetry: swapping every endpoint (v_ji) reproduces the block bit-for-bit, and the
    # per-pair vote is the same function of (h_i, h_j) and (h_j, h_i).
    swapped_total, swapped_parts = block(h, batch.edge_index.flip(0), batch.edge_relation, batch.batch, 3)
    assert torch.equal(swapped_total, total)
    assert torch.equal(swapped_parts['comorbid_contributions'], parts['comorbid_contributions'])
    swapped_h = h.clone()
    swapped_h[[8, 9]] = h[[9, 8]]
    _, mirrored = block(swapped_h, batch.edge_index, batch.edge_relation, batch.batch, 3)
    torch.testing.assert_close(mirrored['comorbid_contributions'][0], parts['comorbid_contributions'][0])
    # The gate is shared by every pair: one logit per class, applied to all pairs, so
    # contribution × denominator / gate recovers the raw vote v_ij of §5.3 for every pair.
    raw = parts['comorbid_contributions'] * parts['comorbid_denominator'][batch.batch[pairs[0]]] / gates
    P_a, V_a = block.pair_projection.weight, block.vote.weight
    a = torch.tanh(h[pairs[0]] @ P_a.t()) + torch.tanh(h[pairs[1]] @ P_a.t())
    torch.testing.assert_close(raw, a @ V_a.t() + block.vote.bias)


def test_reconstruction_holds_and_existing_keys_are_unchanged():
    graph = _comorbid_graph()
    base = _network().eval()
    _randomise_gates(base)
    with torch.no_grad():
        base.absence_vote.normal_()
        base.absence_gate.normal_()
    network, block = _comorbid_network()
    network.eval()
    _share_weights(base, network)
    _randomise_block(block)
    reference, parts = _run(base, graph), _run(network, graph)
    assert set(parts) - set(reference) == set(COMORBID_KEYS)
    for key in reference:
        if key != 'logits':
            assert torch.equal(parts[key], reference[key]), f'{key} changed by the comorbid block'
    assert not torch.allclose(parts['logits'], reference['logits'])
    assert parts['comorbid_pairs'].tolist() == [[8, 9], [9, 10]]
    torch.testing.assert_close(parts['logits'][0], _rebuild(parts, 'comorbid_contributions'),
                               rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(parts['logits'], reference['logits'] + parts['comorbid_contributions'].sum(0))
    torch.testing.assert_close(_run(network, graph, return_parts=False), parts['logits'])
    batch = _batch(_graph(), _comorbid_graph(seed=8), _graph(seed=9))
    batched, single = _run(network, batch), _run(network, _comorbid_graph(seed=8))
    torch.testing.assert_close(batched['logits'][1], single['logits'][0])
    torch.testing.assert_close(batched['comorbid_contributions'], single['comorbid_contributions'])
    edge_graph = batch.batch[batch.edge_index[0]]
    for index in range(3):
        torch.testing.assert_close(batched['logits'][index],
                                   _rebuild(_per_graph(batched, batch.batch, edge_graph, index),
                                            'comorbid_contributions'),
                                   rtol=1e-5, atol=1e-5)
    parts = _run(network.train(), graph)
    parts['logits'].sum().backward()
    for name, parameter in block.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name


def _per_graph(parts, batch_index, edge_graph, index):
    """Contribution rows of graph ``index`` for every block (per-graph reconstruction)."""
    def rows(key, owner):
        value = parts[key]
        return value[owner == index] if value.size(0) else value

    pairs, absence, comorbid = parts['pairs'], parts['absence_items'], parts['comorbid_pairs']
    return {'bias': parts['bias'],
            'node_contributions': rows('node_contributions', batch_index),
            'edge_contributions': rows('edge_contributions', edge_graph),
            'pair_contributions': rows('pair_contributions', batch_index[pairs[0]]),
            'absence_contributions': rows('absence_contributions', absence[0]),
            'comorbid_contributions': rows('comorbid_contributions', batch_index[comorbid[0]])}
