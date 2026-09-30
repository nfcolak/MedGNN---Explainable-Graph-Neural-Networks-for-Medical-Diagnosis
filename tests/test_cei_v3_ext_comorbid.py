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


# ---------------------------------------------------- step 3: comorbid block isolation

from tests.test_cei_gnn_v3_core import _adapter  # noqa: E402
from tests.test_cei_gnn_v3_hooks import _three_training_forwards  # noqa: E402


def test_zero_edge_mask_removes_the_edge_block_and_leaves_the_comorbid_block_intact():
    graph = _comorbid_graph()
    network, block = _comorbid_network()
    network.eval()
    _randomise_gates(network)
    _randomise_block(block)
    ordinary = _run(network, graph)
    assert ordinary['comorbid_contributions'].abs().sum() > 0
    network.set_edge_mask(torch.zeros(graph.num_edges))
    masked = _run(network, graph)
    network.set_edge_mask(None)
    assert torch.count_nonzero(masked['edge_contributions']) == 0
    for key in ('node_contributions', 'pair_contributions', 'absence_contributions',
                'comorbid_contributions', 'comorbid_denominator', 'comorbid_gates'):
        torch.testing.assert_close(masked[key], ordinary[key]), key
    assert torch.equal(masked['comorbid_pairs'], ordinary['comorbid_pairs'])
    expected = (masked['bias'] + ordinary['node_contributions'].sum(0)
                + ordinary['pair_contributions'].sum(0) + ordinary['absence_contributions'].sum(0)
                + ordinary['comorbid_contributions'].sum(0))
    torch.testing.assert_close(masked['logits'][0], expected, rtol=1e-5, atol=1e-5)
    # Masking only the comorbid_with edges to zero also leaves the block intact: the pair
    # set is derived from the edge list, not from the mask (§5.3 / E15).
    mask = torch.ones(graph.num_edges)
    mask[graph.edge_relation == COMORBID] = 0.0
    network.set_edge_mask(mask)
    partial = _run(network, graph)
    network.set_edge_mask(None)
    torch.testing.assert_close(partial['comorbid_contributions'], ordinary['comorbid_contributions'])
    assert torch.equal(partial['comorbid_pairs'], ordinary['comorbid_pairs'])
    assert torch.count_nonzero(partial['edge_contributions'][graph.edge_relation == COMORBID]) == 0
    assert not torch.allclose(partial['logits'], ordinary['logits'])


def test_comorbid_block_consumes_no_global_rng_in_construction_or_training_forwards():
    graph = _comorbid_graph()
    torch.manual_seed(123)
    control = _network(seed=123, dropout=0.3).train()
    control_state = torch.get_rng_state().clone()
    torch.manual_seed(123)
    seeded = torch.get_rng_state().clone()
    block = _block(seed=1234)
    assert torch.equal(torch.get_rng_state(), seeded), 'block construction touched the global RNG'
    e6b = _network(seed=123, dropout=0.3, extra_blocks=(block,)).train()
    assert torch.equal(torch.get_rng_state(), control_state), 'E6b construction touched the global RNG'
    common = dict(control.named_parameters())
    for name, parameter in e6b.named_parameters():
        if name in common:
            assert torch.equal(parameter, common[name]), f'E6b: {name} differs from C'
    assert set(dict(e6b.named_parameters())) - set(common) == set(PARAMETER_NAMES)
    _randomise_block(block)
    control_states, control_outputs = _three_training_forwards(control, graph)
    e6b_states, outputs = _three_training_forwards(e6b, graph)
    for index, (state, expected) in enumerate(zip(e6b_states, control_states)):
        assert torch.equal(state, expected), f'E6b: global RNG state differs after forward {index + 1}'
    # The block itself is deterministic: identical inputs give identical outputs in
    # training mode (no dropout inside the block), while the v2 dropouts still vary.
    assert not torch.equal(outputs[0], outputs[1])
    h = torch.randn(graph.num_nodes, HIDDEN, generator=torch.Generator().manual_seed(4))
    batch_index = torch.zeros(graph.num_nodes, dtype=torch.long)
    first, _ = block(h, graph.edge_index, graph.edge_relation, batch_index, 1)
    second, _ = block(h, graph.edge_index, graph.edge_relation, batch_index, 1)
    assert torch.equal(first, second)
    e6b.eval(), control.eval()
    delta = _run(e6b, graph)['logits'] - _run(control, graph)['logits']
    torch.testing.assert_close(delta[0], _run(e6b, graph)['comorbid_contributions'].sum(0),
                               rtol=1e-5, atol=1e-5)


def test_parameter_names_match_the_registered_set_and_the_network_inventory():
    network, block = _comorbid_network()
    declared = tuple(block.parameter_names())
    registered = tuple(f'extra_blocks.comorbid.{local}' for local, _ in block.named_parameters())
    assert sorted(declared) == sorted(registered)
    assert declared == PARAMETER_NAMES
    inventory = network.parameter_inventory()
    assert {name for name in inventory if name.startswith('extra_blocks.')} == set(PARAMETER_NAMES)
    assert inventory['extra_blocks.comorbid.pair_projection.weight'] == ((16, HIDDEN), True)
    assert inventory['extra_blocks.comorbid.vote.weight'] == ((CLASSES, 16), True)
    assert inventory['extra_blocks.comorbid.vote.bias'] == ((CLASSES,), True)
    assert inventory['extra_blocks.comorbid.gate'] == ((CLASSES,), True)
    state = network.state_dict()
    assert set(PARAMETER_NAMES) <= set(state)
    for name in PARAMETER_NAMES:
        assert torch.equal(state[name], dict(network.named_parameters())[name])
    # A v3 arm-C state_dict loads into the E6b network with exactly the block missing.
    missing, unexpected = _share_weights(_network(), network)
    assert unexpected == set() and missing == set(PARAMETER_NAMES)


def test_adapter_end_to_end_with_comorbid_block_one(tmp_path):
    control = _adapter(tmp_path, 'C')
    e6b = _adapter(tmp_path, 'C', comorbid_block=1)
    assert isinstance(e6b.network.extra_blocks['comorbid'], cb.ComorbidPairBlock)
    assert e6b.network.extra_blocks['comorbid'].relation_id == COMORBID
    config = e6b.run_config()
    assert config['comorbid_block'] == 1 and config['extra_blocks'] == ['comorbid']
    assert config['parameter_count'] - control.run_config()['parameter_count'] == (
        HIDDEN * 16 + 16 * CLASSES + 2 * CLASSES)
    assert config['inactive_parameter_count'] == 0
    for name in PARAMETER_NAMES:
        assert config['parameter_inventory'][name]['active'] is True
    # At the control width the delta is the analytic +2,228 (10 classes) and the binding
    # records common_init_identical_to_c (§2.2); the width-8 fixture is not the control width.
    assert config['common_init_identical_to_c'] is False
    wide = _adapter(tmp_path, 'C', hidden=CONTROL_HIDDEN, comorbid_block=1).run_config()
    assert wide['common_init_identical_to_c'] is True
    assert wide['parameter_count'] - _adapter(tmp_path, 'C', hidden=CONTROL_HIDDEN).run_config()['parameter_count'] == (
        CONTROL_HIDDEN * 16 + 16 * CLASSES + 2 * CLASSES)
    # The adapter builds the block between the runner seed and the network: every common
    # tensor must still start from C's initial values (§2.2), the block's from its generators.
    common = dict(control.named_parameters())
    for name, parameter in e6b.named_parameters():
        if name in common:
            assert torch.equal(parameter, common[name]), f'adapter E6b: {name} differs from C'
    assert set(dict(e6b.named_parameters())) - set(common) == {f'network.{n}' for n in PARAMETER_NAMES}
    assert torch.equal(e6b.network.extra_blocks['comorbid'].pair_projection.weight,
                       _block(seed=1234).pair_projection.weight)
    # Forward on a synthetic batch with comorbid edges: reconstruction within 1e-5.
    e6b.eval()
    _randomise_gates(e6b.network)
    _randomise_block(e6b.network.extra_blocks['comorbid'])
    batch = _batch(_comorbid_graph(), _graph(seed=8), _comorbid_graph(seed=9))
    output = e6b(batch, epoch=0)
    features = e6b.continuous_inputs(batch)
    parts = e6b.forward_continuous(features, batch.edge_index, batch, return_parts=True)
    assert torch.equal(parts['logits'], output.logits)
    assert parts['comorbid_pairs'].size(1) == 4
    edge_graph = batch.batch[batch.edge_index[0]]
    for index in range(3):
        torch.testing.assert_close(parts['logits'][index],
                                   _rebuild(_per_graph(parts, batch.batch, edge_graph, index),
                                            'comorbid_contributions'),
                                   rtol=1e-5, atol=1e-5)
    assert torch.equal(e6b.forward_continuous(features, batch.edge_index, batch), output.logits)
    # Global RNG state equals arm C's after three training forwards (§2.2 / §5.5.2).
    torch.manual_seed(31)
    control_adapter = _adapter(tmp_path, 'C', dropout=0.3, seed=31).train()
    torch.manual_seed(31)
    e6b_adapter = _adapter(tmp_path, 'C', dropout=0.3, seed=31, comorbid_block=1).train()
    states = {}
    for label, adapter in (('C', control_adapter), ('E6b', e6b_adapter)):
        torch.manual_seed(947)
        states[label] = []
        for _ in range(3):
            adapter(batch, epoch=0)
            states[label].append(torch.get_rng_state().clone())
    for index, (state, expected) in enumerate(zip(states['E6b'], states['C'])):
        assert torch.equal(state, expected), f'adapter E6b: RNG state differs after forward {index + 1}'
