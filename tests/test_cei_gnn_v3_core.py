"""Synthetic tests for the CEI-GNN v3 core network and adapter (unit U3).

Everything here is synthetic: no data file, run directory or checkpoint is opened.
Spec: v3 design §4.2–§4.5 as amended by §12 (F3, F8, F9, F11, F12, F13, F18) and
extensions spec §9 U3.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from comparison.standardized.clinical_graph_v2 import NODE_KINDS
from comparison.standardized.clinical_graph_v2.cei_v3_absence import fit_universe
from comparison.standardized.clinical_graph_v2.cei_v3_ple import fit_knots
from comparison.standardized.clinical_graph_v2.methods import cei_gnn_v3 as v3
from comparison.standardized.clinical_graph_v2.methods.base import read_clinical_batch
from comparison.standardized.clinical_graph_v2.methods.cei_gnn_v2 import PairEvidenceNetwork
from comparison.standardized.clinical_graph_v2.tensorize import ClinicalGraphData

# --------------------------------------------------------------------- fixtures
# Vocabulary: index 0 = UNK. Kind ids by name, never literal.
TOKENS = ['complaint:a', 'measurement:lab1', 'vital:hr', 'analyte:x', 'measurement:lab2',
          'diagnosis:d', 'visit:index']
VOCAB = {token: i + 1 for i, token in enumerate(TOKENS)}
NUM_TOKENS = len(TOKENS) + 1
KIND = {name: NODE_KINDS.index(name) for name in NODE_KINDS}
# Synthetic node feature layout: scaled_value / has_value sit at columns 1 and 2.
LAYOUT = ['kind:complaint', 'scaled_value', 'has_value', 'time_signed_log']
NODE_DIM, EDGE_DIM, CLASSES, HIDDEN, TOKEN_DIM = len(LAYOUT), 2, 3, 8, 4
TRIPLES, RELATIONS, PAIR_RANK, K = 4, 15, 4, 4
NEW_PARAMETERS = {'ple_projection.weight', 'absence_vote', 'absence_gate'}


def _knot_table():
    lab1 = np.linspace(-2.0, 2.0, 100, dtype=np.float32)
    hr = np.linspace(0.0, 1.0, 40, dtype=np.float32)
    lab2 = np.full((25,), 0.5, dtype=np.float32)  # one distinct knot -> inactive row
    return fit_knots({'measurement:lab1': lab1, 'vital:hr': hr, 'measurement:lab2': lab2},
                     K=K, min_values=20)


def _universe():
    counts = {'measurement:lab1': 100, 'vital:hr': 30, 'measurement:lab2': 25,
              'analyte:x': 5}
    return fit_universe(counts, dict(VOCAB), min_graphs=20, token_min_count=20)


def _graph(seed=7, *, hr_at_index=False, lab1_value=0.7):
    """Two visits (index visit = 1). Node 3 (lab1) in both visits, node 4 (hr) in
    visit 0 only unless ``hr_at_index``, node 6 (lab2) at the index visit."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(8, NODE_DIM, generator=generator)
    x[:, 1] = 0.0
    x[:, 2] = 0.0
    x[3, 1], x[3, 2] = lab1_value, 1.0
    x[4, 1], x[4, 2] = 0.25, 1.0
    x[6, 1], x[6, 2] = 0.0, 0.0  # invalid value: zero basis, still present
    graph = ClinicalGraphData(
        x=x, edge_index=torch.tensor([[0, 1, 1, 1, 3, 6], [1, 2, 3, 4, 5, 5]]),
        edge_attr=torch.randn(6, EDGE_DIM, generator=generator))
    graph.node_type = torch.tensor([KIND['patient'], KIND['visit'], KIND['complaint'],
                                    KIND['measurement'], KIND['vital'], KIND['analyte'],
                                    KIND['measurement'], KIND['complaint']])
    graph.token = torch.tensor([0, VOCAB['visit:index'], VOCAB['complaint:a'],
                                VOCAB['measurement:lab1'], VOCAB['vital:hr'],
                                VOCAB['analyte:x'], VOCAB['measurement:lab2'],
                                VOCAB['complaint:a']])
    graph.edge_relation = torch.tensor([0, 2, 3, 5, 4, 4])
    graph.edge_triple = torch.tensor([0, 1, 2, 3, 1, 1])
    membership = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 6), (1, 7), (1, 3)]
    if hr_at_index:
        membership.append((1, 4))
    graph.visit_membership_index = torch.tensor(sorted(membership)).t().contiguous()
    graph.num_visits = torch.tensor([2])
    graph.y = torch.tensor([1])
    return graph


def _metadata(graph):
    return read_clinical_batch(graph, method='test', node_dim=NODE_DIM, edge_dim=EDGE_DIM,
                               num_tokens=NUM_TOKENS, num_triples=TRIPLES,
                               num_relations=RELATIONS)


def _visit_graph(graph):
    counts = graph.num_visits.view(-1).long()
    return torch.repeat_interleave(torch.arange(counts.numel()), counts)


def _v2(seed=11, dropout=0.0):
    torch.manual_seed(seed)
    return PairEvidenceNetwork(
        num_tokens=NUM_TOKENS, node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_classes=CLASSES,
        hidden=HIDDEN, token_dim=TOKEN_DIM, num_triples=TRIPLES, num_relations=RELATIONS,
        dropout=dropout, pair_rank=PAIR_RANK, pair_mode='additive',
        num_node_types=len(NODE_KINDS))


def _v3(arm, seed=11, dropout=0.0, *, layout=LAYOUT, init_seed=1234, **extra):
    table, universe = _knot_table(), _universe()
    knots, active, _ = table.tensor()
    rows = table.rows()
    knot_row_of_token = torch.full((NUM_TOKENS,), -1, dtype=torch.long)
    for item, row in rows.items():
        knot_row_of_token[VOCAB[item]] = row
    torch.manual_seed(seed)
    return v3.EvidenceNetworkV3(
        num_tokens=NUM_TOKENS, node_dim=len(layout), edge_dim=EDGE_DIM, num_classes=CLASSES,
        hidden=HIDDEN, token_dim=TOKEN_DIM, num_triples=TRIPLES, num_relations=RELATIONS,
        dropout=dropout, pair_rank=PAIR_RANK, num_node_types=len(NODE_KINDS), arm=arm,
        knots=knots, knot_active=active, slot_of_token=universe.slot_of_token(NUM_TOKENS),
        universe_size=len(universe), feature_layout=list(layout), seed=init_seed,
        knot_row_of_token=knot_row_of_token, **extra)


def _randomise_gates(network):
    with torch.no_grad():
        network.pair_gate.normal_()
        network.edge_context_gate.bias.normal_()


def _run(network, graph, *, return_parts=True, features=None, visit_graph=True):
    metadata = _metadata(graph)
    if features is None:
        features = network.continuous_inputs(metadata)
    extra = {'visit_graph': _visit_graph(graph)} if visit_graph else {}
    return network.forward_continuous(features, metadata.edge_index, metadata,
                                      graph.visit_membership_index,
                                      return_parts=return_parts, **extra)


def _share_weights(source, target):
    missing, unexpected = target.load_state_dict(source.state_dict(), strict=False)
    return set(missing), set(unexpected)


V2_KEYS = ('logits', 'node_contributions', 'edge_contributions', 'pair_contributions',
           'pairs', 'bias', 'pair_gates', 'pair_denominator')


# ------------------------------------------------------------ step 1: arm A parity

def test_arm_a_reproduces_v2_additive_logits_and_parts_on_shared_weights():
    v2 = _v2().eval()
    _randomise_gates(v2)
    arm_a = _v3('A').eval()
    missing, unexpected = _share_weights(v2, arm_a)
    assert unexpected == set(), unexpected
    assert missing == NEW_PARAMETERS, missing
    graph = _graph()
    expected = _run(v2, graph, visit_graph=False)
    actual = _run(arm_a, graph)
    for key in V2_KEYS:
        assert key in actual, key
        assert torch.equal(actual[key], expected[key]), f'{key} differs between arm A and v2'


def test_arm_a_matches_v2_in_training_mode_with_identical_dropout_draws():
    v2, arm_a = _v2(dropout=0.3).train(), _v3('A', dropout=0.3).train()
    _share_weights(v2, arm_a)
    graph = _graph()
    torch.manual_seed(5)
    expected = _run(v2, graph, visit_graph=False)['logits']
    torch.manual_seed(5)
    actual = _run(arm_a, graph)['logits']
    assert torch.equal(actual, expected)


def _three_training_forwards(network):
    graph = _graph()
    metadata = _metadata(graph)
    features = network.continuous_inputs(metadata)
    torch.manual_seed(947)
    states = []
    for _ in range(3):
        network.forward_continuous(features, metadata.edge_index, metadata,
                                   graph.visit_membership_index, return_parts=True,
                                   visit_graph=_visit_graph(graph))
        states.append(torch.get_rng_state().clone())
    return states


def test_rng_state_equals_v2_additive_after_construction_and_three_training_forwards():
    torch.manual_seed(123)
    _v2(seed=123, dropout=0.3)
    expected_construction = torch.get_rng_state().clone()
    networks = {}
    for arm in ('A', 'B', 'C'):
        networks[arm] = _v3(arm, seed=123, dropout=0.3).train()
        assert torch.equal(torch.get_rng_state(), expected_construction), \
            f'arm {arm} construction consumed the global RNG differently from v2'
    expected_states = _three_training_forwards(_v2(seed=123, dropout=0.3).train())
    for arm, network in networks.items():
        states = _three_training_forwards(network)
        for index, (state, expected) in enumerate(zip(states, expected_states)):
            assert torch.equal(state, expected), f'RNG state differs: arm {arm}, forward {index + 1}'


def test_new_tensors_are_initialised_from_per_tensor_generators_not_the_global_stream():
    torch.manual_seed(1)
    first = _v3('C', seed=1, init_seed=1234)
    torch.manual_seed(999)
    second = _v3('C', seed=999, init_seed=1234)
    other = _v3('C', seed=999, init_seed=2025)
    assert torch.equal(first.ple_projection.weight, second.ple_projection.weight)
    assert not torch.equal(first.ple_projection.weight, other.ple_projection.weight)
    assert first.ple_projection.weight.abs().sum() > 0


def test_ple_projection_is_bias_free_and_v2_parameter_names_and_shapes_are_kept():
    v2, arm_c = _v2(), _v3('C')
    v2_shapes = {name: tuple(p.shape) for name, p in v2.named_parameters()}
    v3_shapes = {name: tuple(p.shape) for name, p in arm_c.named_parameters()}
    assert set(v3_shapes) == set(v2_shapes) | NEW_PARAMETERS
    assert all(v3_shapes[name] == shape for name, shape in v2_shapes.items())
    assert 'ple_projection.bias' not in v3_shapes
    assert getattr(arm_c.ple_projection, 'bias', None) is None
    assert v3_shapes['ple_projection.weight'] == (HIDDEN, K + 1)
    assert v3_shapes['absence_vote'] == (3, CLASSES)
    assert v3_shapes['absence_gate'] == (3, CLASSES)
    assert arm_c.pair_mode == 'additive'


def test_ple_columns_are_resolved_by_layout_name():
    permuted = ['time_signed_log', 'has_value', 'kind:complaint', 'scaled_value']
    graph = _graph()
    torch.manual_seed(3)
    reference = _v3('B', layout=LAYOUT).eval()
    torch.manual_seed(3)
    swapped = _v3('B', layout=permuted).eval()
    _share_weights(reference, swapped)
    assert swapped.scaled_value_column == permuted.index('scaled_value')
    assert swapped.has_value_column == permuted.index('has_value')
    order = [LAYOUT.index(name) for name in permuted]
    with torch.no_grad():  # the v2 encoder reads x by position: permute its columns too
        swapped.node_encoder.weight[:, :NODE_DIM] = reference.node_encoder.weight[:, order]
    expected = _run(reference, graph)['logits']
    swapped_graph = _graph()
    swapped_graph.x = graph.x[:, order]
    actual = _run(swapped, swapped_graph)['logits']
    # Column permutation changes the matmul summation order: equal up to float32 rounding.
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError, match='scaled_value'):
        _v3('B', layout=['a', 'has_value', 'b', 'c'])


def test_arm_b_ple_changes_logits_only_through_active_knot_rows():
    graph = _graph()
    graph.x[6, 2] = 1.0  # lab2 now carries a valid value, but its knot row is inactive
    torch.manual_seed(4)
    arm_a = _v3('A').eval()
    torch.manual_seed(4)
    arm_b = _v3('B').eval()
    _share_weights(arm_a, arm_b)
    with torch.no_grad():  # a non-uniform projection: a constant vector would be removed by LayerNorm
        arm_b.ple_projection.weight.copy_(torch.randn(
            arm_b.ple_projection.weight.shape, generator=torch.Generator().manual_seed(77)))
    assert not torch.allclose(_run(arm_b, graph)['logits'], _run(arm_a, graph)['logits'])
    # Detaching the inactive lab2 row from the table (row -1 = zero basis) changes nothing;
    # detaching the active lab1 row does.
    torch.manual_seed(4)
    detached = _v3('B').eval()
    _share_weights(arm_b, detached)
    detached.knot_row_of_token[VOCAB['measurement:lab2']] = -1
    assert torch.equal(_run(detached, graph)['logits'], _run(arm_b, graph)['logits'])
    detached.knot_row_of_token[VOCAB['measurement:lab1']] = -1
    assert not torch.equal(_run(detached, graph)['logits'], _run(arm_b, graph)['logits'])
