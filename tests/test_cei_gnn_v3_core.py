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


def _graph(seed=7, *, hr_at_index=False, lab1_value=0.7, lab1_prior=False):
    """Two visits (index visit = 1). Node 3 (lab1) at the index visit (also visit 0 when
    ``lab1_prior``), node 4 (hr) in visit 0 only unless ``hr_at_index``, node 6 (lab2) at
    the index visit with an invalid value. Measurement/vital nodes have exactly one
    membership unless stated (the U2 membership contract)."""
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
    membership = [(0, 1), (0, 2), (1 if hr_at_index else 0, 4), (0, 5), (1, 2), (1, 3),
                  (1, 6), (1, 7)]
    if lab1_prior:
        membership.append((0, 3))
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


# ---------------------------------------------------- step 2: arm C reconstruction

ABSENCE_KEYS = ('absence_contributions', 'absence_items', 'absence_gates',
                'absence_denominator')
SLOT = {item: slot for slot, item in enumerate(_universe().items)}


def _arm_c(seed=11, dropout=0.0):
    network = _v3('C', seed=seed, dropout=dropout).eval()
    _randomise_gates(network)
    with torch.no_grad():
        network.absence_vote.normal_()
        network.absence_gate.normal_()
    return network


def _rebuild(parts):
    return (parts['bias'] + parts['node_contributions'].sum(0) + parts['edge_contributions'].sum(0)
            + parts['pair_contributions'].sum(0) + parts['absence_contributions'].sum(0))


def test_arm_c_logits_reconstruct_from_parts_with_new_keys_and_unchanged_old_keys():
    arm_c = _arm_c()
    graph = _graph()
    parts = _run(arm_c, graph)
    assert set(V2_KEYS) | set(ABSENCE_KEYS) <= set(parts), sorted(set(parts))
    torch.manual_seed(11)
    reference = _v3('A').eval()
    _share_weights(arm_c, reference)
    old = _run(reference, graph)
    for key in V2_KEYS:
        assert tuple(parts[key].shape) == tuple(old[key].shape), key
    assert parts['absence_contributions'].abs().sum() > 0
    torch.testing.assert_close(parts['logits'][0], _rebuild(parts), rtol=1e-5, atol=1e-5)
    assert not torch.allclose(parts['logits'], old['logits'])


def test_absence_block_follows_index_visit_presence_and_gate_normalisation():
    arm_c = _arm_c()
    # hr only at the prior visit -> absent; lab1 at index -> present;
    # lab2 at index with an invalid value -> present (F9).
    parts = _run(arm_c, _graph())
    items = parts['absence_items']
    assert items.dtype == torch.long and tuple(items.shape) == (2, 1)
    assert items.tolist() == [[0], [SLOT['vital:hr']]]
    gate = arm_c.absence_gate[SLOT['vital:hr']].sigmoid()
    vote = arm_c.absence_vote[SLOT['vital:hr']]
    torch.testing.assert_close(parts['absence_gates'], gate[None, :])
    torch.testing.assert_close(parts['absence_denominator'], (1.0 + gate)[None, :])
    torch.testing.assert_close(parts['absence_contributions'], (gate * vote / (1.0 + gate))[None, :])
    # hr measured at the index visit as well -> nothing absent.
    present = _run(arm_c, _graph(hr_at_index=True))
    assert tuple(present['absence_items'].shape) == (2, 0)


def test_empty_absence_set_gives_exact_zero_block_and_denominator_one():
    arm_c = _arm_c()
    graph = _graph(hr_at_index=True)
    parts = _run(arm_c, graph)
    assert tuple(parts['absence_contributions'].shape) == (0, CLASSES)
    assert torch.equal(parts['absence_denominator'], torch.ones((1, CLASSES)))
    torch.testing.assert_close(parts['logits'][0], _rebuild(parts), rtol=1e-5, atol=1e-5)
    torch.manual_seed(11)
    arm_b = _v3('B').eval()
    _share_weights(arm_c, arm_b)
    assert torch.equal(parts['logits'], _run(arm_b, graph)['logits'])


def test_absence_items_follow_batch_offsets():
    from torch_geometric.data import Batch

    arm_c = _arm_c()
    batch = Batch.from_data_list([_graph(hr_at_index=True), _graph(seed=8), _graph(seed=9)])
    parts = _run(arm_c, batch)
    assert parts['absence_items'].tolist() == [[1, 2], [SLOT['vital:hr'], SLOT['vital:hr']]]
    assert tuple(parts['absence_denominator'].shape) == (3, CLASSES)
    assert torch.equal(parts['absence_denominator'][0], torch.ones(CLASSES))
    single = _run(arm_c, _graph(seed=8))
    torch.testing.assert_close(parts['logits'][1], single['logits'][0])
    torch.testing.assert_close(parts['absence_contributions'][0], single['absence_contributions'][0])


def test_zero_edge_mask_removes_only_the_edge_block_in_arm_c():
    arm_c = _arm_c()
    graph = _graph()
    ordinary = _run(arm_c, graph)
    arm_c.set_edge_mask(torch.zeros(graph.num_edges))
    masked = _run(arm_c, graph)
    arm_c.set_edge_mask(None)
    assert torch.count_nonzero(masked['edge_contributions']) == 0
    for key in ('node_contributions', 'pair_contributions', 'absence_contributions'):
        torch.testing.assert_close(masked[key], ordinary[key])
    expected = (masked['bias'] + ordinary['node_contributions'].sum(0)
                + ordinary['pair_contributions'].sum(0) + ordinary['absence_contributions'].sum(0))
    torch.testing.assert_close(masked['logits'][0], expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('arm, ple_grad, absence_grad', [('A', False, False), ('B', True, False),
                                                          ('C', True, True)])
def test_inactive_paths_produce_no_autograd_tensors(arm, ple_grad, absence_grad):
    network = _v3(arm, dropout=0.3).train()
    _randomise_gates(network)
    graph = _graph()
    parts = _run(network, graph)
    parts['logits'].sum().backward()
    assert (network.ple_projection.weight.grad is not None) == ple_grad
    assert (network.absence_vote.grad is not None) == absence_grad
    assert (network.absence_gate.grad is not None) == absence_grad
    assert network.node_encoder.weight.grad is not None
    if not absence_grad:
        assert tuple(parts['absence_contributions'].shape) == (0, CLASSES)
        assert not parts['absence_denominator'].requires_grad
    if absence_grad:
        assert parts['absence_contributions'].requires_grad


def test_absence_label_uses_the_amended_wording():
    from comparison.standardized.clinical_graph_v2.cei_v3_absence import ABSENCE_LABEL

    label = v3.absence_label('vital:hr')
    assert label == ABSENCE_LABEL.format(item='vital:hr')
    assert label == 'vital:hr: no recorded result at this visit'
    assert 'not measured' not in label


def test_arm_c_requires_visit_graph_for_the_absence_block():
    arm_c = _arm_c()
    with pytest.raises(ValueError, match='visit_graph'):
        _run(arm_c, _graph(), visit_graph=False)


# ------------------------------------------------------- step 3: adapter options

from argparse import Namespace  # noqa: E402

from comparison.standardized.clinical_graph_v2.methods import plugin_cei_gnn_v3 as plugin  # noqa: E402

CLOSED_OPTIONS = frozenset(('arm', 'v3_state', 'k', 'encoder_depth', 'comorbid_block'))
PREP_SHA = hashlib.sha256(b'synthetic preprocessing state').hexdigest()


def _state_document(K=K, layout=LAYOUT, tokens=TOKENS):
    """The v3_state file schema U5 must produce (v3 §12 F18); pinned here by hand."""
    table, universe = _knot_table(), _universe()
    return {
        'version': 'cei_v3_state_v1',
        'K': K,
        'knot_table': table.state(),
        'knot_table_sha256': table.sha256(),
        'universe': universe.state(),
        'universe_sha256': universe.sha256(),
        'vocabulary': {'tokens': list(tokens), 'min_count': 20},
        'preprocessing_sha256': PREP_SHA,
        'node_feature_layout': list(layout),
    }


def _write_state(tmp_path, document=None, name='K4.json'):
    path = tmp_path / name
    path.write_text(json.dumps(document if document is not None else _state_document(),
                               indent=2, sort_keys=True) + '\n')
    return path


def _adapter(tmp_path, arm='C', *, hidden=HIDDEN, layers=1, seed=1234, dropout=0.0,
             state_path=None, edge_direction=None, **options):
    from comparison.standardized.clinical_graph_v2.methods import METHOD_REGISTRY

    assert 'cei_gnn_v3' in METHOD_REGISTRY, 'cei_gnn_v3 plugin is not registered'
    state_path = state_path or _write_state(tmp_path)
    method_options = {'arm': arm, 'v3_state': str(state_path), 'k': K, **options}
    args = Namespace(method_options=method_options, seed=seed)
    if edge_direction is not None:
        args.edge_direction = edge_direction
    torch.manual_seed(seed)
    return METHOD_REGISTRY['cei_gnn_v3'](
        num_tokens=NUM_TOKENS, node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_classes=CLASSES,
        hidden=hidden, layers=layers, dropout=dropout, token_dim=TOKEN_DIM, num_triples=TRIPLES,
        args=args)


def _v2_total():
    """v2 additive parameter count at the adapter's fixed pair_rank (16, the v2 default)."""
    torch.manual_seed(0)
    network = PairEvidenceNetwork(
        num_tokens=NUM_TOKENS, node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_classes=CLASSES,
        hidden=HIDDEN, token_dim=TOKEN_DIM, num_triples=TRIPLES, num_relations=RELATIONS,
        dropout=0.0, pair_rank=16, pair_mode='additive', num_node_types=len(NODE_KINDS))
    return sum(p.numel() for p in network.parameters())


def test_known_options_are_the_closed_set_and_unknown_options_are_refused(tmp_path):
    assert plugin.KNOWN_OPTIONS == CLOSED_OPTIONS
    with pytest.raises(ValueError, match='unknown method option'):
        _adapter(tmp_path, pair_mode='additive')
    with pytest.raises(ValueError, match='unknown method option'):
        _adapter(tmp_path, margin_table='x')


def test_k_option_must_equal_the_state_file_k(tmp_path):
    with pytest.raises(ValueError, match='[Kk]'):
        _adapter(tmp_path, k=K + 4)
    document = _state_document(K=K + 4)  # top-level K disagrees with the knot table
    with pytest.raises(ValueError, match='[Kk]'):
        _adapter(tmp_path, state_path=_write_state(tmp_path, document, 'bad.json'))


def test_state_loader_validates_schema_hashes_vocabulary_and_layout(tmp_path):
    good = _write_state(tmp_path)
    state = plugin.load_v3_state(good)
    assert state.K == K and state.num_tokens == NUM_TOKENS
    assert state.sha256 == hashlib.sha256(good.read_bytes()).hexdigest()
    assert state.preprocessing_sha256 == PREP_SHA
    assert state.node_feature_layout == tuple(LAYOUT)
    assert state.knot_table.items.keys() == {'measurement:lab1', 'vital:hr', 'measurement:lab2'}
    assert state.universe.items == ('measurement:lab1', 'measurement:lab2', 'vital:hr')
    rows = state.knot_row_of_token()
    assert rows.tolist() == [-1, -1, 0, 2, -1, 1, -1, -1]
    for key in ('version', 'K', 'knot_table', 'universe', 'vocabulary', 'preprocessing_sha256',
                'node_feature_layout', 'knot_table_sha256', 'universe_sha256'):
        document = _state_document()
        del document[key]
        with pytest.raises(ValueError, match=key):
            plugin.load_v3_state(_write_state(tmp_path, document, f'missing-{key}.json'))
    tampered = _state_document()
    tampered['universe']['graph_counts']['vital:hr'] = 31
    with pytest.raises(ValueError, match='universe_sha256'):
        plugin.load_v3_state(_write_state(tmp_path, tampered, 'tampered.json'))
    tampered = _state_document()
    tampered['knot_table']['items']['vital:hr']['count'] = 41
    with pytest.raises(ValueError, match='knot_table_sha256'):
        plugin.load_v3_state(_write_state(tmp_path, tampered, 'tampered2.json'))
    with pytest.raises(ValueError, match='vocabulary'):
        renamed = [token if token != 'vital:hr' else 'other' for token in TOKENS]
        plugin.load_v3_state(_write_state(tmp_path, _state_document(tokens=renamed), 'vocab.json'))
    with pytest.raises(ValueError, match='has_value'):
        plugin.load_v3_state(_write_state(tmp_path, _state_document(layout=['a', 'scaled_value']),
                                          'layout.json'))
    with pytest.raises(ValueError, match='preprocessing_sha256'):
        document = _state_document()
        document['preprocessing_sha256'] = 'abc'
        plugin.load_v3_state(_write_state(tmp_path, document, 'sha.json'))
    assert plugin.build_v3_state(
        K=K, knot_table=_knot_table(), universe=_universe(), vocabulary_tokens=TOKENS,
        vocabulary_min_count=20, preprocessing_sha256=PREP_SHA,
        node_feature_layout=LAYOUT) == _state_document()


def test_adapter_refuses_dimension_mismatch_and_missing_arm_or_seed(tmp_path):
    from comparison.standardized.clinical_graph_v2.methods import METHOD_REGISTRY

    cls = METHOD_REGISTRY['cei_gnn_v3']
    path = _write_state(tmp_path)
    common = dict(node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_classes=CLASSES, hidden=HIDDEN,
                  layers=1, dropout=0.0, token_dim=TOKEN_DIM, num_triples=TRIPLES)
    options = {'arm': 'C', 'v3_state': str(path), 'k': K}
    with pytest.raises(ValueError, match='num_tokens'):
        cls(num_tokens=NUM_TOKENS + 1, **common, args=Namespace(method_options=options, seed=1))
    with pytest.raises(ValueError, match='node_feature_layout'):
        cls(num_tokens=NUM_TOKENS, **{**common, 'node_dim': NODE_DIM + 1},
            args=Namespace(method_options=options, seed=1))
    with pytest.raises(ValueError, match='arm'):
        cls(num_tokens=NUM_TOKENS, **common,
            args=Namespace(method_options={'v3_state': str(path), 'k': K}, seed=1))
    with pytest.raises(ValueError, match='arm'):
        cls(num_tokens=NUM_TOKENS, **common,
            args=Namespace(method_options={**options, 'arm': 'D'}, seed=1))
    with pytest.raises(ValueError, match='v3_state'):
        cls(num_tokens=NUM_TOKENS, **common, args=Namespace(method_options={'arm': 'C', 'k': K}, seed=1))
    with pytest.raises(ValueError, match='seed'):
        cls(num_tokens=NUM_TOKENS, **common, args=Namespace(method_options=options))


def test_u3x_options_encoder_depth_accepted_and_invalid_values_refused(tmp_path):
    adapter = _adapter(tmp_path, encoder_depth=1, comorbid_block=0)
    assert adapter.encoder_depth == 1 and adapter.comorbid_block == 0
    assert _adapter(tmp_path, encoder_depth=2).encoder_depth == 2
    assert _adapter(tmp_path, layers=2).encoder_depth == 2
    with pytest.raises(ValueError, match='U3x'):
        _adapter(tmp_path, comorbid_block=1)
    with pytest.raises(ValueError, match='encoder_depth'):
        _adapter(tmp_path, layers=1, encoder_depth=0)


@pytest.mark.parametrize('arm, ple_active, absence_active', [('A', False, False), ('B', True, False),
                                                              ('C', True, True)])
def test_parameter_inventory_and_counts_per_arm(tmp_path, arm, ple_active, absence_active):
    adapter = _adapter(tmp_path, arm)
    inventory = adapter.network.parameter_inventory()
    v2_names = {name for name, _ in _v2().named_parameters()}
    assert set(inventory) == v2_names | NEW_PARAMETERS
    assert all(inventory[name][1] for name in v2_names)
    assert inventory['ple_projection.weight'] == ((HIDDEN, K + 1), ple_active)
    assert inventory['absence_vote'] == ((3, CLASSES), absence_active)
    assert inventory['absence_gate'] == ((3, CLASSES), absence_active)
    ple, absence = (K + 1) * HIDDEN, 2 * 3 * CLASSES
    total = _v2_total() + ple + absence
    inactive = (0 if ple_active else ple) + (0 if absence_active else absence)
    assert adapter.inactive_parameter_count() == inactive
    config = adapter.run_config()
    assert config['parameter_count'] == total
    assert config['active_parameter_count'] == total - inactive
    assert config['inactive_parameter_count'] == inactive
    assert config['architecture']['parameter_count'] == total
    assert config['architecture']['active_parameter_count'] == total - inactive
    assert config['architecture']['inactive_parameter_count'] == inactive
    assert config['parameter_inventory'] == {
        name: {'shape': list(shape), 'active': active} for name, (shape, active) in inventory.items()}


def test_run_config_binds_arm_k_state_hash_and_extension_fields(tmp_path):
    path = _write_state(tmp_path)
    adapter = _adapter(tmp_path, 'B', state_path=path)
    config = adapter.run_config()
    assert config['method'] == 'cei_gnn_v3'
    assert config['arm'] == 'B' and config['k'] == K
    assert config['v3_state_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert config['v3_state_path'] == str(path)
    assert config['preprocessing_sha256'] == PREP_SHA
    assert config['knot_table_sha256'] == _knot_table().sha256()
    assert config['universe_sha256'] == _universe().sha256()
    assert config['universe_size'] == 3
    assert config['encoder_depth'] == 1 and config['comorbid_block'] == 0
    assert config['edge_direction'] == 'forward'
    assert config['hidden'] == HIDDEN
    assert config['seed'] == 1234
    assert config['pair_mode'] == 'additive'
    assert config['common_init_identical_to_c'] is False   # hidden 8 != the control's 128
    assert config['effective_settings']['arm'] == 'B'
    assert config['architecture']['layers'] == 1
    assert config['adaptation_version'] == plugin.EvidenceAdapterV3.adaptation_version
    assert config['native_defaults'] == plugin.EvidenceAdapterV3.runner_defaults
    wide = _adapter(tmp_path, 'C', hidden=plugin.CONTROL_HIDDEN, state_path=path)
    assert wide.run_config()['common_init_identical_to_c'] is True
    bidirectional = _adapter(tmp_path, 'C', state_path=path, edge_direction='bidirectional')
    assert bidirectional.run_config()['edge_direction'] == 'bidirectional'
    assert bidirectional.num_relations > adapter.num_relations


def test_adapter_arm_a_loads_a_v2_adapter_state_dict_and_reproduces_its_output(tmp_path):
    from torch_geometric.data import Batch
    from comparison.standardized.clinical_graph_v2.methods import METHOD_REGISTRY

    torch.manual_seed(2)
    v2_adapter = METHOD_REGISTRY['cei_gnn_v2'](
        num_tokens=NUM_TOKENS, node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_classes=CLASSES,
        hidden=HIDDEN, layers=1, dropout=0.0, token_dim=TOKEN_DIM, num_triples=TRIPLES,
        args=Namespace(method_options={'pair_mode': 'additive'})).eval()  # pair_rank 16 = v3's
    _randomise_gates(v2_adapter.network)
    arm_a = _adapter(tmp_path, 'A').eval()
    missing, unexpected = arm_a.load_state_dict(v2_adapter.state_dict(), strict=False)
    assert set(unexpected) == set()
    assert set(missing) == {f'network.{name}' for name in NEW_PARAMETERS}
    batch = Batch.from_data_list([_graph(), _graph(seed=8)])
    expected = v2_adapter(batch, epoch=0)
    actual = arm_a(batch, epoch=0)
    assert torch.equal(actual.logits, expected.logits)
    assert actual.diagnostics['pairs_per_graph'] == expected.diagnostics['pairs_per_graph']
    assert actual.diagnostics['absent_items_per_graph'] == 0.0
    arm_c = _adapter(tmp_path, 'C').eval()
    arm_c.load_state_dict(v2_adapter.state_dict(), strict=False)
    with torch.no_grad():
        arm_c.network.absence_vote.normal_()
    output_c = arm_c(batch, epoch=0)
    assert output_c.diagnostics['absent_items_per_graph'] == 1.0
    assert not torch.allclose(output_c.logits, expected.logits)
    assert torch.equal(arm_c.forward_continuous(arm_c.continuous_inputs(batch), batch.edge_index,
                                                batch), output_c.logits)
