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
    crossing[1, -1] = 0   # the last comorbid record of graph 2 now ends in graph 0
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
