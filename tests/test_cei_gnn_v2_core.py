"""Synthetic tests for the CEI-GNN v2 core: pair builder and pair-evidence network."""
import importlib
import importlib.util

import pytest
import torch

CORE = "comparison.standardized.clinical_graph_v2.methods.cei_gnn_v2"
EXPECTED_PAIRS = [(2, 3), (2, 4), (2, 6), (2, 7), (3, 4), (3, 6), (3, 7), (6, 7)]


def _v2():
    assert importlib.util.find_spec(CORE) is not None, "CEI-GNN v2 core module is missing"
    return importlib.import_module(CORE)


def _graph(scale=1.0, seed=7):
    """Two visits. Node 2 (complaint) and node 3 (measurement) belong to both."""
    from comparison.standardized.clinical_graph_v2.tensorize import ClinicalGraphData

    generator = torch.Generator().manual_seed(seed)
    graph = ClinicalGraphData(
        x=torch.randn(8, 3, generator=generator) * scale,
        edge_index=torch.tensor([[0, 1, 1, 1, 3, 6], [1, 2, 3, 4, 5, 5]]),
        edge_attr=torch.randn(6, 2, generator=generator))
    # kinds: 0 patient, 1 visit, 2 complaint, 3 measurement, 5 vital, 4 analyte
    graph.node_type = torch.tensor([0, 1, 2, 3, 5, 4, 3, 2])
    graph.token = torch.tensor([1, 2, 3, 4, 5, 6, 4, 3])
    graph.edge_relation = torch.tensor([0, 2, 3, 5, 4, 4])
    graph.edge_triple = torch.tensor([0, 1, 2, 3, 1, 1])
    graph.visit_membership_index = torch.tensor(
        [[0, 0, 0, 0, 0, 1, 1, 1, 1], [1, 2, 3, 4, 5, 2, 6, 7, 3]])
    graph.num_visits = torch.tensor([2])
    graph.y = torch.tensor([1])
    return graph


def _pairs_of(graph):
    v2 = _v2()
    pairs = v2.within_visit_pairs(graph.visit_membership_index, graph.node_type,
                                  int(graph.num_nodes))
    return [tuple(pair) for pair in pairs.t().tolist()]


def test_pairs_are_unique_within_visit_evidence_pairs():
    assert _pairs_of(_graph()) == EXPECTED_PAIRS


def test_pairs_follow_pyg_batch_offsets_and_never_cross_graphs():
    from torch_geometric.data import Batch

    first, second = _graph(), _graph(seed=8)
    batch = Batch.from_data_list([first, second])
    pairs = _pairs_of(batch)
    assert pairs == EXPECTED_PAIRS + [(i + 8, j + 8) for i, j in EXPECTED_PAIRS]
    left, right = torch.tensor(pairs).t()
    assert torch.equal(batch.batch[left], batch.batch[right])


def test_pairs_are_empty_with_fewer_than_two_evidence_nodes_per_visit():
    v2 = _v2()
    membership = torch.tensor([[0, 0, 1], [0, 1, 2]])
    node_type = torch.tensor([1, 2, 5])
    pairs = v2.within_visit_pairs(membership, node_type, 3)
    assert pairs.shape == (2, 0) and pairs.dtype == torch.long
    empty = v2.within_visit_pairs(torch.zeros((2, 0), dtype=torch.long), node_type, 3)
    assert empty.shape == (2, 0)


def test_pair_builder_rejects_malformed_membership():
    v2 = _v2()
    node_type = torch.tensor([2, 3])
    with pytest.raises(ValueError, match="shape"):
        v2.within_visit_pairs(torch.tensor([0, 1]), node_type, 2)
    with pytest.raises(ValueError, match="outside"):
        v2.within_visit_pairs(torch.tensor([[0, 0], [0, 5]]), node_type, 2)


def test_kind_pair_index_is_symmetric_and_covers_six_unordered_pairs():
    v2 = _v2()
    assert v2.KIND_PAIR_COUNT == 6
    node_type = torch.tensor([2, 3, 5])
    seen = {}
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            index = v2.kind_pair_index(node_type, torch.tensor([[i], [j]])).item()
            seen.setdefault(frozenset((i, j)), set()).add(index)
    assert all(len(values) == 1 for values in seen.values())
    same_kind = torch.tensor([2, 2, 3, 3, 5, 5])
    diagonal = v2.kind_pair_index(same_kind, torch.tensor([[0, 2, 4], [1, 3, 5]])).tolist()
    mixed = {next(iter(values)) for values in seen.values()}
    assert sorted(mixed | set(diagonal)) == [0, 1, 2, 3, 4, 5]
    with pytest.raises(ValueError, match="evidence"):
        v2.kind_pair_index(torch.tensor([1, 2]), torch.tensor([[0], [1]]))
