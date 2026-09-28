"""Synthetic tests for the CEI-GNN v2 core: pair builder and pair-evidence network."""
import importlib
import importlib.util
import json
import subprocess
import sys
import textwrap
import time
from itertools import combinations
from pathlib import Path

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


def test_adapter_rejects_cross_graph_and_out_of_range_visits_even_when_off():
    from torch_geometric.data import Batch
    from comparison.standardized.clinical_graph_v2.methods import METHOD_REGISTRY
    from argparse import Namespace

    adapter = METHOD_REGISTRY["cei_gnn_v2"](
        num_tokens=8, node_dim=3, edge_dim=2, num_classes=3, hidden=8, layers=1,
        dropout=0.0, token_dim=4, num_triples=4,
        args=Namespace(method_options={"pair_mode": "off"})).eval()
    batch = Batch.from_data_list([_graph(), _graph(seed=8)])
    batch.visit_membership_index = torch.tensor([[0, 0], [0, 8]])
    with pytest.raises(ValueError, match="crosses graph"):
        adapter(batch, epoch=0)
    batch.visit_membership_index = torch.tensor([[4, 4], [8, 9]])
    with pytest.raises(ValueError, match="outside"):
        adapter(batch, epoch=0)


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


def _reference_pairs(membership, node_type):
    from comparison.standardized.clinical_graph_v2.methods.cei_gnn_v2 import EVIDENCE_KIND_IDS

    by_visit = {}
    for visit, node in membership.t().tolist():
        if int(node_type[node]) in EVIDENCE_KIND_IDS:
            by_visit.setdefault(visit, set()).add(node)
    return sorted({pair for nodes in by_visit.values() for pair in combinations(sorted(nodes), 2)})


def test_pair_builder_matches_brute_force_for_random_duplicate_memberships():
    v2 = _v2()
    generator = torch.Generator().manual_seed(90210)
    for _ in range(50):
        node_count = int(torch.randint(4, 30, (), generator=generator))
        visit_count = int(torch.randint(2, 8, (), generator=generator))
        node_type = torch.randint(0, 6, (node_count,), generator=generator)
        membership = torch.stack((
            torch.randint(0, visit_count, (node_count * 2,), generator=generator),
            torch.randint(0, node_count, (node_count * 2,), generator=generator)))
        membership = torch.cat((membership, membership[:, :5]), dim=1)
        expected = _reference_pairs(membership, node_type)
        actual = v2.within_visit_pairs(membership, node_type, node_count).t().tolist()
        assert actual == [list(pair) for pair in expected]


def test_skewed_visit_pair_enumeration_is_bounded_and_exact():
    script = textwrap.dedent(r"""
        import json, resource, time
        import torch
        from comparison.standardized.clinical_graph_v2.methods.cei_gnn_v2 import within_visit_pairs
        import itertools
        wide, singles = 1000, 20000
        node_count = wide + singles
        membership = torch.empty((2, wide + singles), dtype=torch.long)
        membership[0, :wide] = 0
        membership[1, :wide] = torch.arange(wide)
        membership[0, wide:] = torch.arange(1, singles + 1)
        membership[1, wide:] = torch.arange(wide, node_count)
        node_type = torch.full((node_count,), 2, dtype=torch.long)
        start = time.perf_counter()
        pairs = within_visit_pairs(membership, node_type, node_count)
        elapsed = time.perf_counter() - start
        expected = torch.tensor(list(itertools.combinations(range(wide), 2)), dtype=torch.long).t()
        assert pairs.shape == (2, 499500)
        assert torch.equal(pairs.cpu(), expected)
        print(json.dumps({"seconds": elapsed, "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))
    """)
    started = time.perf_counter()
    try:
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                                timeout=8, cwd=Path(__file__).resolve().parents[1])
    except subprocess.TimeoutExpired:
        assert False, "skewed pair enumeration exceeded the 8-second safety timeout"
    assert result.returncode == 0, result.stderr or result.stdout
    measurements = json.loads(result.stdout.strip().splitlines()[-1])
    assert measurements["seconds"] < 5
    assert time.perf_counter() - started < 10


def _network(mode="product", seed=11):
    v2 = _v2()
    assert hasattr(v2, "PairEvidenceNetwork"), "PairEvidenceNetwork is missing"
    torch.manual_seed(seed)
    network = v2.PairEvidenceNetwork(
        num_tokens=8, node_dim=3, edge_dim=2, num_classes=3, hidden=8, token_dim=4,
        num_triples=4, num_relations=15, dropout=0.0, pair_rank=4, pair_mode=mode,
        num_node_types=8).eval()
    with torch.no_grad():
        network.pair_gate.normal_()
        network.edge_context_gate.bias.normal_()
    return network


def _metadata(graph):
    from comparison.standardized.clinical_graph_v2.methods.base import read_clinical_batch

    return read_clinical_batch(graph, method="test", node_dim=3, edge_dim=2, num_tokens=8,
                               num_triples=4, num_relations=15)


def _run(network, graph, features=None, **kwargs):
    metadata = _metadata(graph)
    if features is None:
        features = network.continuous_inputs(metadata)
    return network.forward_continuous(features, metadata.edge_index, metadata,
                                      graph.visit_membership_index, **kwargs)


def _block_mixed_difference(network, graph, block, first, second):
    metadata = _metadata(graph)
    base = network.continuous_inputs(metadata).detach()
    generator = torch.Generator().manual_seed(3)
    delta_first = torch.randn(base.size(1), generator=generator) * 0.5
    delta_second = torch.randn(base.size(1), generator=generator) * 0.5
    zero = torch.zeros_like(delta_first)

    def total(shift_first, shift_second):
        features = base.clone()
        features[first] += shift_first
        features[second] += shift_second
        parts = network.forward_continuous(features, metadata.edge_index, metadata,
                                           graph.visit_membership_index, return_parts=True)
        return parts[block].sum(0)

    return (total(delta_first, delta_second) - total(delta_first, zero)
            - total(zero, delta_second) + total(zero, zero))


@pytest.mark.parametrize("mode", ["product", "additive", "off"])
def test_parts_reconstruct_logits_in_every_mode(mode):
    parts = _run(_network(mode), _graph(), return_parts=True)
    rebuilt = (parts["bias"] + parts["node_contributions"].sum(0)
               + parts["edge_contributions"].sum(0) + parts["pair_contributions"].sum(0))
    torch.testing.assert_close(parts["logits"][0], rebuilt, rtol=1e-5, atol=1e-5)
    assert parts["pairs"].t().tolist() == [list(pair) for pair in EXPECTED_PAIRS]
    assert parts["pair_contributions"].shape == (8, 3)
    if mode == "off":
        assert torch.count_nonzero(parts["pair_contributions"]) == 0
    else:
        assert parts["pair_contributions"].abs().sum() > 0


def test_modes_share_parameter_names_and_shapes_and_change_predictions():
    networks = {mode: _network(mode) for mode in ("product", "additive", "off")}
    shapes = {mode: [(name, tuple(p.shape)) for name, p in net.named_parameters()]
              for mode, net in networks.items()}
    assert shapes["product"] == shapes["additive"] == shapes["off"]
    state = networks["product"].state_dict()
    for net in networks.values():
        net.load_state_dict(state)
    logits = {mode: _run(net, _graph()) for mode, net in networks.items()}
    assert not torch.allclose(logits["product"], logits["additive"])
    assert not torch.allclose(logits["product"], logits["off"])
    assert not torch.allclose(logits["additive"], logits["off"])


def test_only_product_mode_has_a_pair_cross_term():
    product = _block_mixed_difference(_network("product"), _graph(), "pair_contributions", 2, 3)
    additive = _block_mixed_difference(_network("additive"), _graph(), "pair_contributions", 2, 3)
    assert product.abs().max() > 1e-4, "product pair term collapsed to an additive response"
    assert additive.abs().max() < 1e-5, "additive control contains a hidden cross term"


def test_edge_block_has_no_endpoint_cross_term():
    for mode in ("product", "additive"):
        difference = _block_mixed_difference(_network(mode), _graph(), "edge_contributions", 1, 2)
        assert difference.abs().max() < 1e-5, f"edge endpoints interact in {mode} mode"


def test_zero_edge_mask_removes_only_the_edge_block():
    network = _network("product")
    graph = _graph()
    ordinary = _run(network, graph, return_parts=True)
    network.set_edge_mask(torch.zeros(graph.num_edges))
    masked = _run(network, graph, return_parts=True)
    network.set_edge_mask(None)
    assert torch.count_nonzero(masked["edge_contributions"]) == 0
    expected = (masked["bias"] + ordinary["node_contributions"].sum(0)
                + ordinary["pair_contributions"].sum(0))
    torch.testing.assert_close(masked["logits"][0], expected, rtol=1e-5, atol=1e-5)


def test_pyg_mask_matches_direct_probability_mask_and_carries_gradient():
    from torch_geometric.explain.algorithm.utils import clear_masks, set_masks

    network = _network("product")
    graph = _graph()
    metadata = _metadata(graph)
    raw = torch.nn.Parameter(torch.tensor([-0.9, 0.3, 1.1, -0.2, 0.5, 0.8]))
    set_masks(network, raw, metadata.edge_index, apply_sigmoid=True)
    pyg = _run(network, graph)
    (pyg * torch.tensor([[0.2, -0.7, 1.1]])).sum().backward()
    assert raw.grad is not None and raw.grad.abs().sum() > 0
    clear_masks(network)
    network.set_edge_mask(raw.detach().sigmoid())
    direct = _run(network, graph)
    network.set_edge_mask(None)
    torch.testing.assert_close(pyg.detach(), direct, rtol=1e-6, atol=1e-6)


def test_edgeless_pairless_graph_is_finite():
    from comparison.standardized.clinical_graph_v2.tensorize import ClinicalGraphData

    graph = ClinicalGraphData(x=torch.ones((2, 3)), edge_index=torch.zeros((2, 0), dtype=torch.long),
                              edge_attr=torch.zeros((0, 2)))
    graph.node_type = torch.tensor([1, 2])
    graph.token = torch.tensor([1, 2])
    graph.edge_relation = torch.zeros(0, dtype=torch.long)
    graph.edge_triple = torch.zeros(0, dtype=torch.long)
    graph.visit_membership_index = torch.tensor([[0, 0], [0, 1]])
    graph.num_visits = torch.tensor([1])
    for mode in ("product", "additive", "off"):
        parts = _run(_network(mode), graph, return_parts=True)
        assert parts["logits"].shape == (1, 3) and torch.isfinite(parts["logits"]).all()
        assert parts["pairs"].shape == (2, 0)


def test_network_fails_closed_on_invalid_inputs():
    v2 = _v2()
    network = _network("product")
    graph = _graph()
    metadata = _metadata(graph)
    features = network.continuous_inputs(metadata).detach()
    features[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        network.forward_continuous(features, metadata.edge_index, metadata,
                                   graph.visit_membership_index)
    with pytest.raises(ValueError, match="edge mask"):
        network.set_edge_mask(torch.full((graph.num_edges,), 1.5))
    with pytest.raises(ValueError, match="pair_mode"):
        v2.PairEvidenceNetwork(num_tokens=8, node_dim=3, edge_dim=2, num_classes=3, hidden=8,
                               token_dim=4, num_triples=4, num_relations=15, dropout=0.0,
                               pair_rank=4, pair_mode="both", num_node_types=8)
