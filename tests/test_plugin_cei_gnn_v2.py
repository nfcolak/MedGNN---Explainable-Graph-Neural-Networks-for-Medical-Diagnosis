"""Synthetic adapter tests for the CEI-GNN v2 plugin."""
from argparse import Namespace

import pytest
import torch

from tests.test_cei_gnn_v2_core import EXPECTED_PAIRS, _graph


def _registry():
    from comparison.standardized.clinical_graph_v2.methods import METHOD_REGISTRY

    assert "cei_gnn_v2" in METHOD_REGISTRY, "cei_gnn_v2 plugin is not registered"
    return METHOD_REGISTRY


def _adapter(mode="product", rank=4, seed=11, **options):
    torch.manual_seed(seed)
    return _registry()["cei_gnn_v2"](
        num_tokens=8, node_dim=3, edge_dim=2, num_classes=3, hidden=8, layers=1,
        dropout=0.0, token_dim=4, num_triples=4,
        args=Namespace(method_options={"pair_rank": rank, "pair_mode": mode, **options})).eval()


def test_plugin_uses_v1_runner_defaults_and_parses_runner_options():
    from comparison.standardized.clinical_graph_v2 import train

    registry = _registry()
    assert train.plugin_defaults("cei_gnn_v2") == registry["cei_gnn"].runner_defaults
    parser = train.parser()
    args = train.normalize_method_args(parser.parse_args([
        "--artifact", "a", "--targets", "t", "--output", "o", "--method", "cei_gnn_v2",
        "--method-option", "pair_mode=additive"]), parser)
    adapter = registry["cei_gnn_v2"](
        num_tokens=8, node_dim=3, edge_dim=2, num_classes=3, hidden=args.hidden,
        layers=args.layers, dropout=args.dropout, token_dim=args.token_dim, num_triples=4,
        args=args)
    assert (adapter.pair_mode, adapter.pair_rank, adapter.grad_clip_value) == ("additive", 16, 2.0)
    assert adapter.run_config()["effective_settings"] == {"pair_rank": 16, "pair_mode": "additive"}


def test_constructor_rejects_bad_mode_unknown_option_and_depth():
    cls = _registry()["cei_gnn_v2"]
    common = dict(num_tokens=8, node_dim=3, edge_dim=2, num_classes=3, hidden=8,
                  dropout=0.0, token_dim=4, num_triples=4)
    with pytest.raises(ValueError, match="pair_mode"):
        cls(**common, layers=1, args=Namespace(method_options={"pair_mode": "both"}))
    with pytest.raises(ValueError, match="unknown method option"):
        cls(**common, layers=1, args=Namespace(method_options={"use_interactions": "true"}))
    with pytest.raises(ValueError, match="layers=1"):
        cls(**common, layers=2, args=Namespace(method_options={}))


def test_run_config_reports_inactive_pair_parameters_only_when_off():
    product, off = _adapter("product"), _adapter("off")
    total = sum(p.numel() for p in product.parameters())
    assert product.run_config()["architecture"]["parameter_count"] == total
    assert product.run_config()["architecture"]["inactive_parameter_count"] == 0
    net = off.network
    inactive = (net.pair_projection.weight.numel() + net.pair_vote.weight.numel()
                + net.pair_vote.bias.numel() + net.pair_gate.numel())
    architecture = off.run_config()["architecture"]
    assert architecture["parameter_count"] == total
    assert architecture["inactive_parameter_count"] == inactive
    assert architecture["active_parameter_count"] == total - inactive


def test_forward_equals_continuous_core_and_graphxai_wrapper():
    from comparison.standardized.clinical_graph_v2.cei_graphxai import ClinicalGraphXAIWrapper

    adapter, graph = _adapter("product"), _graph()
    ordinary = adapter(graph, epoch=0).logits
    features = adapter.continuous_inputs(graph)
    continuous = adapter.forward_continuous(features, graph.edge_index, graph)
    torch.testing.assert_close(ordinary, continuous, rtol=0, atol=0)
    wrapped = ClinicalGraphXAIWrapper(adapter, graph)(features, graph.edge_index)
    torch.testing.assert_close(ordinary, wrapped, rtol=0, atol=0)
    parts = adapter.forward_continuous(features, graph.edge_index, graph, return_parts=True)
    assert parts["pairs"].t().tolist() == [list(pair) for pair in EXPECTED_PAIRS]


def test_batched_logits_equal_individual_graphs():
    from torch_geometric.data import Batch

    adapter = _adapter("product")
    first, second = _graph(), _graph(scale=-0.4, seed=8)
    batched = adapter(Batch.from_data_list([first, second]), epoch=0).logits
    torch.testing.assert_close(batched[0], adapter(first, epoch=0).logits[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(batched[1], adapter(second, epoch=0).logits[0], rtol=1e-5, atol=1e-5)


def test_missing_membership_and_float_indices_fail_closed():
    adapter, graph = _adapter("off"), _graph()
    missing = graph.clone()
    del missing.visit_membership_index
    with pytest.raises(ValueError, match="visit_membership_index"):
        adapter(missing, epoch=0)
    floating = graph.clone()
    floating.visit_membership_index = floating.visit_membership_index.float()
    with pytest.raises(ValueError, match="integer"):
        adapter(floating, epoch=0)
    missing_counts = graph.clone()
    del missing_counts.num_visits
    with pytest.raises(ValueError, match="num_visits"):
        adapter(missing_counts, epoch=0)
    floating_counts = graph.clone()
    floating_counts.num_visits = floating_counts.num_visits.float()
    with pytest.raises(ValueError, match="integer"):
        adapter(floating_counts, epoch=0)


def test_gradients_reach_active_parameters_only():
    weights = torch.tensor([[0.2, -0.7, 1.1]])
    for mode in ("product", "additive", "off"):
        adapter = _adapter(mode)
        with torch.no_grad():
            adapter.network.pair_gate.normal_()
        (adapter(_graph(), epoch=0).logits * weights).sum().backward()
        for name, parameter in adapter.network.named_parameters():
            if mode == "off" and name.startswith("pair_"):
                assert parameter.grad is None, f"off mode trains inactive {name}"
            else:
                assert parameter.grad is not None, f"disconnected {name} in {mode}"
                assert torch.isfinite(parameter.grad).all()
