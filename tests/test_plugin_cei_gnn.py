def _fixture():
    import torch
    from torch_geometric.data import Data
    from comparison.standardized.clinical_graph_v2.methods import METHOD_REGISTRY

    adapter = METHOD_REGISTRY["cei_gnn"](
        num_tokens=8, node_dim=3, edge_dim=2, num_classes=3, hidden=8,
        layers=1, dropout=0.0, token_dim=4, num_triples=3, args={})
    graph = Data(
        x=torch.tensor([[0.2, -0.3, 0.5], [0.8, 0.1, -0.4], [-0.1, 0.7, 0.2]]),
        token=torch.tensor([1, 2, 3]), node_type=torch.tensor([0, 1, 2]),
        edge_index=torch.tensor([[0, 1, 0], [1, 2, 1]]),
        edge_attr=torch.tensor([[0.2, 0.1], [0.4, -0.3], [-0.2, 0.6]]),
        edge_relation=torch.tensor([0, 1, 0]), edge_triple=torch.tensor([0, 1, 2]))
    return adapter.eval(), graph


def test_cei_is_independent_registered_method():
    from comparison.standardized.clinical_graph_v2.methods import METHOD_REGISTRY

    assert "cei_gnn" in METHOD_REGISTRY, "independent evidence-interaction method absent"


def test_continuous_core_is_exact_and_uses_supplied_channels_not_metadata_x():
    import torch

    adapter, graph = _fixture()
    features = adapter.continuous_inputs(graph)
    ordinary = adapter(graph, epoch=0).logits
    continuous = adapter.forward_continuous(features, graph.edge_index, graph)
    torch.testing.assert_close(ordinary, continuous, rtol=0, atol=0)

    altered = features.clone()
    altered[:, :adapter.node_dim] += 0.7
    supplied_changed = adapter.forward_continuous(altered, graph.edge_index,
                                                   adapter._read(graph))
    assert not torch.allclose(ordinary, supplied_changed), "supplied numeric channels were ignored"

    metadata_changed = graph.clone()
    metadata_changed.x = metadata_changed.x + 100
    held_features = adapter.forward_continuous(features, graph.edge_index,
                                               adapter._read(metadata_changed))
    torch.testing.assert_close(ordinary, held_features, rtol=0, atol=0)

    embedded_changed = features.clone()
    embedded_changed[:, adapter.node_dim:adapter.node_dim + adapter.token_dim] += 0.6
    embedded_logits = adapter.forward_continuous(embedded_changed, graph.edge_index,
                                                  adapter._read(graph))
    assert not torch.allclose(ordinary, embedded_logits), "supplied embedding channels were ignored"


def test_edge_mask_is_applied_once_to_numerator_and_denominator_and_parts_reconstruct():
    import torch
    import torch.nn.functional as F

    adapter, graph = _fixture()
    features = adapter.continuous_inputs(graph)
    net = adapter.network
    src, dst = graph.edge_index
    h = F.gelu(net.node_norm(net.node_encoder(features)))
    node_vote, node_gate_logits = net.node_head(h).chunk(2, dim=-1)
    node_gate = node_gate_logits.sigmoid()
    context = (net.relation_embedding(graph.edge_relation)
               + net.triple_embedding(graph.edge_triple)
               + net.edge_feature_projection(graph.edge_attr))
    q = (net.endpoint_source(h[src]).tanh() * net.endpoint_target(h[dst]).tanh()
         * net.interaction_context(context).sigmoid())
    edge_input = torch.cat((h[src], h[dst], context, q), dim=-1)
    edge_vote, edge_gate_logits = net.edge_head(edge_input).chunk(2, dim=-1)
    edge_gate = edge_gate_logits.sigmoid()

    mask = torch.tensor([0.2, 0.6, 0.9])
    node_term = (node_gate * node_vote).sum(0) / (1 + node_gate.sum(0))

    def explicit(mask_value):
        numerator = (mask_value[:, None] * edge_gate * edge_vote).sum(0)
        denominator = 1 + (mask_value[:, None] * edge_gate).sum(0)
        return net.bias + node_term + numerator / denominator

    adapter.network.set_edge_mask(mask)
    parts = adapter.forward_continuous(features, graph.edge_index, graph, return_parts=True)
    torch.testing.assert_close(parts["logits"][0], explicit(mask), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(parts["logits"][0], parts["bias"]
                               + parts["node_contributions"].sum(0)
                               + parts["edge_contributions"].sum(0), rtol=1e-6, atol=1e-6)
    assert not torch.allclose(explicit(mask), explicit(mask.square())), "fractional mask cannot distinguish squared-mask mutant"

    adapter.network.set_edge_mask(torch.ones_like(mask))
    all_ones = adapter.forward_continuous(features, graph.edge_index, graph)
    adapter.network.set_edge_mask(None)
    ordinary = adapter.forward_continuous(features, graph.edge_index, graph)
    torch.testing.assert_close(all_ones, ordinary)
    adapter.network.set_edge_mask(torch.zeros_like(mask))
    zero_parts = adapter.forward_continuous(features, graph.edge_index, graph, return_parts=True)
    torch.testing.assert_close(zero_parts["logits"][0], net.bias + node_term)
    torch.testing.assert_close(zero_parts["edge_contributions"], torch.zeros_like(edge_vote))


def test_batch_permutation_parallel_edges_and_empty_edge_graphs_are_supported():
    import torch
    from torch_geometric.data import Batch, Data

    adapter, graph = _fixture()
    second = graph.clone()
    second.x = second.x * -0.4
    second.token = torch.tensor([4, 5, 6])
    batched = Batch.from_data_list([graph, second])
    batch_logits = adapter(batched, epoch=0).logits
    torch.testing.assert_close(batch_logits[0], adapter(graph, epoch=0).logits[0])
    torch.testing.assert_close(batch_logits[1], adapter(second, epoch=0).logits[0])

    node_order = torch.tensor([2, 0, 1])
    old_to_new = torch.empty_like(node_order)
    old_to_new[node_order] = torch.arange(node_order.numel())
    edge_order = torch.tensor([2, 1, 0])
    permuted = graph.clone()
    for key in ("x", "token", "node_type"):
        setattr(permuted, key, getattr(graph, key)[node_order])
    permuted.edge_index = old_to_new[graph.edge_index[:, edge_order]]
    for key in ("edge_attr", "edge_relation", "edge_triple"):
        setattr(permuted, key, getattr(graph, key)[edge_order])
    torch.testing.assert_close(adapter(permuted, epoch=0).logits,
                               adapter(graph, epoch=0).logits, rtol=1e-6, atol=1e-6)

    empty = Data(x=torch.ones((1, 3)), token=torch.tensor([2]), node_type=torch.tensor([0]),
                 edge_index=torch.empty((2, 0), dtype=torch.long),
                 edge_attr=torch.empty((0, 2)), edge_relation=torch.empty((0,), dtype=torch.long),
                 edge_triple=torch.empty((0,), dtype=torch.long))
    output = adapter(empty, epoch=0).logits
    assert output.shape == (1, 3) and torch.isfinite(output).all()


def test_all_active_predictive_blocks_receive_finite_gradients():
    import torch

    adapter, graph = _fixture()
    features = adapter.continuous_inputs(graph).detach().requires_grad_(True)
    # Unequal class weights avoid cancellation in the small synthetic graph.
    weights = torch.tensor([[0.2, -0.7, 1.1]])
    logits = adapter.forward_continuous(features, graph.edge_index, graph)
    (logits * weights).sum().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    network = adapter.network
    for name, parameter in network.named_parameters():
        if "endpoint_" in name or "interaction_context" in name or "edge_head" in name or "node_head" in name or "relation_embedding" in name or "triple_embedding" in name or "edge_feature_projection" in name:
            assert parameter.grad is not None, f"active parameter block disconnected: {name}"
            assert torch.isfinite(parameter.grad).all(), f"nonfinite gradient: {name}"
            assert parameter.grad.abs().sum() > 0, f"zero gradient: {name}"


def test_constructor_rejects_unsupported_depth_unknown_options_and_cross_graph_edges():
    import pytest
    import torch
    from torch_geometric.data import Batch
    from comparison.standardized.clinical_graph_v2.methods import METHOD_REGISTRY

    adapter_type = METHOD_REGISTRY["cei_gnn"]
    with pytest.raises(ValueError, match="layers=1"):
        adapter_type(num_tokens=8, node_dim=3, edge_dim=2, num_classes=3, hidden=8,
                     layers=2, dropout=0, token_dim=4, num_triples=3, args={})
    with pytest.raises(ValueError, match="unknown method option"):
        adapter_type(num_tokens=8, node_dim=3, edge_dim=2, num_classes=3, hidden=8,
                     layers=1, dropout=0, token_dim=4, num_triples=3,
                     args={"method_options": {"use_interactons": False}})

    adapter, graph = _fixture()
    batched = Batch.from_data_list([graph, graph.clone()])
    batched.edge_index[:, 0] = torch.tensor([0, graph.num_nodes])
    with pytest.raises(ValueError, match="cross-graph"):
        adapter(batched, epoch=0)


def test_product_off_is_same_capacity_control_and_interaction_changes_predictions():
    import torch
    from argparse import Namespace
    from comparison.standardized.clinical_graph_v2.methods import METHOD_REGISTRY

    _, graph = _fixture()
    cls = METHOD_REGISTRY["cei_gnn"]
    common = dict(num_tokens=8, node_dim=3, edge_dim=2, num_classes=3, hidden=8,
                  layers=1, dropout=0.0, token_dim=4, num_triples=3)
    enabled = cls(**common, args=Namespace(method_options={"interaction_rank": 4,
                                                           "use_interactions": True})).eval()
    disabled = cls(**common, args=Namespace(method_options={"interaction_rank": 4,
                                                            "use_interactions": False})).eval()
    disabled.load_state_dict(enabled.state_dict())
    assert sorted(tuple(p.shape) for p in enabled.parameters()) == sorted(
        tuple(p.shape) for p in disabled.parameters())
    assert sum(p.numel() for p in enabled.parameters()) == sum(
        p.numel() for p in disabled.parameters())
    assert enabled.run_config()["architecture"]["active_parameter_count"] == sum(
        p.numel() for p in enabled.parameters())
    assert disabled.run_config()["architecture"]["inactive_parameter_count"] > 0
    assert not torch.allclose(enabled(graph, epoch=0).logits,
                              disabled(graph, epoch=0).logits), "explicit q product had no predictive effect"


def test_invalid_continuous_inputs_masks_and_metadata_fail_closed():
    import pytest
    import torch

    adapter, graph = _fixture()
    features = adapter.continuous_inputs(graph)
    with pytest.raises(ValueError, match="finite"):
        invalid = features.clone()
        invalid[0, 0] = float("nan")
        adapter.forward_continuous(invalid, graph.edge_index, graph)
    with pytest.raises(ValueError, match="edge_index differs"):
        adapter.forward_continuous(features, graph.edge_index.roll(1, dims=1), graph)
    with pytest.raises(ValueError, match="edge mask length"):
        adapter.network.set_edge_mask(torch.ones(graph.num_edges + 1))
        adapter.forward_continuous(features, graph.edge_index, graph)
    adapter.network.set_edge_mask(None)
    for bad_mask in (torch.tensor([float("nan")] * graph.num_edges),
                     torch.full((graph.num_edges,), 1.1)):
        with pytest.raises(ValueError, match="edge mask"):
            adapter.network.set_edge_mask(bad_mask)
