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
