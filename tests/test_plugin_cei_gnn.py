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
