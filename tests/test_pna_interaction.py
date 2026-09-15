"""Executable interaction-PNA acceptance probes (synthetic, no patient IDs)."""
import importlib.util
import pytest
import torch
from torch_geometric.data import Batch
from torch_geometric.explain.algorithm.utils import set_masks, clear_masks
from torch_geometric.nn import PNAConv, MessagePassing


def test_real_edge_masks_gate_context_once_and_primary_once():
    from pna_analysis.model import InteractionPNAConv
    torch.manual_seed(23)
    layer = InteractionPNAConv(8, torch.tensor([1, 8, 2]), rank=3)
    x = torch.randn(3, 8)
    edges = torch.tensor([[0, 1, 2, 2], [2, 2, 0, 1]])
    types = torch.tensor([0, 1, 2])
    seen = []
    handle = layer.register_message_forward_hook(lambda m, i, o: seen.append(o.detach().clone()))
    baseline = layer(x, edges, node_type=types)
    set_masks(layer, torch.ones(4), edges, apply_sigmoid=False)
    assert torch.equal(layer(x, edges, node_type=types), baseline)
    # Kill complaint->hub: medication content cannot see its features indirectly.
    set_masks(layer, torch.tensor([1., 0., 1., 1.]), edges, apply_sigmoid=False)
    layer(x, edges, node_type=types)
    masked_message = seen[-1][0].clone()
    changed = x.clone(); changed[1] += 20
    layer(changed, edges, node_type=types)
    assert torch.equal(masked_message, seen[-1][0]), 'hidden pair context bypasses edge mask'
    mask = torch.tensor([.5, 1., 1., 1.], requires_grad=True)
    set_masks(layer, mask, edges, apply_sigmoid=False)
    out = layer(x, edges, node_type=types, cross_pairs=False)
    partial = seen[-1].clone()
    reference = PNAConv(8, 8, ['mean', 'min', 'max', 'std'],
                        ['identity', 'amplification', 'attenuation'], torch.tensor([1, 8, 2]))
    reference.load_state_dict({k: v for k, v in layer.state_dict().items() if not k.startswith('pair.')})
    set_masks(reference, mask, edges, apply_sigmoid=False)
    assert torch.equal(out, reference(x, edges)), 'PyG must not apply a second final mask'
    clear_masks(reference)
    clear_masks(layer)
    layer(x, edges, node_type=types, cross_pairs=False)
    assert torch.equal(partial[0], seen[-1][0] * .5), 'primary edge must be masked once'
    gradient = torch.autograd.grad(out.square().sum(), mask)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    set_masks(layer, torch.zeros(4), edges, apply_sigmoid=False)
    zero = layer(x, edges, node_type=types)[2]
    assert torch.equal(zero, layer(changed, edges, node_type=types)[2])
    assert torch.count_nonzero(seen[-1]) == 0
    clear_masks(layer); handle.remove()


def test_classifier_equivariance_fallback_and_no_hidden_access():
    import pna_analysis.model as module
    assert hasattr(module, 'PNAPredictor'), 'end-to-end predictor missing'
    from pna_analysis.data import concept_graph
    names = ['med:a', 'med:b', 'cc:c', 'cc:d']
    torch.manual_seed(31)
    model = module.PNAPredictor(5, 6, torch.tensor([1, 8, 2, 1]), width=8, layers=2, rank=3).eval()
    pair = concept_graph([0, 2], 1, names)
    mixed = Batch.from_data_list([pair, concept_graph([1, 3], 2, names), concept_graph([], 0, names)])
    single = Batch.from_data_list([pair])
    out = model(mixed)
    assert torch.allclose(out[:1], model(single), atol=1e-6)
    # Unrelated patient's contents cannot alter the first patient's representation.
    changed = mixed.clone(); changed.x[3:5] *= -11
    assert torch.equal(out[0], model(changed)[0])
    perm = torch.randperm(mixed.num_nodes); inverse = torch.argsort(perm)
    shuffled = mixed.clone()
    shuffled.x = mixed.x[perm]; shuffled.node_type = mixed.node_type[perm]
    shuffled.batch = mixed.batch[perm]; shuffled.edge_index = inverse[mixed.edge_index][:, torch.randperm(mixed.num_edges)]
    assert torch.allclose(model(shuffled), out, atol=1e-6)
    for codes in ([], [0], [0, 1], [2, 3]):
        graph = Batch.from_data_list([concept_graph(codes, 0, names)])
        assert torch.equal(model(graph), model(graph, cross_pairs=False))
        model(graph).sum().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    allones = torch.ones(mixed.num_edges)
    assert torch.equal(model(mixed, edge_mask=allones), out)
    zeros = torch.zeros(mixed.num_edges)
    changed = mixed.clone(); changed.x[changed.node_type != 2] = torch.randn_like(changed.x[changed.node_type != 2]) * 30
    assert torch.equal(model(mixed, edge_mask=zeros), model(changed, edge_mask=zeros))
    assert torch.equal(model(mixed, no_messages=True), model(mixed, edge_mask=zeros))
    features = mixed.x.clone().requires_grad_()
    probe = mixed.clone(); probe.x = features
    model(probe).square().sum().backward()
    assert features.grad[mixed.node_type != 2].abs().sum() > 0
    # Metadata does not come from argmax: perturbing x leaves node_type untouched.
    assert torch.equal(probe.node_type, mixed.node_type)
    from comparison.standardized.common_input_improvement import pyg_graph
    expected = pyg_graph([0, 2], 1, 4)
    assert torch.equal(pair.x, expected.x) and torch.equal(pair.edge_index, expected.edge_index)
    invalid = mixed.clone(); invalid.edge_index[1, 0] = 5
    with pytest.raises(ValueError, match='cross-graph'):
        model(invalid)



def test_graphxai_real_algorithms_use_stable_metadata():
    import pna_analysis.model as module
    assert hasattr(module, 'GraphXAIWrapper'), 'GraphXAI predictor wrapper missing'
    from pna_analysis.data import concept_graph
    from shared.lib.graphxai_standardized import explain_algorithms
    torch.manual_seed(42)
    model = module.PNAPredictor(5, 6, torch.tensor([1, 8, 2]), width=8, layers=2, rank=3).eval()
    for codes in ([0, 2], []):
        graph = Batch.from_data_list([concept_graph(codes, 0, ['med:a', 'med:b', 'cc:c', 'cc:d'])])
        wrapper = module.GraphXAIWrapper(model, graph.node_type)
        assert torch.equal(wrapper(graph.x, graph.edge_index, batch=graph.batch), model(graph))
        result = explain_algorithms(wrapper, graph.x, graph.edge_index, batch=graph.batch, steps=4, epochs=3)
        assert set(result) == {'GradExplainer', 'IntegratedGradExplainer', 'GNNExplainer'}
        for value in result.values():
            assert value['status'] == 'success'
            assert torch.isfinite(torch.tensor(value['node_explanation']['node_importance'])).all()
        assert result['GNNExplainer']['provenance']['edge_gradient_verified'] == bool(codes)
        assert all(not conv.explain for conv in model.convs)


def test_pair_changes_incoming_content_before_standard_pna():
    assert importlib.util.find_spec('pna_analysis') is not None, 'PNA implementation missing'
    from pna_analysis.model import InteractionPNAConv
    torch.manual_seed(17)
    layer = InteractionPNAConv(8, torch.tensor([1, 8, 2]), rank=3)
    assert isinstance(layer, MessagePassing)
    x = torch.randn(3, 8)
    edges = torch.tensor([[0, 1, 2, 2], [2, 2, 0, 1]])
    types = torch.tensor([0, 1, 2])
    messages = []
    handle = layer.register_message_forward_hook(lambda module, inputs, output: messages.append(output.detach().clone()))
    layer(x, edges, node_type=types)
    changed = x.clone(); changed[1] += 1.3
    layer(changed, edges, node_type=types)
    # Medication and hub fixed: its incoming message changes with this complaint.
    assert not torch.allclose(messages[0][0], messages[1][0])
    layer(x, edges, node_type=types, cross_pairs=False)
    layer(changed, edges, node_type=types, cross_pairs=False)
    assert torch.equal(messages[2][0], messages[3][0])
    handle.remove()
    reference = PNAConv(8, 8, ['mean', 'min', 'max', 'std'],
                        ['identity', 'amplification', 'attenuation'], torch.tensor([1, 8, 2]))
    reference.load_state_dict({k: v for k, v in layer.state_dict().items() if not k.startswith('pair.')})
    assert torch.equal(layer(x, edges, node_type=types, cross_pairs=False), reference(x, edges))
    out = layer(x, edges, node_type=types)
    out.square().sum().backward()
    grads = [p.grad for p in layer.pair.parameters()]
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    before = [p.detach().clone() for p in layer.pair.parameters()]
    torch.optim.Adam(layer.parameters(), lr=.01).step()
    assert all(not torch.equal(a, b) for a, b in zip(before, layer.pair.parameters()))
