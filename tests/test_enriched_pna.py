"""Enriched PNA wiring and actual GraphXAI synthetic tests."""
import importlib

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from pna_analysis.data import concept_graph
from pna_analysis.model import PNAPredictor, GraphXAIWrapper


def enriched_model():
    try:
        return importlib.import_module('comparison.standardized.enriched_input_v1.model')
    except ModuleNotFoundError:
        pytest.fail('Numerical PNA hub path missing')


@pytest.mark.parametrize('interactions', [False, True])
def test_numeric_hub_zero_baseline_logits_gradients_and_metadata(interactions):
    module = enriched_model()
    names = ['med:a', 'cc:b']; degree = torch.tensor([1, 4, 2])
    torch.manual_seed(1)
    base = PNAPredictor(3, 2, degree, width=8, interactions=interactions)
    torch.manual_seed(1)
    no_extra = module.EnrichedPNA(3, 2, degree, numeric_dim=0, width=8, interactions=interactions)
    graph = concept_graph([0, 1], 0, names)
    assert base.state_dict().keys() == no_extra.state_dict().keys()
    assert torch.equal(base(graph), no_extra(graph))
    torch.manual_seed(1)
    model = module.EnrichedPNA(3, 2, degree, numeric_dim=2, width=8, interactions=interactions)
    zero = module.numeric_graph([0, 1], 0, names, [0., 0.])
    rich = module.numeric_graph([0, 1], 0, names, [1., -2.])
    assert torch.equal(base(graph), model(zero))
    assert not torch.equal(model(zero), model(rich))
    assert torch.equal(rich.node_type, graph.node_type) and torch.equal(rich.edge_index, graph.edge_index)
    assert torch.equal(rich.x[:, :3], graph.x)
    model(rich).sum().backward()
    assert model.encoder.numeric.weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    # No edges removes concepts, not patient numeric state. This is legitimate.
    swapped = module.numeric_graph([0], 0, names, [1., -2.])
    torch.testing.assert_close(model(rich, no_messages=True), model(swapped, no_messages=True))
    assert not torch.equal(model(zero, no_messages=True), model(rich, no_messages=True))


def test_enriched_node_permutation_batch_isolation_hub_only():
    module = enriched_model(); names = ['med:a', 'cc:b']
    torch.manual_seed(2)
    model = module.EnrichedPNA(3, 2, torch.tensor([1, 4, 2]), numeric_dim=2, width=8).eval()
    graph = module.numeric_graph([0, 1], 0, names, [2., -1.])
    perm = torch.tensor([2, 0, 1]); inverse = torch.argsort(perm)
    moved = graph.clone(); moved.x = graph.x[perm]; moved.node_type = graph.node_type[perm]
    moved.edge_index = inverse[graph.edge_index]
    torch.testing.assert_close(model(graph), model(moved))
    hub = module.numeric_graph([], 0, names, [-4., 7.])
    batch = Batch.from_data_list([graph, hub])
    torch.testing.assert_close(model(batch), torch.cat([model(graph), model(hub)]), atol=1e-6, rtol=1e-5)
    assert torch.isfinite(model(hub)).all() and hub.num_nodes == 1
    masked = graph.clone(); masked.x[:, 3:] = 0
    assert torch.equal(masked.node_type, graph.node_type)


def test_real_graphxai_continuous_hub_feature_attribution():
    module = enriched_model()
    assert hasattr(module, 'explain_numeric'), 'Continuous GraphXAI attribution missing'
    torch.manual_seed(3)
    model = module.EnrichedPNA(3, 2, torch.tensor([1, 4, 2]), numeric_dim=2, width=8).eval()
    for codes in ([0, 1], []):
        graph = module.numeric_graph(codes, 0, ['med:a', 'cc:b'], [1., -2.])
        result = module.explain_numeric(model, graph, ['age', 'temperature'], steps=4, epochs=3)
        assert set(result['algorithms']) == {'GradExplainer', 'IntegratedGradExplainer', 'GNNExplainer'}
        for value in result['algorithms'].values():
            assert value['status'] == 'success'
        for name in ['GradExplainer', 'IntegratedGradExplainer']:
            importance = np.asarray(result['numeric_attribution'][name])
            assert importance.shape == (2,) and np.isfinite(importance).all()
            assert abs(importance).sum() > 0
        wrapper = GraphXAIWrapper(model, graph.node_type)
        assert torch.equal(wrapper(graph.x, graph.edge_index), model(graph))


@pytest.mark.parametrize('method', ['protgnn', 'gsat'])
def test_comparator_actual_numeric_forward_backward(method):
    module = enriched_model()
    assert hasattr(module, 'build_comparator'), 'Opt-in comparator numeric adapter missing'
    names = ['med:a', 'cc:b']
    model, device = module.build_comparator(method, len(names), 2)
    graphs = [module.numeric_graph([0, 1], i, names, [float(i + 1), -2.]) for i in range(3)]
    batch = Batch.from_data_list(graphs)
    batch.x.requires_grad_(True)
    from comparison.standardized.common_input_improvement import forward_loss
    logits, y, extra = forward_loss(model, method, batch, 0, True, device)
    assert logits.shape == (3, 30) and torch.isfinite(logits).all()
    (torch.nn.functional.cross_entropy(logits, y) + extra).backward()
    numeric_gradient = batch.x.grad[:, -2:][batch.node_type == 2]
    assert torch.isfinite(numeric_gradient).all() and numeric_gradient.abs().sum() > 0
    assert torch.equal(batch.x[:, :3].detach(), Batch.from_data_list([concept_graph([0, 1], i, names) for i in range(3)]).x)
    with pytest.raises(NotImplementedError, match='categorical'):
        module.build_comparator('graphcare', len(names), 2)
