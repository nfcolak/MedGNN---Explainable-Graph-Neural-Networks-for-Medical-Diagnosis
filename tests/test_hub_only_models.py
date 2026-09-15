"""Standardized edgeless patient hubs through real PyG model paths, no training."""
import json

import pytest
import torch
from torch_geometric.data import Batch, Data

from protgnn_analysis.config import ModelParser
from protgnn_analysis.models import GnnNets
from gsat_analysis.models import ExtractorMLP, GIN, GSAT
from gsat_analysis.explainability.graphxai_wrapper import GSATGraphXAIWrapper
from shared.lib.explanation_contract import build_node_explanation


@pytest.mark.parametrize("training", [False, True])
def test_gsat_non_singleton_batchnorm_exact_equivalence(training):
    from gsat_analysis.models.gin import SingletonSafeBatchNorm1d
    actual = SingletonSafeBatchNorm1d(5).train(training)
    expected = torch.nn.BatchNorm1d(5).train(training)
    expected.load_state_dict(actual.state_dict())
    x = torch.randn(3, 5, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    a, b = actual(x), expected(y)
    assert torch.equal(a, b)
    a.square().sum().backward()
    b.square().sum().backward()
    assert torch.equal(x.grad, y.grad)
    for key, value in actual.state_dict().items():
        assert torch.equal(value, expected.state_dict()[key])


def test_gsat_singleton_does_not_update_running_statistics():
    from gsat_analysis.models.gin import SingletonSafeBatchNorm1d
    norm = SingletonSafeBatchNorm1d(5).train()
    before = {k: v.clone() for k, v in norm.state_dict().items()}
    norm(torch.randn(1, 5, requires_grad=True)).sum().backward()
    assert norm.weight.grad is not None
    assert all(torch.equal(v, norm.state_dict()[k]) for k, v in before.items())


def graphs(mixed):
    hub = Data(x=torch.tensor([[1., 0., 0., 0., 0.]]),
               edge_index=torch.empty((2, 0), dtype=torch.long), y=torch.tensor([0]))
    other = Data(x=torch.eye(5)[:2], edge_index=torch.tensor([[0, 1], [1, 0]]),
                 y=torch.tensor([1]))
    return Batch.from_data_list([hub, other] if mixed else [hub])


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("prototypes", [False, True])
def test_protgnn_hub_only_finite_forward_backward(mixed, prototypes):
    torch.manual_seed(1234)
    config = ModelParser()
    config.device = "cpu"
    config.latent_dim = [8, 8]
    config.mlp_hidden = [8]
    config.enable_prot = prototypes
    config.num_prototypes_per_class = 2
    model = GnnNets(5, 3, config)
    batch = graphs(mixed)
    logits, probabilities, *_ = model(batch)
    assert torch.isfinite(logits).all() and torch.isfinite(probabilities).all()
    torch.nn.functional.cross_entropy(logits, batch.y).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    from protgnn_analysis.explainability.graphxai_wrapper import ProtGNNWrapper
    model.eval()
    hub = graphs(False)
    wrapper = ProtGNNWrapper(model)
    assert torch.equal(wrapper(hub.x, hub.edge_index, hub.batch), model(hub)[0])
    explanation = build_node_explanation(wrapper, hub.x, hub.edge_index, batch=hub.batch)
    json.dumps(explanation, allow_nan=False)


@pytest.mark.parametrize("mixed", [False, True])
def test_gsat_hub_only_finite_forward_backward_and_explanation(mixed):
    torch.manual_seed(1234)
    model = GSAT(GIN(5, 3, hidden_dim=8, num_layers=2, dropout=0., readout="mean"),
                 ExtractorMLP(8, attention_level="node", dropout=0.), attention_level="node")
    batch = graphs(mixed)
    result = model(batch, training=True)
    for name in ("logits", "node_att", "edge_att", "loss", "info_loss"):
        assert torch.isfinite(result[name]).all(), name
    result["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    model.eval()
    hub = graphs(False)
    wrapper = GSATGraphXAIWrapper(model)
    explanation = build_node_explanation(wrapper, hub.x, hub.edge_index, batch=hub.batch)
    json.dumps(explanation, allow_nan=False)
