"""Synthetic standardized GCN smoke, with and without prototypes; no optimizer step."""
import pytest
import torch
from torch_geometric.data import Batch, Data

from protgnn_analysis.config import ModelParser
from protgnn_analysis.models import GnnNets


@pytest.mark.parametrize("prototypes", [False, True])
def test_protgnn_gcn_synthetic_forward_backward(prototypes):
    torch.manual_seed(1234)
    config = ModelParser()
    config.device = "cpu"
    config.latent_dim = [8, 8]
    config.mlp_hidden = [8]
    config.enable_prot = prototypes
    config.num_prototypes_per_class = 2
    model = GnnNets(4, 3, config)
    graphs = [Data(x=torch.randn(3, 4),
                   edge_index=torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]]),
                   y=torch.tensor([label])) for label in (0, 2)]
    batch = Batch.from_data_list(graphs)
    logits, probabilities, *_ = model(batch)
    assert logits.shape == (2, 3)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))
    loss = torch.nn.functional.cross_entropy(logits, batch.y)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
