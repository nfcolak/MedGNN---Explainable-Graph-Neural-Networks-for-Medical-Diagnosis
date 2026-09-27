"""Synthetic contract tests for the token-count Wide & Deep plugin."""
from argparse import Namespace

import pytest
import torch
from torch_geometric.data import Batch, Data

from comparison.standardized.clinical_graph_v2 import methods
from comparison.standardized.clinical_graph_v2.methods.base import relation_count


def make(args=None, *, hidden=16, layers=2, dropout=0.3, num_tokens=8,
         node_dim=4, edge_dim=22, num_classes=3, token_dim=4, num_triples=3):
    adapter_type = methods.METHOD_REGISTRY.get("tokenfusion")
    if adapter_type is None:
        pytest.fail("tokenfusion plugin not registered")
    return adapter_type(num_tokens=num_tokens, node_dim=node_dim, edge_dim=edge_dim,
                        num_classes=num_classes, hidden=hidden, layers=layers,
                        dropout=dropout, token_dim=token_dim,
                        num_triples=num_triples, args=args or Namespace())


def graph(tokens, *, node_dim=4, offset=0):
    n = len(tokens)
    x = torch.arange(n * node_dim, dtype=torch.float32).reshape(n, node_dim) / 10 + offset
    edges = torch.tensor([[i for i in range(n - 1)], [i + 1 for i in range(n - 1)]], dtype=torch.long)
    if n <= 1:
        edges = torch.empty((2, 0), dtype=torch.long)
    return Data(x=x, token=torch.tensor(tokens), node_type=torch.zeros(n, dtype=torch.long),
                edge_index=edges, edge_attr=torch.zeros((edges.size(1), 22)),
                edge_relation=torch.zeros(edges.size(1), dtype=torch.long),
                edge_triple=torch.zeros(edges.size(1), dtype=torch.long))


def test_registration_forward_backward_runner_profile_and_parameter_budget():
    adapter = make()
    batch = Batch.from_data_list([graph([1, 2, 2]), graph([3, 4])])
    output = adapter(batch, epoch=0)
    assert output.logits.shape == (2, 3)
    assert output.auxiliary_loss.ndim == 0 and output.auxiliary_loss.requires_grad
    assert all(torch.isfinite(value).all() for value in (output.logits, output.auxiliary_loss))
    (output.logits.sum() + output.auxiliary_loss).backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters())
    assert set(adapter.runner_defaults) == {
        "hidden", "layers", "dropout", "lr", "weight_decay", "batch_size",
        "epochs", "patience", "min_delta"}
    assert adapter.grad_clip_value == 2.0
    assert adapter.early_stopping_start() == 20

    real = make(hidden=adapter.runner_defaults["hidden"], layers=3, dropout=adapter.runner_defaults["dropout"],
                num_tokens=391, node_dim=63, edge_dim=22, num_classes=10,
                token_dim=32, num_triples=16)
    assert sum(p.numel() for p in real.parameters()) <= 405_000


def test_placeholder_until_plugin_is_registered():
    """Keep a useful assertion failure rather than failing on plugin import."""
    assert "tokenfusion" in methods.METHOD_REGISTRY, "tokenfusion plugin not registered"
