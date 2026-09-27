"""Synthetic-only tests for the clinical virtual-node plugin."""
from argparse import Namespace

import pytest
import torch
from torch_geometric.data import Batch, Data

from comparison.standardized.clinical_graph_v2 import methods
from comparison.standardized.clinical_graph_v2.methods.base import relation_count


def make_batch(seed=4):
    generator = torch.Generator().manual_seed(seed)
    graphs = []
    for count in (3, 2):
        graphs.append(Data(
            x=torch.randn(count, 63, generator=generator),
            token=torch.randint(1, 391, (count,), generator=generator),
            node_type=torch.randint(0, 8, (count,), generator=generator),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 22)),
            edge_relation=torch.empty((0,), dtype=torch.long),
            edge_triple=torch.empty((0,), dtype=torch.long),
            y=torch.tensor([0]),
        ))
    return Batch.from_data_list(graphs)


def make(**overrides):
    adapter = methods.METHOD_REGISTRY.get("vnode")
    if adapter is None:
        pytest.fail("vnode plugin not registered")
    return adapter(
        num_tokens=391, node_dim=63, edge_dim=22, num_classes=10,
        hidden=104, layers=3, dropout=0.3, token_dim=32,
        num_triples=16, args=overrides.pop("args", Namespace()), **overrides)


def test_registration_forward_backward_defaults_and_real_dimension_budget():
    model = make()
    batch = make_batch()
    output = model(batch, epoch=0)
    assert output.logits.shape == (2, 10)
    assert output.auxiliary_loss.ndim == 0
    assert output.auxiliary_loss.requires_grad
    assert torch.isfinite(output.logits).all()
    (output.logits.sum() + output.auxiliary_loss).backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.runner_defaults == {
        "hidden": 104, "layers": 3, "dropout": 0.3, "lr": 1.79e-3,
        "weight_decay": 4.3e-5, "batch_size": 128, "epochs": 40,
        "patience": 10, "min_delta": 0.005,
    }
    assert model.grad_clip_value == 2.0
    assert model.early_stopping_start() == 20
    assert sum(p.numel() for p in model.parameters()) <= 405_000
    assert model.optimizer_groups(Namespace()) == [{"params": list(model.parameters())}]
